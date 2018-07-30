import numpy as np
import sys
from pytools import memoize

import pyopencl as cl
import pyopencl.array
import pyopencl.cltypes
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.scan import GenericScanKernel

from pysph.base.gpu_nnps_helper import GPUNNPSHelper
from pysph.base.opencl import DeviceArray
from pysph.cpy.opencl import get_context, get_queue, profile_kernel, \
    named_profile
from pysph.base.tree.helpers import cache_result, ParticleArrayWrapper, \
    ctype_to_dtype

from mako.template import Template

# For Mako
disable_unicode = False if sys.version_info.major > 2 else True

NODE_KERNEL_TEMPLATE = r"""
uint node_idx = i;
%(setup)s;
int child_offset = offsets[node_idx];
if (child_offset == -1) {
    uint2 pbound = pbounds[node_idx];
    if (pbound.s0 < pbound.s1) {
        %(leaf_operation)s;
    }
} else {
    %(node_operation)s;
}
%(output_expr)s;
"""

DFS_TEMPLATE = r"""
    /*
     * Owner of properties -
     * dst octree: cids, unique_cids
     * src octree: offsets
     */
   int cid_dst = unique_cids[i];

    /*
     * Assuming max depth of 21
     * stack_idx is also equal to current layer of octree
     * child_idx = number of children iterated through
     * idx_stack = current node
     */
    char child_stack[21];
    int cid_stack[21];

    char idx = 0;
    child_stack[0] = 0;
    cid_stack[0] = 1;
    char flag;
    int cid_src;
    int child_offset;

    %(setup)s;
    while (idx >= 0) {

        // Recurse to find either leaf node or invalid node
        cid_src = cid_stack[idx];

        child_offset = offsets[cid_src];
        %(common_operation)s;

        while (child_offset != -1) {
            %(node_operation)s;

            idx++;
            cid_src = child_offset;
            cid_stack[idx] = cid_src;
            child_stack[idx] = 0;
            child_offset = offsets[cid_src];
        }

        if (child_offset == -1) {
            %(leaf_operation)s;
        }

        // Recurse back to find node with a valid neighbor
        while (child_stack[idx] >= 7 && idx >= 0)
            idx--;

        // Iterate to next neighbor
        if (idx >= 0) {
            cid_stack[idx]++;
            child_stack[idx]++;
        }
    }

    %(output_expr)s;
"""


# `(sfc[i] & mask) >> rshift` directly gives the octant (0-7) a particle
# belongs to in a given layer
# The segmented scan generates an array of vectors. Let's call this array v.
# Then `v[i][j]` gives the number of particles before or at index `i` which are
# going to be in octant `j` or an octant lesser than `j`.
# Eg. v[23][4] gives the particles indexed between 0 and 23 which are going
# to be in an octant from 0-4
@named_profile('particle_reordering')
@memoize
def _get_particle_kernel(ctx):
    return GenericScanKernel(
        ctx, cl.cltypes.uint8, neutral="0",
        arguments=r"""__global ulong *sfc,
                    __global char *seg_flag,
                    __global uint8 *octant_vector,
                    ulong mask, char rshift
                    """,
        input_expr="eye[eye_index(sfc[i], mask, rshift)]",
        scan_expr="(across_seg_boundary ? b : a + b)",
        is_segment_start_expr="seg_flag[i]",
        output_statement=r"""octant_vector[i]=item;""",
        preamble=r"""uint8 constant eye[8] = {
                    (uint8)(1, 1, 1, 1, 1, 1, 1, 1),
                    (uint8)(0, 1, 1, 1, 1, 1, 1, 1),
                    (uint8)(0, 0, 1, 1, 1, 1, 1, 1),
                    (uint8)(0, 0, 0, 1, 1, 1, 1, 1),
                    (uint8)(0, 0, 0, 0, 1, 1, 1, 1),
                    (uint8)(0, 0, 0, 0, 0, 1, 1, 1),
                    (uint8)(0, 0, 0, 0, 0, 0, 1, 1),
                    (uint8)(0, 0, 0, 0, 0, 0, 0, 1),
                    };

                    inline char eye_index(ulong sfc, ulong mask, char rshift) {
                        return ((sfc & mask) >> rshift);
                    }
                    """
    )


# The offset of a node's child is given by:
# offset of first child in next layer + 8 * (number of non-leaf nodes before
# given node).
# If the node is a leaf, we set this value to be -1.
@named_profile('set_offset')
@memoize
def _get_set_offset_kernel(ctx, leaf_size):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""__global uint2 *pbounds, __global uint *offsets,
                      __global int *leaf_count, int csum_nodes_next""",
        input_expr="(pbounds[i].s1 - pbounds[i].s0 > %(leaf_size)s)" % {
            'leaf_size': leaf_size},
        scan_expr="a + b",
        output_statement=r"""{
            offsets[i] = ((pbounds[i].s1 - pbounds[i].s0 > %(leaf_size)s) ?
                           csum_nodes_next + (8 * (item - 1)) : -1);
            if (i == N - 1) { *leaf_count = (N - item); }
        }""" % {'leaf_size': leaf_size}
    )


# Each particle belongs to a given cell and the cell is indexed as cid.
# The unique_cids algorithm
# 1) unique_cids_map: unique_cids_map[i] gives the number of unique_cids
#                     before index i.
# 2) unique_cids: List of unique cids.
# 3) unique_cids_count: Number of unique cids
@named_profile('unique_cids')
@memoize
def _get_unique_cids_kernel(ctx):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""int *cids, int *unique_cids_map,
                int *unique_cids, int *unique_cids_count""",
        input_expr="(i == 0 || cids[i] != cids[i-1])",
        scan_expr="a + b",
        output_statement=r"""
            if (item != prev_item) {
                unique_cids[item - 1] = cids[i];
            }
            unique_cids_map[i] = item - 1;
            if (i == N - 1) *unique_cids_count = item;
        """
    )


@named_profile("group_cids")
@memoize
def _get_cid_groups_kernel(ctx):
    return GenericScanKernel(
        ctx, np.uint32, neutral="0",
        arguments="""int *unique_cids, uint2 *pbounds,
            int *group_cids, int *group_count, int gmin, int gmax""",
        input_expr="pass(pbounds[unique_cids[i]], gmin, gmax)",
        scan_expr="(a + b)",
        output_statement=r"""
        if (item != prev_item) {
            group_cids[item - 1] = unique_cids[i];
        }
        if (i == N - 1) *group_count = item;
        """,
        preamble="""
        char pass(uint2 pbound, int gmin, int gmax) {
            int leaf_size = pbound.s1 - pbound.s0;
            return (leaf_size > gmin && leaf_size <= gmax);
        }
        """
    )


@memoize
def get_helper(ctx, src_file, c_type):
    # ctx and c_type are the only parameters that
    # change here
    return GPUNNPSHelper(ctx, src_file,
                         c_type=c_type)


def get_dtype_type(use_double):
    if use_double:
        return np.float64
    else:
        return np.float32


def get_dtype_str(use_double):
    if use_double:
        return 'double'
    else:
        return 'float'


def get_leaf_sizes(offsets, pbounds):
    offsets = offsets.get()
    pbounds = pbounds.get()
    is_leaf = offsets & (pbounds['y'] - pbounds['x'] < 32)
    p25 = is_leaf & (pbounds['y'] - pbounds['x'] < 8)
    p50 = is_leaf & (pbounds['y'] - pbounds['x'] < 16)
    p75 = is_leaf & (pbounds['y'] - pbounds['x'] < 24)

    print(np.sum(p25) / np.sum(is_leaf),
          np.sum(p50 & ~p25) / np.sum(is_leaf),
          np.sum(p75 & ~p50) / np.sum(is_leaf),
          np.sum(is_leaf & ~p75) / np.sum(is_leaf))


class OctreeGPU(object):
    def __init__(self, pa, radius_scale=1.0,
                 use_double=False, leaf_size=32, c_type='float',
                 src_file='tree/point_octree.mako'):
        self.c_type = c_type
        self.c_type_src = 'double' if use_double else 'float'
        self.pa = ParticleArrayWrapper(pa, self.c_type_src,
                                       self.c_type)

        self.radius_scale = radius_scale
        self.use_double = use_double
        self.ctx = get_context()
        self.queue = get_queue()
        self.sorted = False
        self.helper = get_helper(self.ctx, src_file, self.c_type)

        if c_type == 'half':
            self.makeo_vec = cl.array.vec.make_half3
        elif c_type == 'float':
            self.make_vec = cl.array.vec.make_float3
        elif c_type == 'double':
            self.make_vec = cl.array.vec.make_double3

        self.initialized = False

        norm2 = \
            """
            #define NORM2(X, Y, Z) ((oX)*(X) + (Y)*(Y) + (Z)*(Z))
            """

        self.preamble = norm2
        self.depth = 0
        self.leaf_size = leaf_size

    ##########################################################################
    # Octree construction algorithms
    ##########################################################################
    def refresh(self, xmin, xmax, hmin, fixed_depth=None):
        self.pa.sync()
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.hmin = hmin

        # Convert width of domain to a power of 2 multiple of cell size
        # (Optimal width for cells)
        cell_size = self.hmin * self.radius_scale * (1. + 1e-5)
        max_width = np.max(self.xmax - self.xmin)

        new_width = cell_size * \
                    2. ** int(np.ceil(np.log2(max_width / cell_size)))

        diff = (new_width - (self.xmax - self.xmin)) / 2

        self.xmin -= diff
        self.xmax += diff

        self._calc_cell_size_and_depth()

        if not self.initialized:
            self._initialize_data()
        else:
            self._reinitialize_data()

        self._bin()
        self._build_tree(fixed_depth)

    def _calc_cell_size_and_depth(self):
        self.cell_size = self.hmin * self.radius_scale * (1. + 1e-3)
        self.cell_size /= 128
        max_width = max((self.xmax[i] - self.xmin[i]) for i in range(3))
        self.max_depth = int(np.ceil(np.log2(max_width / self.cell_size))) + 1

    def _initialize_data(self):
        self.sorted = False
        num_particles = self.pa.get_number_of_particles()
        self.pids = DeviceArray(np.uint32, n=num_particles)
        self.pid_keys = DeviceArray(np.uint64, n=num_particles)
        self.cids = DeviceArray(np.uint32, n=num_particles)
        self.cids.fill(0)

        # Filled after tree built
        self.pbounds = None
        self.offsets = None
        self.initialized = True

    def _reinitialize_data(self):
        self.sorted = False
        num_particles = self.pa.get_number_of_particles()
        self.pids.resize(num_particles)
        self.pid_keys.resize(num_particles)
        self.cids.resize(num_particles)
        self.cids.fill(0)

        # Filled after tree built
        self.pbounds = None
        self.offsets = None

    def _bin(self):
        dtype = ctype_to_dtype(self.c_type)
        fill_particle_data = self.helper.get_kernel("fill_particle_data")
        pa_gpu = self.pa.gpu
        fill_particle_data(pa_gpu.x, pa_gpu.y, pa_gpu.z,
                           dtype(self.cell_size),
                           self.make_vec(
                               self.xmin[0], self.xmin[1], self.xmin[2]),
                           self.pid_keys.array, self.pids.array)

    # A little bit of manual book-keeping for temporary variables
    # We could instead just allocate new arrays for each iteration of
    # the build step and let the GC take care of stuff but this is probably
    # a better approach to save on memory
    def _create_temp_vars(self, temp_vars):
        n = self.pa.get_number_of_particles()
        temp_vars['pids'] = DeviceArray(np.uint32, n)
        temp_vars['pid_keys'] = DeviceArray(np.uint64, n)
        temp_vars['cids'] = DeviceArray(np.uint32, n)

    def _exchange_temp_vars(self, temp_vars):
        for k in temp_vars.keys():
            t = temp_vars[k]
            temp_vars[k] = getattr(self, k)
            setattr(self, k, t)

    def _clean_temp_vars(self, temp_vars):
        for k in list(temp_vars.keys()):
            del temp_vars[k]

    def _reorder_particles(self, depth, octants, offsets_parent,
                           pbounds_parent,
                           seg_flag, csum_nodes_prev, temp_vars):
        rshift = np.uint8(3 * (self.max_depth - depth - 1))
        mask = np.uint64(7 << rshift)

        particle_kernel = _get_particle_kernel(self.ctx)

        particle_kernel(self.pid_keys.array, seg_flag.array,
                        octants.array, mask, rshift)

        reorder_particles = self.helper.get_kernel('reorder_particles')
        reorder_particles(self.pids.array, self.pid_keys.array,
                          self.cids.array,
                          seg_flag.array,
                          pbounds_parent.array, offsets_parent.array,
                          octants.array,
                          temp_vars['pids'].array, temp_vars['pid_keys'].array,
                          temp_vars['cids'].array,
                          mask, rshift,
                          np.uint32(csum_nodes_prev))
        self._exchange_temp_vars(temp_vars)

    def _update_node_data(self, offsets_prev, pbounds_prev, offsets, pbounds,
                          seg_flag, octants, csum_nodes, csum_nodes_next, n):
        """Update node data and return number of children which are leaves."""

        # Update particle-related data of children
        set_node_data = self.helper.get_kernel("set_node_data")
        set_node_data(offsets_prev.array, pbounds_prev.array,
                      offsets.array, pbounds.array,
                      seg_flag.array, octants.array, np.uint32(csum_nodes),
                      np.uint32(n))

        # Set children offsets
        leaf_count = DeviceArray(np.uint32, 1)
        set_offsets = _get_set_offset_kernel(self.ctx, self.leaf_size)
        set_offsets(pbounds.array, offsets.array, leaf_count.array,
                    np.uint32(csum_nodes_next))
        return leaf_count.array[0].get()

    def _compress_layers(self, offsets_temp, pbounds_temp):
        curr_offset = 0
        total_nodes = 0

        for i in range(self.depth + 1):
            total_nodes += self.num_nodes[i]

        self.offsets = DeviceArray(np.int32, total_nodes)
        self.pbounds = DeviceArray(cl.cltypes.uint2, total_nodes)

        append_layer = self.helper.get_kernel('append_layer')

        self.total_nodes = total_nodes
        for i in range(self.depth + 1):
            append_layer(
                offsets_temp[i].array, pbounds_temp[i].array,
                self.offsets.array, self.pbounds.array,
                np.int32(curr_offset), np.uint8(i == self.depth)
            )
            curr_offset += self.num_nodes[i]

    def _get_unique_cids_and_count(self):
        n = self.pa.get_number_of_particles()
        self.unique_cids = DeviceArray(np.uint32, n=n)
        self.unique_cids_map = DeviceArray(np.uint32, n=n)
        uniq_count = DeviceArray(np.uint32, n=1)
        unique_cids_kernel = _get_unique_cids_kernel(self.ctx)
        unique_cids_kernel(self.cids.array, self.unique_cids_map.array,
                           self.unique_cids.array, uniq_count.array)
        self.unique_cid_count = uniq_count.array[0].get()

    def _build_tree(self, fixed_depth=None):
        """Build octree
        """
        num_leaves_here = 0
        n = self.pa.get_number_of_particles()
        temp_vars = {}
        csum_nodes_prev = 0
        csum_nodes = 1
        self.depth = 0
        self.num_nodes = [1]

        # Initialize temporary data (but persistent across layers)
        self._create_temp_vars(temp_vars)

        octants = DeviceArray(cl.cltypes.uint8, n)

        seg_flag = DeviceArray(cl.cltypes.char, n)
        seg_flag.fill(0)
        seg_flag.array[0] = 1

        offsets_temp = [DeviceArray(np.int32, 1)]
        offsets_temp[-1].fill(1)

        pbounds_temp = [DeviceArray(cl.cltypes.uint2, 1)]
        pbounds_temp[-1].array[0].set(cl.cltypes.make_uint2(0, n))

        loop_lim = min(self.max_depth if fixed_depth is None else fixed_depth,
                       20)

        for depth in range(1, loop_lim):
            num_nodes = 8 * (self.num_nodes[-1] - num_leaves_here)
            if num_nodes == 0:
                break
            else:
                self.depth += 1
            self.num_nodes.append(num_nodes)

            # Allocate new layer
            offsets_temp.append(DeviceArray(np.int32, self.num_nodes[-1]))
            pbounds_temp.append(DeviceArray(cl.cltypes.uint2,
                                            self.num_nodes[-1]))

            self._reorder_particles(depth, octants, offsets_temp[-2],
                                    pbounds_temp[-2], seg_flag,
                                    csum_nodes_prev,
                                    temp_vars)
            num_leaves_here = self._update_node_data(
                offsets_temp[-2], pbounds_temp[-2],
                offsets_temp[-1], pbounds_temp[-1],
                seg_flag, octants,
                csum_nodes, csum_nodes + self.num_nodes[-1], n
            )

            csum_nodes_prev = csum_nodes
            csum_nodes += self.num_nodes[-1]

        self._compress_layers(offsets_temp, pbounds_temp)
        self._get_unique_cids_and_count()
        self._clean_temp_vars(temp_vars)
        # get_leaf_sizes(self.offsets.array, self.pbounds.array)
    ###########################################################################
    # Octree API
    ###########################################################################
    def allocate_node_prop(self, dtype):
        return DeviceArray(dtype, self.total_nodes)

    def allocate_leaf_prop(self, dtype):
        return DeviceArray(dtype, int(self.unique_cid_count))

    def get_preamble(self):
        if self.sorted:
            return "#define PID(idx) (idx)"
        else:
            return "#define PID(idx) (pids[idx])"

    ###########################################################################
    # General Octree Algorithms
    ###########################################################################
    def tree_bottom_up(self, args, setup, leaf_operation, node_operation,
                       output_expr):
        @cache_result('_tree_bottom_up')
        def _tree_bottom_up(ctx, data_t, sorted, args, setup,
                            leaf_operation, node_operation, output_expr):

            if sorted:
                preamble = "#define PID(idx) (idx)"
            else:
                preamble = "#define PID(idx) (pids[idx])"

            operation = Template(
                NODE_KERNEL_TEMPLATE % dict(setup=setup,
                                            leaf_operation=leaf_operation,
                                            node_operation=node_operation,
                                            output_expr=output_expr),
                disable_unicode=disable_unicode
            ).render(data_t=data_t, sorted=sorted)

            args = Template(
                "int *offsets, uint2 *pbounds, " + args,
                disable_unicode=disable_unicode
            ).render(data_t=data_t, sorted=sorted)

            kernel = ElementwiseKernel(
                ctx, args, operation=operation, preamble=preamble)

            def callable(octree, *args):
                csum_nodes = octree.total_nodes
                out = None
                for i in range(octree.depth, -1, -1):
                    csum_nodes_next = csum_nodes
                    csum_nodes -= octree.num_nodes[i]
                    out = kernel(octree.offsets.array, octree.pbounds.array,
                                 *args,
                                 slice=slice(csum_nodes, csum_nodes_next))
                return out

            return callable

        return _tree_bottom_up(self.ctx, self.c_type, self.sorted, args, setup,
                               leaf_operation, node_operation, output_expr)

    def get_leaf_size_partitions(self, group_min, group_max):
        """Partition leaves based on leaf size

        Parameters
        ----------
        group_min
            Minimum leaf size
        group_max
            Maximum leaf size
        Returns
        -------
        groups : DeviceArray
            An array which contains the cell ids of leaves
            with leaf size > group_min and leaf size <= group_max
        group_count : int
            The number of leaves which satisfy the given condition
            on the leaf size
        """
        groups = DeviceArray(np.uint32, int(self.unique_cid_count))
        group_count = DeviceArray(np.uint32, 1)

        get_cid_groups = _get_cid_groups_kernel(self.ctx)
        get_cid_groups(self.unique_cids.array[:self.unique_cid_count],
                       self.pbounds.array, groups.array, group_count.array,
                       np.int32(group_min), np.int32(group_max))
        result = groups, int(group_count.array[0].get())
        return result

    ###########################################################################
    # Point Octree Algorithms
    ###########################################################################
    def leaf_tree_traverse(self, args, setup, node_operation, leaf_operation,
                           output_expr, common_operation=""):
        """
        Traverse this (source) octree. One thread for each leaf of
        destination octree.
        """

        @cache_result('_leaf_tree_traverse')
        def _leaf_tree_traverse(ctx, data_t, sorted, args, setup,
                                node_operation, leaf_operation,
                                output_expr, common_operation):
            if sorted:
                preamble = "#define PID(idx) (idx)"
            else:
                preamble = "#define PID(idx) (pids[idx])"
            premable = preamble + Template("""
                #define IN_BOUNDS(X, MIN, MAX) ((X >= MIN) && (X < MAX))
                #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
                #define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
                #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
                #define AVG(X, Y) (((X) + (Y)) / 2)
                #define ABS(X) ((X) > 0 ? (X) : -(X))
                #define EPS 1e-6f
                #define INF 1e6
                #define SQR(X) ((X) * (X))

                char contains(private ${data_t} *n1, private ${data_t} *n2) {
                    // Check if node n1 contains node n2
                    char res = 1;
                    %for i in range(3):
                        res = res && (n1[${i}] <= n2[${i}]) &&
                              (n1[3 + ${i}] >= n2[3 + ${i}]);
                    %endfor

                    return res;
                }

                char contains_search(private ${data_t} *n1,
                                     private ${data_t} *n2) {
                    // Check if node n1 contains node n2 with n1 having
                    // its search radius extension
                    ${data_t} h = n1[6];
                    char res = 1;
                    %for i in range(3):
                        res = res & (n1[${i}] - h - EPS <= n2[${i}]) &
                              (n1[3 + ${i}] + h + EPS >= n2[3 + ${i}]);
                    %endfor

                    return res;
                }

                char intersects(private ${data_t} *n1, private ${data_t} *n2) {
                    // Check if node n1 'intersects' node n2
                    ${data_t} cdist;
                    ${data_t} w1, w2, wavg = 0;
                    char res = 1;
                    ${data_t} h = MAX(n1[6], n2[6]);

                    % for i in range(3):
                        cdist = fabs((n1[${i}] + n1[3 + ${i}]) / 2 -
                                     (n2[${i}] + n2[3 + ${i}]) / 2);
                        w1 = fabs(n1[${i}] - n1[3 + ${i}]);
                        w2 = fabs(n2[${i}] - n2[3 + ${i}]);
                        wavg = AVG(w1, w2);
                        res &= (cdist - wavg <= h + EPS);
                    % endfor

                    return res;
                }
            """, disable_unicode=disable_unicode).render(data_t=data_t)

            operation = Template(
                DFS_TEMPLATE % dict(setup=setup,
                                    leaf_operation=leaf_operation,
                                    node_operation=node_operation,
                                    common_operation=common_operation,
                                    output_expr=output_expr),
                disable_unicode=disable_unicode
            ).render(data_t=data_t, sorted=sorted)

            args = Template(
                "int *unique_cids, int *cids, int *offsets, " + args,
                disable_unicode=disable_unicode
            ).render(data_t=data_t, sorted=sorted)

            kernel = ElementwiseKernel(
                ctx, args, operation=operation, preamble=premable)

            def callable(octree_src, octree_dst, *args):
                return kernel(
                    octree_dst.unique_cids.array[:octree_dst.unique_cid_count],
                    octree_dst.cids.array, octree_src.offsets.array, *args
                )

            return callable

        return _leaf_tree_traverse(self.ctx, self.c_type, self.sorted, args,
                                   setup, node_operation, leaf_operation,
                                   output_expr, common_operation)

    def _set_node_bounds(self):
        if self.c_type == 'half':
            data_t3 = cl.cltypes.half3
        elif self.c_type == 'float':
            data_t3 = cl.cltypes.float3
        else:
            data_t3 = cl.cltypes.double3
        data_t = self.c_type

        self.node_xmin = self.allocate_node_prop(data_t3)
        self.node_xmax = self.allocate_node_prop(data_t3)
        self.node_hmax = self.allocate_node_prop(data_t)

        # TODO: Recheck EPS 1e-6 here
        set_node_bounds = self.tree_bottom_up(
            setup=r"""
                ${data_t} xmin[3] = {1e6, 1e6, 1e6};
                ${data_t} xmax[3] = {-1e6, -1e6, -1e6};
                ${data_t} hmax = 0;
                """,
            args="""int *pids, ${data_t} *x, ${data_t} *y, ${data_t} *z,
            ${data_t} *h,
            ${data_t} radius_scale,
            ${data_t}3 *node_xmin, ${data_t}3 *node_xmax,
            ${data_t} *node_hmax""",
            leaf_operation="""
                <% ch = ['x', 'y', 'z'] %>
                for (int j=pbound.s0; j < pbound.s1; j++) {
                    int pid = PID(j);
                    % for d in range(3):
                        xmin[${d}] = fmin(xmin[${d}], ${ch[d]}[pid] - 1e-6f);
                        xmax[${d}] = fmax(xmax[${d}], ${ch[d]}[pid] + 1e-6f);
                    % endfor
                    hmax = fmax(h[pid] * radius_scale, hmax);
                }
                """,
            node_operation="""
                % for i in range(8):
                    % for d in range(3):
                        xmin[${d}] = fmin(
                            xmin[${d}], node_xmin[child_offset + ${i}].s${d}
                        );
                        xmax[${d}] = fmax(
                            xmax[${d}], node_xmax[child_offset + ${i}].s${d}
                        );
                    % endfor
                    hmax = fmax(hmax, node_hmax[child_offset + ${i}]);
                % endfor
                """,
            output_expr="""
                % for d in range(3):
                    node_xmin[node_idx].s${d} = xmin[${d}];
                    node_xmax[node_idx].s${d} = xmax[${d}];
                % endfor
                node_hmax[node_idx] = hmax;
                """
        )
        set_node_bounds = profile_kernel(set_node_bounds, 'set_node_bounds')
        pa_gpu = self.pa.gpu
        dtype = ctype_to_dtype(self.c_type)
        set_node_bounds(self, self.pids.array, pa_gpu.x, pa_gpu.y, pa_gpu.z,
                        pa_gpu.h,
                        dtype(self.radius_scale),
                        self.node_xmin.array, self.node_xmax.array,
                        self.node_hmax.array)

    def _sort(self):
        """Set octree as being sorted

        The particle array needs to be aligned by the caller!
        """
        if not self.sorted:
            self.pa.force_sync()
            self.sorted = 1
