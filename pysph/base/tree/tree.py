import numpy as np
from pytools import memoize

import pyopencl as cl
import pyopencl.cltypes
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.scan import GenericScanKernel

from pysph.base.opencl import DeviceArray
from pysph.cpy.opencl import get_context, get_queue, named_profile
from pysph.base.tree.helpers import get_vector_dtype, get_helper

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
    char child_stack[%(max_depth)s];
    int cid_stack[%(max_depth)s];

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
        while (child_stack[idx] >= %(powdim)s-1 && idx >= 0)
            idx--;

        // Iterate to next neighbor
        if (idx >= 0) {
            cid_stack[idx]++;
            child_stack[idx]++;
        }
    }

    %(output_expr)s;
"""


def get_eye_str(k):
    result = "uint%(k)s constant eye[%(k)s]  = {" % dict(k=k)
    for i in range(k):
        result += "(uint%(k)s)(%(init_value)s)," % dict(
            k=k, init_value=','.join((np.arange(k) >= i).astype(
                int).astype(str))
        )
    result += "};"
    return result


# `(sfc[i] & mask) >> rshift` directly gives the octant (0-7) a particle
# belongs to in a given layer
# The segmented scan generates an array of vectors. Let's call this array v.
# Then `v[i][j]` gives the number of particles before or at index `i` which are
# going to be in octant `j` or an octant lesser than `j`.
# Eg. v[23][4] gives the particles indexed between 0 and 23 which are going
# to be in an octant from 0-4
@named_profile('particle_reordering')
@memoize
def _get_particle_kernel(ctx, k, args, index_code):
    return GenericScanKernel(
        ctx, get_vector_dtype('uint', k), neutral="0",
        arguments=r"""__global char *seg_flag,
                    __global uint%(k)s *octant_vector,
                    """ % dict(k=k) + args,
        input_expr="eye[%(index_code)s]" % dict(index_code=index_code),
        scan_expr="(across_seg_boundary ? b : a + b)",
        is_segment_start_expr="seg_flag[i]",
        output_statement=r"""octant_vector[i]=item;""",
        preamble=get_eye_str(k)
    )


# The offset of a node's child is given by:
# offset of first child in next layer + 8 * (number of non-leaf nodes before
# given node).
# If the node is a leaf, we set this value to be -1.
@named_profile('set_offset')
@memoize
def _get_set_offset_kernel(ctx, k, leaf_size):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""__global uint2 *pbounds, __global uint *offsets,
                      __global int *leaf_count, int csum_nodes_next""",
        input_expr="(pbounds[i].s1 - pbounds[i].s0 > %(leaf_size)s)" % {
            'leaf_size': leaf_size},
        scan_expr="a + b",
        output_statement=r"""{
            offsets[i] = ((pbounds[i].s1 - pbounds[i].s0 > %(leaf_size)s) ?
                           csum_nodes_next + (%(k)s * (item - 1)) : -1);
            if (i == N - 1) { *leaf_count = (N - item); }
        }""" % {'leaf_size': leaf_size, 'k': k}
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


@named_profile('leaves')
@memoize
def _get_leaves_kernel(ctx, leaf_size):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments="int *offsets, uint2 pbounds, int *leaf_cids, "
                  "int *num_leaves",
        input_expr="(pbounds[i].s1 - pbounds[i].s0 <= %(leaf_size)s)" % dict(
            leaf_size=leaf_size
        ),
        scan_expr="a+b",
        output_statement=r"""
            if (item != prev_item) {
                leaf_cids[item - 1] = i;
            }
            if (i == N - 1) *num_leaves = item;
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
def tree_bottom_up(ctx, args, setup, leaf_operation, node_operation,
                   output_expr, preamble=""):
    operation = NODE_KERNEL_TEMPLATE % dict(
        setup=setup,
        leaf_operation=leaf_operation,
        node_operation=node_operation,
        output_expr=output_expr
    )

    args = ', '.join(["int *offsets, uint2 *pbounds", args])

    kernel = ElementwiseKernel(ctx, args, operation=operation,
                               preamble=preamble)

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


@memoize
def leaf_tree_traverse(ctx, k, args, setup, node_operation, leaf_operation,
                       output_expr, common_operation, preamble=""):
    operation = DFS_TEMPLATE % dict(
        setup=setup,
        leaf_operation=leaf_operation,
        node_operation=node_operation,
        common_operation=common_operation,
        output_expr=output_expr,
        max_depth=21,
        powdim=k
    )

    args = ', '.join(["int *unique_cids, int *cids, int *offsets", args])

    kernel = ElementwiseKernel(
        ctx, args, operation=operation, preamble=preamble)

    def callable(octree_src, octree_dst, *args):
        return kernel(
            octree_dst.unique_cids.array[:octree_dst.unique_cid_count],
            octree_dst.cids.array, octree_src.offsets.array, *args
        )

    return callable


class Tree(object):
    def __init__(self, n, k=8, leaf_size=32):
        """k-ary tree"""
        self.ctx = get_context()
        self.queue = get_queue()
        self.sorted = False
        self.main_helper = get_helper(self.ctx, 'tree/tree.mako')

        self.initialized = False
        self.preamble = ""
        self.depth = 0
        self.leaf_size = leaf_size
        self.k = k
        self.n = n
        self.sorted = False
        self.depth = 0

        self.data_vars = []
        self.data_var_ctypes = []
        self.data_var_dtypes = []
        self.const_vars = []
        self.const_var_ctypes = []
        self.index_code = ""
        self.initialized = False
        self.set_vars()

    def set_vars(self):
        raise NotImplementedError

    def get_data_args(self):
        return [getattr(self, v) for v in self.data_vars]

    def get_index_constants(self, depth):
        raise NotImplementedError

    def _initialize_data(self):
        self.sorted = False
        num_particles = self.n
        self.pids = DeviceArray(np.uint32, n=num_particles)
        self.cids = DeviceArray(np.uint32, n=num_particles)
        self.cids.fill(0)

        for var, dtype in zip(self.data_vars, self.data_var_dtypes):
            setattr(self, var, DeviceArray(dtype, n=num_particles))

        # Filled after tree built
        self.pbounds = None
        self.offsets = None
        self.initialized = True

    def _reinitialize_data(self):
        self.sorted = False
        num_particles = self.n
        self.pids.resize(num_particles)
        self.cids.resize(num_particles)
        self.cids.fill(0)

        for var in self.data_vars:
            getattr(self, var).resize(num_particles)

        # Filled after tree built
        self.pbounds = None
        self.offsets = None

    def _setup_build(self):
        if not self.initialized:
            self._initialize_data()
        else:
            self._reinitialize_data()

    def _build(self, fixed_depth=None):
        self._build_tree(fixed_depth)

    ###########################################################################
    # Core construction algorithm and helper functions
    ###########################################################################

    # A little bit of manual book-keeping for temporary variables
    # We could instead just allocate new arrays for each iteration of
    # the build step and let the GC take care of stuff but this is probably
    # a better approach to save on memory
    def _create_temp_vars(self, temp_vars):
        n = self.n
        temp_vars['pids'] = DeviceArray(np.uint32, n)
        for var, dtype in zip(self.data_vars, self.data_var_dtypes):
            temp_vars[var] = DeviceArray(dtype, n)
        temp_vars['cids'] = DeviceArray(np.uint32, n)

    def _exchange_temp_vars(self, temp_vars):
        for k in temp_vars.keys():
            t = temp_vars[k]
            temp_vars[k] = getattr(self, k)
            setattr(self, k, t)

    def _clean_temp_vars(self, temp_vars):
        for k in list(temp_vars.keys()):
            del temp_vars[k]

    def _get_temp_data_args(self, temp_vars):
        result = [temp_vars[v] for v in self.data_vars]
        return result

    def _reorder_particles(self, depth, octants, offsets_parent,
                           pbounds_parent,
                           seg_flag, csum_nodes_prev, temp_vars):
        # Scan

        args = [('__global ' + ctype + ' *' + v) for v, ctype in
                zip(self.data_vars, self.data_var_ctypes)]
        args += [(ctype + ' ' + v) for v, ctype in
                 zip(self.const_vars, self.const_var_ctypes)]
        args = ', '.join(args)

        particle_kernel = _get_particle_kernel(self.ctx, self.k,
                                               args, self.index_code)
        args = [seg_flag.array, octants.array]
        args += [x.array for x in self.get_data_args()]
        args += self.get_index_constants(depth)
        particle_kernel(*args)

        # Reorder particles
        reorder_particles = self.main_helper.get_kernel(
            'reorder_particles', k=self.k, data_vars=tuple(self.data_vars),
            data_var_ctypes=tuple(self.data_var_ctypes),
            const_vars=tuple(self.const_vars),
            const_var_ctypes=tuple(self.const_var_ctypes),
            index_code=self.index_code
        )

        args = [self.pids.array, self.cids.array,
                seg_flag.array,
                pbounds_parent.array, offsets_parent.array,
                octants.array,
                temp_vars['pids'].array, temp_vars['cids'].array]
        args += [x.array for x in self.get_data_args()]
        args += [x.array for x in self._get_temp_data_args(temp_vars)]
        args += self.get_index_constants(depth)
        args += [np.uint32(csum_nodes_prev)]

        reorder_particles(*args)
        self._exchange_temp_vars(temp_vars)

    def _compress_layers(self, offsets_temp, pbounds_temp):
        curr_offset = 0
        total_nodes = 0

        for i in range(self.depth + 1):
            total_nodes += self.num_nodes[i]

        self.offsets = DeviceArray(np.int32, total_nodes)
        self.pbounds = DeviceArray(cl.cltypes.uint2, total_nodes)

        append_layer = self.main_helper.get_kernel('append_layer')

        self.total_nodes = total_nodes
        for i in range(self.depth + 1):
            append_layer(
                offsets_temp[i].array, pbounds_temp[i].array,
                self.offsets.array, self.pbounds.array,
                np.int32(curr_offset), np.uint8(i == self.depth)
            )
            curr_offset += self.num_nodes[i]

    def _update_node_data(self, offsets_prev, pbounds_prev, offsets, pbounds,
                          seg_flag, octants, csum_nodes, csum_nodes_next, n):
        """Update node data and return number of children which are leaves."""

        # Update particle-related data of children
        set_node_data = self.main_helper.get_kernel("set_node_data", k=self.k)
        set_node_data(offsets_prev.array, pbounds_prev.array,
                      offsets.array, pbounds.array,
                      seg_flag.array, octants.array, np.uint32(csum_nodes),
                      np.uint32(n))

        # Set children offsets
        leaf_count = DeviceArray(np.uint32, 1)
        set_offsets = _get_set_offset_kernel(self.ctx, self.k, self.leaf_size)
        set_offsets(pbounds.array, offsets.array, leaf_count.array,
                    np.uint32(csum_nodes_next))
        return leaf_count.array[0].get()

    def _build_tree(self, fixed_depth=None):
        """Build octree
        """
        num_leaves_here = 0
        n = self.n
        temp_vars = {}
        csum_nodes_prev = 0
        csum_nodes = 1
        self.depth = 0
        self.num_nodes = [1]

        # Initialize temporary data (but persistent across layers)
        self._create_temp_vars(temp_vars)

        octants = DeviceArray(get_vector_dtype('uint', self.k), n)

        seg_flag = DeviceArray(cl.cltypes.char, n)
        seg_flag.fill(0)
        seg_flag.array[0] = 1

        offsets_temp = [DeviceArray(np.int32, 1)]
        offsets_temp[-1].fill(1)

        pbounds_temp = [DeviceArray(cl.cltypes.uint2, 1)]
        pbounds_temp[-1].array[0].set(cl.cltypes.make_uint2(0, n))

        loop_lim = min(fixed_depth, 20)

        for depth in range(1, loop_lim):
            num_nodes = self.k * (self.num_nodes[-1] - num_leaves_here)
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
        self._clean_temp_vars(temp_vars)

    ###########################################################################
    # Misc
    ###########################################################################

    def _get_unique_cids_and_count(self):
        n = self.n
        self.unique_cids = DeviceArray(np.uint32, n=n)
        self.unique_cids_map = DeviceArray(np.uint32, n=n)
        uniq_count = DeviceArray(np.uint32, n=1)
        unique_cids_kernel = _get_unique_cids_kernel(self.ctx)
        unique_cids_kernel(self.cids.array, self.unique_cids_map.array,
                           self.unique_cids.array, uniq_count.array)
        self.unique_cid_count = uniq_count.array[0].get()

    def get_leaves(self):
        leaves = DeviceArray(np.uint32, n=self.offsets.array.shape[0])
        num_leaves = DeviceArray(np.uint32, n=1)
        leaves_kernel = _get_leaves_kernel(self.ctx, self.leaf_size)
        leaves_kernel(self.offsets.array, self.pbounds.array,
                      leaves.array, num_leaves.array)

        num_leaves = num_leaves.array[0].get()
        return leaves.array[:num_leaves], num_leaves

    def _sort(self):
        """Set octree as being sorted

        The particle array needs to be aligned by the caller!
        """
        if not self.sorted:
            self.sorted = 1

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

    def tree_bottom_up(self, args, setup, leaf_operation, node_operation,
                       output_expr, preamble=""):
        return tree_bottom_up(self.ctx, args, setup, leaf_operation,
                              node_operation, output_expr, preamble)

    def leaf_tree_traverse(self, args, setup, node_operation, leaf_operation,
                           output_expr, common_operation="", preamble=""):
        """
        Traverse this (source) octree. One thread for each leaf of
        destination octree.
        """

        return leaf_tree_traverse(self.ctx, self.k, args, setup,
                                  node_operation, leaf_operation,
                                  output_expr, common_operation, preamble)
