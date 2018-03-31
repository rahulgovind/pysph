import pyopencl as cl
import pyopencl.array
import pyopencl.algorithm
import pyopencl.cltypes
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.scan import GenericScanKernel

import numpy as np

import numpy as np

from pysph.base.gpu_nnps_helper import GPUNNPSHelper
from pysph.base.opencl import DeviceArray, profile_kernel
from pysph.base.opencl import get_context, get_queue, profile_with_name
from pytools import memoize, memoize_method
import sys

from mako.template import Template

octree_gpu_counter = 0
disable_unicode = False if sys.version_info.major > 2 else True


def get_octree_id():
    global octree_gpu_counter
    t = octree_gpu_counter
    octree_gpu_counter += 1
    return t


class IncompatibleOctreesException(Exception):
    pass


# def memoize(f, *args, **kwargs):
#    return f


LEAF_KERNEL_TEMPLATE = r"""
uint node_idx = unique_cids_idx[i];
uint2 pbound = pbounds[node_idx];

%(operation)s;
"""

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
   int cid = cids[unique_cids_idx[i]];

    /*
     * Assuming max depth of 16
     * stack_idx is also equal to current layer of octree
     * child_idx = number of children iterated through
     * idx_stack = current node
     */
    char child_stack[16];
    int cid_stack[16];

    char idx = 0;
    child_stack[0] = 0;
    cid_stack[0] = 1;
    char flag;
    int curr_cid;
    int child_offset;

    %(setup)s;
    while (idx >= 0) {

        // Recurse to find either leaf node or invalid node
        curr_cid = cid_stack[idx];

        child_offset = offsets[curr_cid];
        %(common_operation)s;

        while (child_offset != -1) {
            %(node_operation)s;

            idx++;
            curr_cid = child_offset;
            cid_stack[idx] = curr_cid;
            child_stack[idx] = 0;
            child_offset = offsets[curr_cid];
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


@profile_with_name('particle_reordering')
@memoize
def _get_particle_kernel(ctx):
    # TODO: Combine with node kernel
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


@profile_with_name('set_offset')
@memoize
def _get_set_offset_kernel(ctx, leaf_size):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""__global uint2 *pbounds, __global uint *offsets,
                      __global int *leaf_count, int csum_nodes_next""",
        input_expr="(pbounds[i].s1 - pbounds[i].s0 <= %(leaf_size)s)" % {'leaf_size': leaf_size},
        scan_expr="a + b",
        output_statement=r"""{
            offsets[i] = ((pbounds[i].s1 - pbounds[i].s0 > %(leaf_size)s) ? csum_nodes_next + (8 * (i - item)) : -1);
            if (i == N - 1) { *leaf_count = item; }
        }""" % {'leaf_size': leaf_size}
    )


@profile_with_name('neighbor_psum')
@memoize
def _get_neighbor_psum_kernel(ctx):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""__global int *neighbor_counts""",
        input_expr="neighbor_counts[i]",
        scan_expr="a + b",
        output_statement=r"""neighbor_counts[i] = prev_item;"""
    )


@profile_with_name('unique_cids')
@memoize
def _get_unique_cids_kernel(ctx):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""int *cids, int *unique_cids_map,
                int *unique_cids_idx, int *unique_cids_count""",
        input_expr="(i == 0 || cids[i] != cids[i-1])",
        scan_expr="a + b",
        output_statement=r"""
            if (item != prev_item) {
                unique_cids_idx[item - 1] = i;
            }
            unique_cids_map[i] = item - 1;
            if (i == N - 1) *unique_cids_count = item;
        """
    )


class OctreeGPU(object):
    def __init__(self, pa, radius_scale=1.0,
                 use_double=False):
        self.pa = pa

        self.radius_scale = radius_scale
        self.use_double = use_double
        self.ctx = get_context()
        self.queue = get_queue()
        self.sorted = False
        self.helper = GPUNNPSHelper(self.ctx, "tree/point_octree.mako",
                                    self.use_double)
        if use_double:
            self.make_vec = cl.array.vec.make_double3
        else:
            self.make_vec = cl.array.vec.make_float3

        self.initialized = False

        self.id = get_octree_id()
        norm2 = \
            """
            #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
            """

        self.preamble = norm2
        self.depth = 0
        self.leaf_size = 32

    def refresh(self, xmin, xmax, hmin):
        self.xmin = xmin
        self.xmax = xmax
        self.hmin = hmin

        self.id = get_octree_id()
        self._calc_cell_size_and_depth()

        if not self.initialized:
            self._initialize_data()
        else:
            self._reinitialize_data()

        self._bin()
        self._build_tree()

    def _calc_cell_size_and_depth(self):
        self.cell_size = self.hmin * self.radius_scale * (1. + 1e-5)
        self.cell_size /= 128
        max_width = max((self.xmax[i] - self.xmin[i]) for i in range(3))
        self.max_depth = int(np.ceil(np.log2(max_width / self.cell_size))) + 1

    def _initialize_data(self):
        num_particles = self.pa.get_number_of_particles()
        self.pids = DeviceArray(np.uint32, n=num_particles)
        self.pid_keys = DeviceArray(np.uint64, n=num_particles)
        dtype = np.float64 if self.use_double else np.float32
        self.cids = DeviceArray(np.uint32, n=num_particles)
        self.cids.fill(0)

        # Filled after tree built
        self.pbounds = None
        self.offsets = None

    def _reinitialize_data(self):
        num_particles = self.pa.get_number_of_particles()
        self.pids.resize(num_particles)
        self.pid_keys.resize(num_particles)
        dtype = np.float64 if self.use_double else np.float32
        self.cids.resize(num_particles)
        self.cids.fill(0)

        # Filled after tree built
        self.pbounds = None
        self.offsets = None

    def _bin(self, ):
        dtype = np.float64 if self.use_double else np.float32
        fill_particle_data = self.helper.get_kernel("fill_particle_data")
        pa_gpu = self.pa.gpu
        fill_particle_data(pa_gpu.x, pa_gpu.y, pa_gpu.z,
                           dtype(self.cell_size),
                           self.make_vec(self.xmin[0], self.xmin[1], self.xmin[2]),
                           self.pid_keys.array, self.pids.array)

    def _update_node_data(self, offsets_prev, pbounds_prev, offsets, pbounds, seg_flag, octants,
                          csum_nodes, csum_nodes_next):
        """Update node data. Return number of children which are leaves."""

        # Update particle-related data of children
        set_node_data = self.helper.get_kernel("set_node_data")
        set_node_data(offsets_prev.array, pbounds_prev.array, offsets.array, pbounds.array,
                      seg_flag.array, octants.array, np.uint32(csum_nodes))

        # Set children offsets
        leaf_count = DeviceArray(np.uint32, 1)
        set_offsets = _get_set_offset_kernel(self.ctx, self.leaf_size)
        set_offsets(pbounds.array, offsets.array, leaf_count.array,
                    np.uint32(csum_nodes_next))
        return leaf_count.array[0].get()

    def _refresh(self):
        self._calc_cell_size_and_depth()
        self._refresh()
        self._bin()
        self._build_tree()

    def _build_tree(self):
        num_leaves_here = 0
        n = self.pa.get_number_of_particles()
        temp_vars = {}
        csum_nodes_prev = 0
        csum_nodes = 1

        self.num_nodes = [1]

        # Initialize temporary data (but persistent across layers)
        self.create_temp_vars(temp_vars)

        octants = DeviceArray(cl.cltypes.uint8, n)

        seg_flag = DeviceArray(cl.cltypes.char, n)
        seg_flag.fill(0)
        seg_flag.array[0] = 1

        offsets_temp = [DeviceArray(np.int32, 1)]
        offsets_temp[-1].fill(1)

        pbounds_temp = [DeviceArray(cl.cltypes.uint2, 1)]
        pbounds_temp[-1].array[0].set(cl.cltypes.make_uint2(0, n))

        for depth in range(1, self.max_depth):
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

            self._reorder_particles(depth, octants, offsets_temp[-2], pbounds_temp[-2],
                                    seg_flag, csum_nodes_prev, temp_vars)
            num_leaves_here = self._update_node_data(offsets_temp[-2], pbounds_temp[-2],
                                                     offsets_temp[-1], pbounds_temp[-1],
                                                     seg_flag, octants,
                                                     csum_nodes,
                                                     csum_nodes + self.num_nodes[-1])

            csum_nodes_prev = csum_nodes
            csum_nodes += self.num_nodes[-1]

        # Compress all layers into a single array
        self._compress_layers(offsets_temp, pbounds_temp)

        self.unique_cids_idx = DeviceArray(np.uint32, n=n)
        self.unique_cids_map = DeviceArray(np.uint32, n=n)
        uniq_count = DeviceArray(np.uint32, n=1)
        unique_cids_kernel = _get_unique_cids_kernel(self.ctx)
        unique_cids_kernel(self.cids.array, self.unique_cids_map.array,
                           self.unique_cids_idx.array, uniq_count.array)
        self.unique_cid_count = uniq_count.array[0].get()
        self.clean_temp_vars(temp_vars)

    def create_temp_vars(self, temp_vars):
        n = self.pa.get_number_of_particles()
        temp_vars['pids'] = DeviceArray(np.uint32, n)
        temp_vars['pid_keys'] = DeviceArray(np.uint64, n)
        temp_vars['cids'] = DeviceArray(np.uint32, n)

    def clean_temp_vars(self, temp_vars):
        for k in list(temp_vars.keys()):
            del temp_vars[k]

    def exchange_temp_vars(self, temp_vars):
        for k in temp_vars.keys():
            t = temp_vars[k]
            temp_vars[k] = getattr(self, k)
            setattr(self, k, t)

    def _reorder_particles(self, depth, octants, offsets_parent, pbounds_parent, seg_flag, csum_nodes_prev,
                           temp_vars):
        rshift = np.uint8(3 * (self.max_depth - depth - 1))
        mask = np.uint64(7 << rshift)

        particle_kernel = _get_particle_kernel(self.ctx)

        particle_kernel(self.pid_keys.array, seg_flag.array,
                        octants.array, mask, rshift)

        reorder_particles = self.helper.get_kernel('reorder_particles')
        reorder_particles(self.pids.array, self.pid_keys.array, self.cids.array,
                          seg_flag.array,
                          pbounds_parent.array, offsets_parent.array,
                          octants.array,
                          temp_vars['pids'].array, temp_vars['pid_keys'].array,
                          temp_vars['cids'].array,
                          mask, rshift,
                          np.uint32(csum_nodes_prev))
        self.exchange_temp_vars(temp_vars)

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

    def _find_neighbors(self, store=True):
        n = self.pa.get_number_of_particles()
        self.neighbor_count = DeviceArray(np.uint32, n + 1)
        wgs = self.leaf_size
        pa_gpu = self.pa.gpu
        find_neighbor_counts = self.helper.get_kernel('find_neighbor_counts', sorted=self.sorted,
                                                      wgs=wgs)
        find_neighbor_counts(self.unique_cids_idx.array, self.pids.array,
                             self.cids.array,
                             self.pbounds.array,
                             self.offsets.array,
                             pa_gpu.x, pa_gpu.y, pa_gpu.z, pa_gpu.h,
                             self.neighbor_cid_count.array,
                             self.neighbor_cids.array,
                             self.neighbor_count.array,
                             range=slice(wgs * self.unique_cid_count),
                             fixed_work_items=wgs)

        if not store:
            return

        neighbor_psum = _get_neighbor_psum_kernel(self.ctx)
        neighbor_psum(self.neighbor_count.array)

        total_neighbors = self.neighbor_count.array[-1].get()
        self.neighbors = DeviceArray(np.uint32, int(total_neighbors))

        find_neighbors = self.helper.get_kernel('find_neighbors', sorted=self.sorted,
                                                wgs=wgs)
        find_neighbors(self.unique_cids_idx.array, self.pids.array,
                       self.cids.array,
                       self.pbounds.array,
                       self.offsets.array,
                       pa_gpu.x, pa_gpu.y, pa_gpu.z, pa_gpu.h,
                       self.neighbor_cid_count.array,
                       self.neighbor_cids.array,
                       self.neighbor_count.array,
                       self.neighbors.array,
                       range=slice(wgs * self.unique_cid_count),
                       fixed_work_items=wgs)

    def allocate_node_prop(self, dtype):
        return DeviceArray(dtype, self.total_nodes)

    def allocate_leaf_prop(self, dtype):
        return DeviceArray(dtype, int(self.unique_cid_count))

    def get_preamble(self):
        @memoize
        def _get_preamble(sorted):
            return Template("""
                % if sorted:
                    #define PID(idx) (idx)
                % else:
                    #define PID(idx) (pids[idx])
                % endif
                """, disable_unicode=disable_unicode).render(sorted=self.sorted)

        return _get_preamble(self.sorted)

    def tree_bottom_up(self, args, setup, leaf_operation, node_operation, output_expr):
        @memoize
        def _tree_bottom_up(use_double, sorted, args, setup,
                            leaf_operation, node_operation, output_expr):
            data_t = 'double' if use_double else 'float'
            preamble = self.get_preamble()
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

            kernel = ElementwiseKernel(self.ctx, args, operation=operation, preamble=preamble)

            def callable(octree, *args):
                csum_nodes = octree.total_nodes
                out = None
                for i in range(octree.depth, -1, -1):
                    csum_nodes_next = csum_nodes
                    csum_nodes -= octree.num_nodes[i]
                    out = kernel(octree.offsets.array, octree.pbounds.array, *args,
                                 slice=slice(csum_nodes, csum_nodes_next))
                return out

            return callable

        return _tree_bottom_up(self.use_double, self.sorted, args, setup,
                               leaf_operation, node_operation, output_expr)

    def leaf_tree_traverse(self, args, setup, node_operation, leaf_operation, output_expr, common_operation=""):
        @memoize
        def _leaf_tree_traverse(use_double, sorted, args, setup, node_operation,
                                leaf_operation, output_expr, common_operation):
            data_t = 'double' if use_double else 'float'
            premable = self.get_preamble() + Template("""
                #define IN_BOUNDS(X, MIN, MAX) ((X >= MIN) && (X < MAX))
                #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
                #define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
                #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
                #define AVG(X, Y) (((X) + (Y)) / 2)
                #define ABS(X) ((X) > 0 ? (X) : -(X))
                #define EPS 1e-6
                #define INF 1e6
                #define SQR(X) ((X) * (X))

                char contains(private int *n1, private int *n2) {
                    // Check if node n1 contains node n2
                    char res = 1;
                    %for i in range(3):
                        res = res && (n1[${i}] <= n2[${i}]) && (n1[3 + ${i}] >= n2[3 + ${i}]);
                    %endfor

                    return res;
                }

                char contains_search(private ${data_t} *n1, private ${data_t} *n2) {
                    // Check if node n1 contains node n2 with n1 having its search radius extension
                    ${data_t} h = n1[6];
                    char res = 1;
                    %for i in range(3):
                        res = res & (n1[${i}] - h <= n2[${i}]) & (n1[3 + ${i}] + h >= n2[3 + ${i}]);
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
                        cdist = fabs((n1[${i}] + n1[3 + ${i}]) / 2 - (n2[${i}] + n2[3 + ${i}]) / 2);
                        w1 = fabs(n1[${i}] - n1[3 + ${i}]);
                        w2 = fabs(n2[${i}] - n2[3 + ${i}]);
                        wavg = AVG(w1, w2);
                        res &= (cdist - wavg <= h);
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
                "int *unique_cids_idx, int *cids, int *offsets, " + args,
                disable_unicode=disable_unicode
            ).render(data_t=data_t, sorted=sorted)

            kernel = ElementwiseKernel(self.ctx, args, operation=operation, preamble=premable)

            def callable(octree, *args):
                return kernel(octree.unique_cids_idx.array[:octree.unique_cid_count],
                              octree.cids.array, octree.offsets.array, *args)

            return callable

        return _leaf_tree_traverse(self.use_double, self.sorted, args, setup, node_operation,
                                   leaf_operation, output_expr, common_operation)

    def _set_node_bounds(self):
        # TODO: type based on use_double
        self.node_xmin = self.allocate_node_prop(cl.cltypes.float3)
        self.node_xmax = self.allocate_node_prop(cl.cltypes.float3)
        self.node_hmax = self.allocate_node_prop(cl.cltypes.float)

        set_node_bounds = self.tree_bottom_up(
            setup=r"""
                ${data_t} xmin[3] = {1e6, 1e6, 1e6};
                ${data_t} xmax[3] = {-1e6, -1e6, -1e6};
                ${data_t} hmax = 0;
                """,
            args="""int *pids, ${data_t} *x, ${data_t} *y, ${data_t} *z, ${data_t} *h,
                     float3 *node_xmin, float3 *node_xmax, float *node_hmax""",
            leaf_operation="""
                <% ch = ['x', 'y', 'z'] %>
                for (int j=pbound.s0; j < pbound.s1; j++) {
                    int pid = PID(j);
                    % for d in range(3):
                        xmin[${d}] = fmin(xmin[${d}], ${ch[d]}[pid]);
                        xmax[${d}] = fmax(xmax[${d}], ${ch[d]}[pid]);
                    % endfor
                    hmax = fmax(h[pid], hmax);
                }
                """,
            node_operation="""
                % for i in range(8):
                    % for d in range(3):
                        xmin[${d}] = fmin(xmin[${d}], node_xmin[child_offset + ${i}].s${d});
                        xmax[${d}] = fmax(xmax[${d}], node_xmax[child_offset + ${i}].s${d});
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
        set_node_bounds(self, self.pids.array, pa_gpu.x, pa_gpu.y, pa_gpu.z, pa_gpu.h,
                        self.node_xmin.array, self.node_xmax.array, self.node_hmax.array)

    def _leaf_neighbor_operation(self, args, setup, operation, output_expr):
        setup = """
        ${data_t} ndst[8];
        ${data_t} nsrc[8];

        %% for i in range(3):
            ndst[${i}] = node_xmin[cid].s${i};
            ndst[${i} + 3] = node_xmax[cid].s${i};
        %% endfor
        ndst[6] = node_hmax[cid];

        %(setup)s;
        """ % dict(setup=setup)

        node_operation = """
        % for i in range(3):
            nsrc[${i}] = node_xmin[curr_cid].s${i};
            nsrc[${i} + 3] = node_xmax[curr_cid].s${i};
        % endfor
        nsrc[6] = node_hmax[curr_cid];

        if (!intersects(ndst, nsrc) && !contains(nsrc, ndst)) {
            flag = 0;
            break;
        }
        """

        leaf_operation = """
        %% for i in range(3):
            nsrc[${i}] = node_xmin[curr_cid].s${i};
            nsrc[${i} + 3] = node_xmax[curr_cid].s${i};
        %% endfor
        nsrc[6] = node_hmax[curr_cid];

        if (intersects(ndst, nsrc) || contains_search(ndst, nsrc)) {
            %(operation)s;
        }
        """ % dict(operation=operation)

        output_expr = output_expr
        args = "${data_t}3 *node_xmin, ${data_t}3 *node_xmax, ${data_t} *node_hmax, " + args

        kernel = self.leaf_tree_traverse(args, setup,
                                         node_operation, leaf_operation,
                                         output_expr)

        def callable(*args):
            return kernel(self, self.node_xmin.array, self.node_xmax.array, self.node_hmax.array,
                          *args)

        return callable

    def _find_neighbor_cids(self):
        self.neighbor_cid_count = DeviceArray(np.uint32, self.unique_cid_count + 1)
        find_neighbor_cid_counts = self._leaf_neighbor_operation(
            args="uint2 *pbounds, int *cnt",
            setup="int count=0",
            operation="""
                    if (pbounds[curr_cid].s0 < pbounds[curr_cid].s1)
                        count++;
                    """,
            output_expr="cnt[i] = count;"
        )
        find_neighbor_cid_counts = profile_kernel(find_neighbor_cid_counts, 'find_neighbor_cid_count')
        find_neighbor_cid_counts(self.pbounds.array,
                                 self.neighbor_cid_count.array)

        neighbor_psum = _get_neighbor_psum_kernel(self.ctx)
        neighbor_psum(self.neighbor_cid_count.array)

        total_neighbors = int(self.neighbor_cid_count.array[-1].get())
        self.neighbor_cids = DeviceArray(np.uint32, total_neighbors)

        find_neighbor_cids = self._leaf_neighbor_operation(
            args="uint2 *pbounds, int *cnt, int *neighbor_cids",
            setup="int offset=cnt[i];",
            operation="""
            if (pbounds[curr_cid].s0 < pbounds[curr_cid].s1)
                neighbor_cids[offset++] = curr_cid;
            """,
            output_expr=""
        )
        find_neighbor_cids = profile_kernel(find_neighbor_cids, 'find_neighbor_cids')
        find_neighbor_cids(self.pbounds.array,
                           self.neighbor_cid_count.array, self.neighbor_cids.array)

    def _sort(self):
        if not self.sorted:
            self.sorted = True