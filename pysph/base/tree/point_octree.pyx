import pyopencl as cl
import pyopencl.array
import pyopencl.algorithm
import pyopencl.cltypes
from pyopencl.scan import GenericScanKernel

import numpy as np
cimport numpy as np

from pysph.base.gpu_nnps_helper import GPUNNPSHelper
from pysph.base.opencl import DeviceArray
from pysph.base.opencl import get_context, get_queue, profile_with_name
from pytools import memoize, memoize_method

cdef int octree_gpu_counter = 0

cdef get_octree_id():
    global octree_gpu_counter
    cdef int t
    t = octree_gpu_counter
    octree_gpu_counter += 1
    return t


class IncompatibleOctreesException(Exception):
    pass


def memoize(f, *args, **kwargs):
    return f


@profile_with_name('particle_reordering')
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

cdef class OctreeGPU:
    def __init__(self, NNPSParticleArrayWrapper pa_wrapper, radius_scale=1.0,
                 bint use_double=False):
        self.pa_wrapper = pa_wrapper

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
        cdef str norm2 = \
            """
            #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
            """

        self.preamble = norm2

    def refresh(self, np.ndarray xmin, np.ndarray xmax, double hmin):
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
        cdef double max_width
        self.cell_size = self.hmin * self.radius_scale * (1. + 1e-5)
        self.cell_size /= 128
        max_width = max((self.xmax[i] - self.xmin[i]) for i in range(3))
        self.max_depth = int(np.ceil(np.log2(max_width / self.cell_size))) + 1

    def _initialize_data(self):
        num_particles = self.pa_wrapper.get_number_of_particles()
        self.pids = DeviceArray(np.uint32, n=num_particles)
        self.pid_keys = DeviceArray(np.uint64, n=num_particles)
        dtype = np.float64 if self.use_double else np.float32
        self.cids = DeviceArray(np.uint32, n=num_particles)
        self.cids.fill(0)

        # Filled after tree built
        self.pbounds = None
        self.offsets = None

    def _reinitialize_data(self):
        num_particles = self.pa_wrapper.get_number_of_particles()
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
        pa_gpu = self.pa_wrapper.pa.gpu
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
        set_offsets = _get_set_offset_kernel(self.ctx, 32)
        set_offsets(pbounds.array, offsets.array, leaf_count.array,
                    np.uint32(csum_nodes_next))
        return leaf_count.array[0].get()

    def _refresh(self):
        self._calc_cell_size_and_depth()
        self._refresh()
        self._bin()
        self._build_tree()

    def _build_tree(self):
        cdef int num_leaves_here = 0
        cdef int n = self.pa_wrapper.get_number_of_particles()
        cdef dict temp_vars = {}
        cdef int csum_nodes_prev = 0
        cdef int csum_nodes = 1

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
        return temp_vars, octants, pbounds_temp, offsets_temp, num_leaves_here

    def create_temp_vars(self, dict temp_vars):
        cdef int n = self.pa_wrapper.get_number_of_particles()
        temp_vars['pids'] = DeviceArray(np.uint32, n)
        temp_vars['pid_keys'] = DeviceArray(np.uint64, n)
        temp_vars['cids'] = DeviceArray(np.uint32, n)

    def clean_temp_vars(self, dict temp_vars):
        for k in temp_vars.iterkeys():
            del temp_vars[k]

    def exchange_temp_vars(self, dict temp_vars):
        for k in temp_vars.iterkeys():
            t = temp_vars[k]
            temp_vars[k] = getattr(self, k)
            setattr(self, k, t)

    def _reorder_particles(self, depth, octants, offsets_parent, pbounds_parent, seg_flag, csum_nodes_prev,
                           dict temp_vars):
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
        cdef int curr_offset = 0
        cdef int total_nodes = 0

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

    def _set_node_bounds(self):
        dtype = np.float64 if self.use_double else np.float32
        self.node_data = DeviceArray(dtype, self.total_nodes * 8)
        set_node_bounds = self.helper.get_kernel('set_node_bounds', sorted=self.sorted)
        pa_gpu = self.pa_wrapper.pa.gpu

        csum_nodes = self.total_nodes
        for i in range(self.depth, -1, -1):
            csum_nodes_next = csum_nodes
            csum_nodes -= self.num_nodes[i]

            set_node_bounds(
                self.pbounds.array,
                self.offsets.array,
                self.pids.array,
                self.node_data.array,
                pa_gpu.x, pa_gpu.y, pa_gpu.z, pa_gpu.h,
                np.int32(csum_nodes),
                slice=slice(csum_nodes, csum_nodes_next)
            )

    def _find_neighbor_cids(self):
        self.neighbor_cid_count = DeviceArray(np.uint32, self.unique_cid_count + 1)
        find_neighbor_cid_counts = self.helper.get_kernel('find_neighbor_cid_counts')
        find_neighbor_cid_counts(self.unique_cids_idx.array, self.cids.array,
                           self.pbounds.array,
                           self.offsets.array,
                           self.node_data.array, self.node_data.array,
                           self.neighbor_cid_count.array)
        neighbor_psum = _get_neighbor_psum_kernel(self.ctx)
        neighbor_psum(self.neighbor_cid_count.array)
        print('neighbor_counts', self.neighbor_cid_count.array[:6].get())
        total_neighbors = int(self.neighbor_cid_count.array[-1].get())
        print(self.neighbor_cid_count.array[-1].get())
        self.neighbor_cids = DeviceArray(np.uint32, total_neighbors)
        find_neighbor_cids = self.helper.get_kernel('find_neighbor_cids')
        find_neighbor_cids(self.unique_cids_idx.array, self.cids.array,
                           self.pbounds.array,
                           self.offsets.array,
                           self.node_data.array, self.node_data.array,
                           self.neighbor_cid_count.array,
                           self.neighbor_cids.array)
        print('neighbor_cids[:35]', self.neighbor_cids.array[:35].get())

    def _find_neighbors(self):

        n = self.pa_wrapper.get_number_of_particles()
        self.neighbor_count = DeviceArray(np.uint32, n + 1)
        wgs = 32
        pa_gpu = self.pa_wrapper.pa.gpu
        find_neighbor_counts = self.helper.get_kernel('find_neighbor_counts', sorted=self.sorted,
                                                      wgs=wgs)
        print('slice', slice(wgs * self.unique_cid_count))
        find_neighbor_counts(self.unique_cids_idx.array, self.pids.array,
                             self.cids.array,
                             self.pbounds.array,
                             self.offsets.array,
                             pa_gpu.x, pa_gpu.y, pa_gpu.z, pa_gpu.h,
                             self.neighbor_cid_count.array,
                             self.neighbor_cids.array,
                             self.neighbor_count.array,
                             range=slice(wgs * self.unique_cid_count))



