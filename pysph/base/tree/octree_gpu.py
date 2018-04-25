import pyopencl as cl
import pyopencl.array
import pyopencl.algorithm
import pyopencl.cltypes
from pyopencl.scan import GenericScanKernel

import numpy as np

from pysph.base.gpu_nnps_helper import GPUNNPSHelper
from pysph.base.opencl import DeviceArray
from pysph.base.opencl import get_context, get_queue, named_profile
from pytools import memoize, memoize_method

octree_gpu_counter = 0

def get_octree_id():
    global octree_gpu_counter
    t = octree_gpu_counter
    octree_gpu_counter += 1
    return t


class IncompatibleOctreesException(Exception):
    pass


def memoize(f, *args, **kwargs):
    return f


@named_profile('particle_reordering')
@memoize
def _get_particle_kernel(ctx):
    return GenericScanKernel(
        ctx, cl.cltypes.uint8, neutral="0",
        arguments=r"""__global int *pids,
                    __global ulong *sfc, __global int *cids,
                    __global char *levels, __global char *seg_flag,
                    __global uint2 *pbounds,
                    __global int *offsets,
                    __global uint8 *octant_vector,
                    __global int *pids_next, __global ulong *sfc_next,
                    __global int *cids_next,
                    __global char *levels_next,
                    char curr_level, ulong mask, char rshift,
                    uint csum_nodes_prev""",
        input_expr="eye[eye_index(sfc[i], mask, rshift, levels[i] == curr_level)]",
        scan_expr="(across_seg_boundary ? b : a + b)",
        is_segment_start_expr="seg_flag[i]",
        output_statement=r"""octant_vector[i]=item;""",
        preamble=r"""uint8 constant eye[9] = {
                    (uint8)(1, 1, 1, 1, 1, 1, 1, 1),
                    (uint8)(0, 1, 1, 1, 1, 1, 1, 1),
                    (uint8)(0, 0, 1, 1, 1, 1, 1, 1),
                    (uint8)(0, 0, 0, 1, 1, 1, 1, 1),
                    (uint8)(0, 0, 0, 0, 1, 1, 1, 1),
                    (uint8)(0, 0, 0, 0, 0, 1, 1, 1),
                    (uint8)(0, 0, 0, 0, 0, 0, 1, 1),
                    (uint8)(0, 0, 0, 0, 0, 0, 0, 1),
                    (uint8)(0, 0, 0, 0, 0, 0, 0, 0)
                    };

                    char eye_index(ulong sfc, ulong mask, char rshift, bool same_level) {
                        return (same_level ? 8 : ((sfc & mask) >> rshift));
                    }
                    """
    )

@named_profile('set_offset')
@memoize
def _get_set_offset_kernel(ctx):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""__global uint2 *pbounds, __global uint *offsets,
                      __global int *leaf_count, int csum_nodes_next""",
        input_expr="(pbounds[i].s1 - pbounds[i].s0 <= 1)",
        scan_expr="a + b",
        output_statement=r"""{
            offsets[i] = ((pbounds[i].s1 - pbounds[i].s0 > 1) ? csum_nodes_next + (8 * (i - item)) : -1);
            if (i == N - 1) { *leaf_count = item; }
        }"""
    )

@named_profile('neighbor_psum')
@memoize
def _get_neighbor_psum_kernel(ctx):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""__global int *neighbor_counts""",
        input_expr="neighbor_counts[i]",
        scan_expr="a + b",
        output_statement=r"""neighbor_counts[i] = prev_item;"""
    )

@named_profile('unique_cids')
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
        self.helper = GPUNNPSHelper(self.ctx, "tree/octree_gpu.mako",
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
        max_width = max((self.xmax[i] - self.xmin[i]) for i in range(3))
        self.max_depth = int(np.ceil(np.log2(max_width / self.cell_size))) + 1

    def _initialize_data(self):
        num_particles = self.pa.get_number_of_particles()
        self.pids = DeviceArray(np.uint32, n=num_particles)
        self.pid_keys = DeviceArray(np.uint64, n=num_particles)
        self.levels = DeviceArray(np.uint8, n=num_particles)
        dtype = np.float64 if self.use_double else np.float32
        self.r = DeviceArray(dtype, n=num_particles)
        self.cids = DeviceArray(np.uint32, n=num_particles)
        self.cids.fill(0)

        # Filled after tree built
        self.pbounds = None
        self.offsets = None

        # NNPS stuff
        self.neighbour_cids = {}
        self.neighbor_counts = {}
        self.neighbor_offsets = {}
        self.neighbors = {}
        self.neighbors_stored = {}

    def _reinitialize_data(self):
        num_particles = self.pa.get_number_of_particles()
        self.pids.resize(num_particles)
        self.pid_keys.resize(num_particles)
        self.levels.resize(num_particles)
        dtype = np.float64 if self.use_double else np.float32
        self.r.resize(num_particles)
        self.cids.resize(num_particles)
        self.cids.fill(0)

        # Filled after tree built
        self.pbounds = None
        self.offsets = None

        # NNPS Stuff
        self.neighbour_cids = {}
        self.neighbor_counts = {}
        self.neighbor_offsets = {}
        self.neighbors = {}
        self.neighbors_stored = {}

    def _bin(self):
        dtype = np.float64 if self.use_double else np.float32
        fill_particle_data = self.helper.get_kernel("fill_particle_data")
        pa_gpu = self.pa.gpu
        fill_particle_data(pa_gpu.x, pa_gpu.y, pa_gpu.z, pa_gpu.h,
                           dtype(self.cell_size),
                           self.make_vec(self.xmin[0], self.xmin[1], self.xmin[2]),
                           self.r.array, dtype(self.radius_scale),
                           self.pid_keys.array, self.pids.array,
                           self.levels.array, np.uint8(self.max_depth))

    def _update_node_data(self, offsets_prev, pbounds_prev, offsets, pbounds, seg_flag, octants,
                          csum_nodes, csum_nodes_next):
        """Update node data. Return number of children which are leaves."""

        # Update particle-related data of children
        set_node_data = self.helper.get_kernel("set_node_data")
        set_node_data(offsets_prev.array, pbounds_prev.array, offsets.array, pbounds.array,
                      seg_flag.array, octants.array, np.uint32(csum_nodes),
                      np.uint32(self.pa.get_number_of_particles()))

        # Set children offsets
        leaf_count = DeviceArray(np.uint32, 1)
        set_offsets = _get_set_offset_kernel(self.ctx)
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
            self.num_nodes.append(8 * (self.num_nodes[-1] - num_leaves_here))
            if self.num_nodes[-1] == 0:
                break
            else:
                self.depth += 1

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

    def create_temp_vars(self, temp_vars):
        n = self.pa.get_number_of_particles()
        temp_vars['pids'] = DeviceArray(np.uint32, n)
        temp_vars['pid_keys'] = DeviceArray(np.uint64, n)
        temp_vars['levels'] = DeviceArray(np.uint8, n)
        temp_vars['cids'] = DeviceArray(np.uint32, n)

    def clean_temp_vars(self, temp_vars):
        for k in temp_vars.iterkeys():
            del temp_vars[k]

    def exchange_temp_vars(self, temp_vars):
        for k in temp_vars.iterkeys():
            t = temp_vars[k]
            temp_vars[k] = getattr(self, k)
            setattr(self, k, t)

    def _reorder_particles(self, depth, octants, offsets_parent, pbounds_parent, seg_flag, csum_nodes_prev,
                           temp_vars):
        rshift = np.uint8(3 * (self.max_depth - depth - 1))
        mask = np.uint64(7 << rshift)

        particle_kernel = _get_particle_kernel(self.ctx)

        particle_kernel(self.pids.array, self.pid_keys.array, self.cids.array,
                        self.levels.array, seg_flag.array,
                        pbounds_parent.array, offsets_parent.array,
                        octants.array,
                        temp_vars['pids'].array, temp_vars['pid_keys'].array,
                        temp_vars['cids'].array, temp_vars['levels'].array,
                        np.uint8(depth), mask, rshift,
                        np.uint32(csum_nodes_prev))

        reorder_particles = self.helper.get_kernel('reorder_particles')
        reorder_particles(self.pids.array, self.pid_keys.array, self.cids.array,
                          self.levels.array, seg_flag.array,
                          pbounds_parent.array, offsets_parent.array,
                          octants.array,
                          temp_vars['pids'].array, temp_vars['pid_keys'].array,
                          temp_vars['cids'].array, temp_vars['levels'].array,
                          np.uint8(depth), mask, rshift,
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

        for i in range(self.depth + 1):
            append_layer(
                offsets_temp[i].array, pbounds_temp[i].array,
                self.offsets.array, self.pbounds.array,
                np.int32(curr_offset)
            )
            curr_offset += self.num_nodes[i]

    def _sort(self):
        if not self.sorted:
            self.sorted = True

    def _nnps_preprocess(self, octree_dst):
        n = self.pa.get_number_of_particles()
        pa_src = self.pa.gpu
        pa_dst = octree_dst.pa.gpu
        dst_id = octree_dst.id

        self.neighbour_cids[dst_id] = DeviceArray(np.int32, 27 * self.unique_cid_count)

        # TODO: Resizing
        store_neighbor_cids = self.helper.get_kernel('store_neighbor_cids', sorted=self.sorted)

        dtype = np.float64 if self.use_double else np.float32
        store_neighbor_cids(self.unique_cids_idx.array[:self.unique_cid_count],
                            self.pids.array,
                            octree_dst.offsets.array, octree_dst.pbounds.array,
                            pa_src.x, pa_src.y, pa_src.z,
                            dtype(self.cell_size),
                            self.make_vec(self.xmin[0], self.xmin[1], self.xmin[2]),
                            self.neighbour_cids[dst_id].array,
                            self.levels.array, np.uint8(octree_dst.depth), octree_dst.max_depth)

        self.neighbor_counts[dst_id] = DeviceArray(np.uint32, n=(n + 1))
        self.neighbor_counts[dst_id].fill(0)
        self.neighbor_offsets[dst_id] = DeviceArray(np.uint32, n=(n + 1))


    cpdef _store_neighbour_counts(self, octree_dst):
        n_src = self.pa.get_number_of_particles()
        n_dst = octree_dst.pa.get_number_of_particles()

        pa_gpu = self.pa.gpu
        pa_dst = octree_dst.pa.gpu
        dst_id = octree_dst.id

        store_neighbor_counts = self.helper.get_kernel('store_neighbor_counts', sorted=self.sorted)

        store_neighbor_counts(
            self.pids.array,
            octree_dst.pids.array,
            self.unique_cids_map.array,
            octree_dst.pbounds.array,
            self.levels.array, octree_dst.levels.array,
            self.neighbour_cids[dst_id].array,
            pa_gpu.x, pa_gpu.y, pa_gpu.z, pa_gpu.h,
            pa_dst.x, pa_dst.y, pa_dst.z, pa_dst.h,
            self.neighbor_counts[dst_id].array,
            octree_dst.neighbor_counts[self.id].array,
            self.radius_scale
        )



    def _prefix_sum_neighbor_counts(self, octree_dst):
        n = self.pa.get_number_of_particles()
        dst_id = octree_dst.id
        copy_kernel = self.helper.get_kernel('copy_int')
        copy_kernel(self.neighbor_counts[dst_id].array,
                    self.neighbor_offsets[dst_id].array)


        neighbor_count_psum = _get_neighbor_psum_kernel(self.ctx)
        neighbor_count_psum(self.neighbor_offsets[dst_id].array)
        total_neighbor_count_src = np.int32(self.neighbor_offsets[dst_id].array[n].get())
        self.neighbors[dst_id] = DeviceArray(np.int32, n=total_neighbor_count_src)

    def _store_neighbours(self, octree_dst):
        n = self.pa.get_number_of_particles()
        pa_gpu = self.pa.gpu
        pa_dst = octree_dst.pa.gpu
        dst_id = octree_dst.id

        store_neighbors = self.helper.get_kernel('store_neighbors', sorted=self.sorted)
        store_neighbors(
            self.pids.array, octree_dst.pids.array,
            self.unique_cids_map.array,
            octree_dst.pbounds.array,
            self.levels.array, octree_dst.levels.array,
            self.neighbour_cids[dst_id].array,
            pa_gpu.x, pa_gpu.y, pa_gpu.z, pa_gpu.h,
            pa_dst.x, pa_dst.y, pa_dst.z, pa_dst.h,
            self.neighbor_offsets[dst_id].array,
            octree_dst.neighbor_offsets[self.id].array,
            self.neighbors[dst_id].array, octree_dst.neighbors[self.id].array,
            self.radius_scale
        )

    def store_neighbors(self, octree_dst):
        if self.sorted != octree_dst.sorted:
            raise IncompatibleOctreesException(
                """Particles within both octrees need to be
                either sorted or unsorted"""
            )

        if octree_dst.id in self.neighbors_stored:
            return self.neighbor_counts[octree_dst.id], self.neighbors[octree_dst.id]

        self._nnps_preprocess(octree_dst)
        if self.id != octree_dst.id:
            octree_dst._nnps_preprocess(self)

        self._store_neighbour_counts(octree_dst)
        if self.id != octree_dst.id:
            octree_dst._store_neighbour_counts(self)

        self._prefix_sum_neighbor_counts(octree_dst)
        if self.id != octree_dst.id:
            octree_dst._prefix_sum_neighbor_counts(self)

        self._store_neighbours(octree_dst)
        if self.id != octree_dst.id:
            octree_dst._store_neighbours(self)

        octree_dst.neighbors_stored[self.id] = True
        self.neighbors_stored[octree_dst.id] = True

        return self.neighbor_counts[octree_dst.id], self.neighbors[octree_dst.id]