from pysph.base.tree.point_octree import OctreeGPU

import numpy as np
from pytools import memoize

from pyopencl.scan import GenericScanKernel

from pysph.base.opencl import DeviceArray, DeviceWGSException
from pysph.cpy.opencl import get_queue, profile_kernel, named_profile
from pysph.base.tree.helpers import ctype_to_dtype


class IncompatibleOctreesException(Exception):
    pass


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


# Separating the generic Octree implementation details
# and the NNPS specific details
class OctreeGPUNNPS(OctreeGPU):
    def __init__(self, pa, radius_scale=1.0,
                 use_double=False, leaf_size=32, c_type='float'):
        super(OctreeGPUNNPS, self).__init__(pa, radius_scale, use_double,
                                            leaf_size, c_type)

    def _leaf_neighbor_operation(self, octree_src, args, setup, operation,
                                 output_expr):
        """
        Template for finding neighboring cids of a cell.
        """
        setup = """
        ${data_t} ndst[8];
        ${data_t} nsrc[8];

        %% for i in range(3):
            ndst[${i}] = node_xmin_dst[cid_dst].s${i};
            ndst[${i} + 3] = node_xmax_dst[cid_dst].s${i};
        %% endfor
        ndst[6] = node_hmax_dst[cid_dst];

        %(setup)s;
        """ % dict(setup=setup)

        node_operation = """
        % for i in range(3):
            nsrc[${i}] = node_xmin_src[cid_src].s${i};
            nsrc[${i} + 3] = node_xmax_src[cid_src].s${i};
        % endfor
        nsrc[6] = node_hmax_src[cid_src];

        if (!intersects(ndst, nsrc) && !contains(nsrc, ndst)) {
            flag = 0;
            break;
        }
        """

        leaf_operation = """
        %% for i in range(3):
            nsrc[${i}] = node_xmin_src[cid_src].s${i};
            nsrc[${i} + 3] = node_xmax_src[cid_src].s${i};
        %% endfor
        nsrc[6] = node_hmax_src[cid_src];

        if (intersects(ndst, nsrc) || contains_search(ndst, nsrc)) {
            %(operation)s;
        }
        """ % dict(operation=operation)

        output_expr = output_expr
        args = """
        ${data_t}3 *node_xmin_src, ${data_t}3 *node_xmax_src,
        ${data_t} *node_hmax_src,
        ${data_t}3 *node_xmin_dst, ${data_t}3 *node_xmax_dst,
        ${data_t} *node_hmax_dst,
        """ + args

        kernel = octree_src.leaf_tree_traverse(args, setup,
                                               node_operation, leaf_operation,
                                               output_expr)

        def callable(*args):
            return kernel(octree_src, self,
                          octree_src.node_xmin.array,
                          octree_src.node_xmax.array,
                          octree_src.node_hmax.array,
                          self.node_xmin.array, self.node_xmax.array,
                          self.node_hmax.array,
                          *args)

        return callable

    def _find_neighbor_cids(self, octree_src):
        neighbor_cid_count = DeviceArray(np.uint32, self.unique_cid_count + 1)
        find_neighbor_cid_counts = self._leaf_neighbor_operation(
            octree_src,
            args="uint2 *pbounds, int *cnt",
            setup="int count=0",
            operation="""
                    if (pbounds[cid_src].s0 < pbounds[cid_src].s1)
                        count++;
                    """,
            output_expr="cnt[i] = count;"
        )
        find_neighbor_cid_counts = profile_kernel(
            find_neighbor_cid_counts, 'find_neighbor_cid_count')
        find_neighbor_cid_counts(octree_src.pbounds.array,
                                 neighbor_cid_count.array)

        neighbor_psum = _get_neighbor_psum_kernel(self.ctx)
        neighbor_psum(neighbor_cid_count.array)

        total_neighbors = int(neighbor_cid_count.array[-1].get())
        neighbor_cids = DeviceArray(np.uint32, total_neighbors)

        find_neighbor_cids = self._leaf_neighbor_operation(
            octree_src,
            args="uint2 *pbounds, int *cnt, int *neighbor_cids",
            setup="int offset=cnt[i];",
            operation="""
            if (pbounds[cid_src].s0 < pbounds[cid_src].s1)
                neighbor_cids[offset++] = cid_src;
            """,
            output_expr=""
        )
        find_neighbor_cids = profile_kernel(
            find_neighbor_cids, 'find_neighbor_cids')
        find_neighbor_cids(octree_src.pbounds.array,
                           neighbor_cid_count.array, neighbor_cids.array)
        return neighbor_cid_count, neighbor_cids

    def _find_neighbor_lengths_elementwise(self, neighbor_cid_count,
                                           neighbor_cids, octree_src,
                                           neighbor_count):
        self.check_nnps_compatibility(octree_src)

        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = octree_src.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        find_neighbor_counts = self.helper.get_kernel(
            'find_neighbor_counts_elementwise', sorted=self.sorted
        )
        find_neighbor_counts(self.unique_cids_map.array, octree_src.pids.array,
                             self.pids.array,
                             self.cids.array,
                             octree_src.pbounds.array, self.pbounds.array,
                             pa_gpu_src.x, pa_gpu_src.y, pa_gpu_src.z,
                             pa_gpu_src.h,
                             pa_gpu_dst.x, pa_gpu_dst.y, pa_gpu_dst.z,
                             pa_gpu_dst.h,
                             dtype(self.radius_scale),
                             neighbor_cid_count.array,
                             neighbor_cids.array,
                             neighbor_count)

    def _find_neighbors_elementwise(self, neighbor_cid_count, neighbor_cids,
                                    octree_src, start_indices, neighbors):
        self.check_nnps_compatibility(octree_src)

        n = self.pa.get_number_of_particles()
        wgs = self.leaf_size
        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = octree_src.pa.gpu

        dtype = ctype_to_dtype(self.c_type)

        find_neighbors = self.helper.get_kernel(
            'find_neighbors_elementwise', sorted=self.sorted)
        find_neighbors(self.unique_cids_map.array, octree_src.pids.array,
                       self.pids.array,
                       self.cids.array,
                       octree_src.pbounds.array, self.pbounds.array,
                       pa_gpu_src.x, pa_gpu_src.y, pa_gpu_src.z, pa_gpu_src.h,
                       pa_gpu_dst.x, pa_gpu_dst.y, pa_gpu_dst.z, pa_gpu_dst.h,
                       dtype(self.radius_scale),
                       neighbor_cid_count.array,
                       neighbor_cids.array,
                       start_indices,
                       neighbors)

    def _is_valid_nnps_wgs(self):
        # Max work group size can only be found by building the
        # kernel.
        try:
            find_neighbor_counts = self.helper.get_kernel(
                'find_neighbor_counts', sorted=self.sorted, wgs=self.leaf_size
            )

            find_neighbor = self.helper.get_kernel(
                'find_neighbors', sorted=self.sorted, wgs=self.leaf_size
            )
        except DeviceWGSException:
            return False
        else:
            return True

    def _find_neighbor_lengths(self, neighbor_cid_count, neighbor_cids,
                               octree_src, neighbor_count,
                               use_partitions=False):
        self.check_nnps_compatibility(octree_src)

        n = self.pa.get_number_of_particles()
        wgs = self.leaf_size
        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = octree_src.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        def find_neighbor_counts_for_partition(partition_cids, partition_size,
                                               partition_wgs, q=None):
            find_neighbor_counts = self.helper.get_kernel(
                'find_neighbor_counts', sorted=self.sorted, wgs=wgs
            )
            find_neighbor_counts(partition_cids.array, octree_src.pids.array,
                                 self.pids.array,
                                 self.cids.array,
                                 octree_src.pbounds.array, self.pbounds.array,
                                 pa_gpu_src.x, pa_gpu_src.y, pa_gpu_src.z,
                                 pa_gpu_src.h,
                                 pa_gpu_dst.x, pa_gpu_dst.y, pa_gpu_dst.z,
                                 pa_gpu_dst.h,
                                 dtype(self.radius_scale),
                                 neighbor_cid_count.array,
                                 neighbor_cids.array,
                                 neighbor_count,
                                 gs=(partition_wgs * partition_size,),
                                 ls=(partition_wgs,),
                                 queue=(get_queue() if q is None else q))

        if use_partitions and wgs > 32:
            if wgs < 128:
                wgs1 = 32
            else:
                wgs1 = 64

            m1, n1 = self.get_leaf_size_partitions(0, wgs1)

            find_neighbor_counts_for_partition(m1, n1, min(wgs, wgs1))
            m2, n2 = self.get_leaf_size_partitions(wgs1, wgs)
            find_neighbor_counts_for_partition(m2, n2, wgs)
        else:
            find_neighbor_counts_for_partition(
                self.unique_cids, self.unique_cid_count, wgs)

    def _find_neighbors(self, neighbor_cid_count, neighbor_cids, octree_src,
                        start_indices, neighbors, use_partitions=False):
        self.check_nnps_compatibility(octree_src)

        wgs = self.leaf_size if self.leaf_size % 32 == 0 else \
            self.leaf_size + 32 - self.leaf_size % 32
        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = octree_src.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        def find_neighbors_for_partition(partition_cids, partition_size,
                                         partition_wgs, q=None):
            find_neighbors = self.helper.get_kernel('find_neighbors',
                                                    sorted=self.sorted,
                                                    wgs=wgs)
            find_neighbors(partition_cids.array, octree_src.pids.array,
                           self.pids.array,
                           self.cids.array,
                           octree_src.pbounds.array, self.pbounds.array,
                           pa_gpu_src.x, pa_gpu_src.y, pa_gpu_src.z,
                           pa_gpu_src.h,
                           pa_gpu_dst.x, pa_gpu_dst.y, pa_gpu_dst.z,
                           pa_gpu_dst.h,
                           dtype(self.radius_scale),
                           neighbor_cid_count.array,
                           neighbor_cids.array,
                           start_indices,
                           neighbors,
                           gs=(partition_wgs * partition_size,),
                           ls=(partition_wgs,),
                           queue=(get_queue() if q is None else q))

        if use_partitions and wgs > 32:
            if wgs < 128:
                wgs1 = 32
            else:
                wgs1 = 64

            m1, n1 = self.get_leaf_size_partitions(0, wgs1)
            fraction = (n1 / int(self.unique_cid_count))

            if fraction > 0.3:
                find_neighbors_for_partition(m1, n1, wgs1)
                m2, n2 = self.get_leaf_size_partitions(wgs1, wgs)
                assert (n1 + n2 == self.unique_cid_count)
                find_neighbors_for_partition(m2, n2, wgs)
                return
        else:
            find_neighbors_for_partition(
                self.unique_cids, self.unique_cid_count, wgs)

    def check_nnps_compatibility(self, octree):
        """Check if octree types and parameters are compatible for NNPS

        Two octrees must satisfy a few conditions so that NNPS can be performed
        on one octree using the other as reference. In this case, the following
        conditions must be satisfied -

        1) Currently both should be instances of point_octree.OctreeGPUNNPS
        2) Both must have the same sortedness
        3) Both must use the same floating-point datatype
        4) Both must have the same leaf sizes

        Parameters
        ----------
        octree
        """
        if not isinstance(octree, OctreeGPUNNPS):
            raise IncompatibleOctreesException(
                "Both octrees must be of the same type for NNPS"
            )

        if self.sorted != octree.sorted:
            raise IncompatibleOctreesException(
                "Octree sortedness need to be the same for NNPS"
            )

        if self.c_type != octree.c_type or \
                        self.use_double != octree.use_double:
            raise IncompatibleOctreesException(
                "Octree floating-point data types need to be the same for NNPS"
            )

        if self.leaf_size != octree.leaf_size:
            raise IncompatibleOctreesException(
                "Octree leaf sizes need to be the same for NNPS (%d != %d)" % (
                    self.leaf_size, octree.leaf_size)
            )

        return
