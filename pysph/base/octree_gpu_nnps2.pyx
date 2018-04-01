#cython: embedsignature=True


import pyopencl as cl
import pyopencl.array
import pyopencl.algorithm
from pyopencl.scan import GenericScanKernel
from pyopencl.scan import GenericDebugScanKernel
from pyopencl.elementwise import ElementwiseKernel

import numpy as np
cimport numpy as np
from mako.template import Template

from pysph.base.gpu_nnps_helper import GPUNNPSHelper
from pysph.base.opencl import DeviceArray, get_config, profile
from pysph.base.tree.point_octree import OctreeGPU

cdef class OctreeGPUNNPS2(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, bint fixed_h=False,
                 bint cache=True, bint sort_gids=False, ctx=None,
                 allow_sort = False):
        GPUNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, ctx
        )

        self.src_index = -1
        self.dst_index = -1
        self.sort_gids = sort_gids

        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles

        self.octrees = []
        for i in range(self.narrays):
            self.octrees.append(OctreeGPU(self.pa_wrappers[i].pa, radius_scale,
                                          self.use_double))
        self.allow_sort = allow_sort
        self.domain.update()
        self.update()

    cpdef _bin(self, int pa_index):
        self.octrees[pa_index].refresh(self.xmin, self.xmax,
                                       self.domain.manager.hmin)
        self.octrees[pa_index]._set_node_bounds()

        if self.allow_sort:
            self.spatially_order_particles(pa_index)

    def get_spatially_ordered_indices(self, int pa_index):
        def update():
            self.octrees[pa_index]._sort()

        return self.octrees[pa_index].pids.array, update

    cpdef _refresh(self):
        pass

    cpdef set_context(self, int src_index, int dst_index):
        """Setup the context before asking for neighbors.  The `dst_index`
        represents the particles for whom the neighbors are to be determined
        from the particle array with index `src_index`.

        Parameters
        ----------

         src_index: int: the source index of the particle array.
         dst_index: int: the destination index of the particle array.
        """
        GPUNNPS.set_context(self, src_index, dst_index)

        self.src_index = src_index
        self.dst_index = dst_index

        octree_src = self.octrees[src_index]
        octree_dst = self.octrees[dst_index]
        self.dst_src = src_index != dst_index

        self.neighbor_cid_counts, self.neighbor_cids = octree_dst._find_neighbor_cids(octree_src)

    cdef void find_neighbor_lengths(self, nbr_lengths):
        octree_src = self.octrees[self.src_index]
        octree_dst = self.octrees[self.dst_index]
        octree_dst._find_neighbor_lengths(
            self.neighbor_cid_counts, self.neighbor_cids, octree_src,
            nbr_lengths
        )

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        octree_src = self.octrees[self.src_index]
        octree_dst = self.octrees[self.dst_index]
        octree_dst._find_neighbors(
            self.neighbor_cid_counts, self.neighbor_cids, octree_src,
            start_indices, nbrs
        )
