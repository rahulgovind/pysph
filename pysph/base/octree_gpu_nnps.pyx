#cython: embedsignature = True

# Cython for compiler directives
cimport cython

import pyopencl as cl
import pyopencl.array
import pyopencl.algorithm
from pyopencl.scan import GenericScanKernel
from pyopencl.scan import GenericDebugScanKernel
from pyopencl.elementwise import ElementwiseKernel

import numpy as np
cimport numpy as np

from pysph.base.gpu_nnps_helper import GPUNNPSHelper
from pysph.base.octree_gpu import OctreeGPU

cdef class OctreeGPUNNPS(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, bint fixed_h=False,
                 bint cache=True, bint sort_gids=False, ctx=None):
        GPUNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, ctx
        )

        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i

        self.octrees = []

        for i in range(self.narrays):
            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            self.octrees.append(OctreeGPU(pa_wrapper, radius_scale, self.use_double))

        self.domain.update()
        self.update()

    cpdef _bin(self, int pa_index):
        """Group particles into bins
        """
        self.octrees[pa_index].refresh(self.xmin, self.xmax,
                                       self.domain.manager.hmin)

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

    cdef void find_neighbor_lengths(self, nbr_lengths):
        self.octrees[self.src_index].store_neighbors(self.octrees[self.dst_index])
        nbr_lengths.set(self.octrees[self.src_index].neighbor_counts[self.dst_index].array)

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        self.octrees[self.src_index].store_neighbors(self.octrees[self.dst_index])
        nbrs.set(self.octrees[self.src_index].neighbor_counts[self.dst_index].array)
