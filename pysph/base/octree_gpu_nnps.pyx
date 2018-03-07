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
from pysph.base.opencl import get_context
from pysph.base.octree_gpu import OctreeGPU

cdef class OctreeGPUNNPS(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, bint fixed_h=False,
                 bint cache=True, bint sort_gids=False, ctx=None,
                 bint allow_sort=False):
        GPUNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, ctx, self_cache=True
        )

        self.allow_sort = allow_sort

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
        if self.allow_sort:
            print("Sorting")
            for i in range(self.narrays):
                self.spatially_order_particles(i)

    cpdef _refresh(self):
        pass

    def get_spatially_ordered_indices(self, int pa_index):
        def update():
            self.octrees[pa_index]._sort()

        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        num_particles = pa_wrapper.get_number_of_particles()

        return self.octrees[pa_index].pids.array, update

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
        self.octrees[src_index].store_neighbors(self.octrees[dst_index])

    cdef void find_neighbor_lengths(self, nbr_lengths):
        counts, nbrs = self.octrees[self.src_index].store_neighbors(self.octrees[self.dst_index])
        nbr_lengths.set_data(counts.array)

    cdef void find_nearest_neighbors_gpu(self, neighbors, _):
        psum, nbrs = self.octrees[self.src_index].store_neighbors(self.octrees[self.dst_index])
        neighbors.set_data(nbrs.array)
