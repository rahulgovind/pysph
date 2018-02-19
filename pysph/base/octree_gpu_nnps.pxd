# cython: embedsignature=True
from libcpp.map cimport map
from libcpp.pair cimport pair

from pysph.base.gpu_nnps_base cimport *

cdef class OctreeGPUNNPS(GPUNNPS):
    cdef NNPSParticleArrayWrapper src, dst  # Current source and destination..
    cdef public list octrees

    # Functions
    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices)

    cdef void find_neighbor_lengths(self, nbr_lengths)
    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices)
