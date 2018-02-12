import numpy as np
cimport numpy as np

from pysph.base.gpu_nnps_base cimport *

cdef class OctreeGPU:
    cdef public NNPSParticleArrayWrapper pa_wrapper

    cdef public np.ndarray xmin
    cdef public np.ndarray xmax
    cdef public double hmin
    cdef public double hmax
    cdef public double cell_size
    cdef public double radius_scale

    cdef public object pbounds
    cdef public object offsets
    cdef public object pids
    cdef public object cids
    cdef public object pid_keys
    cdef public object levels
    cdef public object r

    cdef public list num_nodes
    cdef public char max_depth
    cdef public char depth
    cdef public bint use_double
    cdef public bint sorted

    cdef object helper
    cdef object ctx
    cdef object queue

    cdef object particle_kernel
    cdef object set_offsets_kernel
    cdef object neighbor_count_psum
    cdef object make_vec
    cdef object preamble

    # For NNPS
    cdef public object unique_cids
    cdef public int unique_cid_count
    cdef public object neighbour_cids
    cdef public object neighbor_counts
    cdef public object neighbors