import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.cltypes
from pyopencl.elementwise import ElementwiseKernel
from pytools import memoize
from pysph.base.opencl import get_context, get_queue
from pysph.base.gpu_nnps_helper import GPUNNPSHelper

make_vec_dict = {
    'float': {
        1: np.float32,
        2: cl.array.vec.make_float2,
        3: cl.array.vec.make_float3
    },
    'double': {
        1: np.float64,
        2: cl.array.vec.make_double2,
        3: cl.array.vec.make_double3
    }
}


@memoize
def get_helper(ctx, src_file, c_type=None):
    # ctx and c_type are the only parameters that
    # change here
    return GPUNNPSHelper(ctx, src_file,
                         c_type=c_type)


@memoize
def get_copy_kernel(ctx, dtype1, dtype2, varnames):
    arg_list = [('%(data_t1)s *%(v)s1' % dict(data_t1=dtype1, v=v))
                for v in varnames]
    arg_list += [('%(data_t2)s *%(v)s2' % dict(data_t2=dtype2, v=v))
                 for v in varnames]
    args = ', '.join(arg_list)

    operation = '; '.join(('%(v)s2[i] = (%(data_t2)s)%(v)s1[i];' %
                           dict(v=v, data_t2=dtype2))
                          for v in varnames)
    return ElementwiseKernel(ctx, args, operation=operation)


_vector_dtypes = {
    'uint': {
        2: cl.cltypes.uint2,
        4: cl.cltypes.uint4,
        8: cl.cltypes.uint8
    },
    'float': {
        1: cl.cltypes.float,
        2: cl.cltypes.float2,
        3: cl.cltypes.float3,
    },
    'double': {
        1: cl.cltypes.double,
        2: cl.cltypes.double2,
        3: cl.cltypes.double3
    }
}


def get_vector_dtype(ctype, dim):
    try:
        return _vector_dtypes[ctype][dim]
    except KeyError:
        # TODO: What to throw?
        raise Exception()


c2d = {
    'half': np.float16,
    'float': np.float32,
    'double': np.float64
}


def ctype_to_dtype(ctype):
    return c2d[ctype]


class GPUParticleArrayWrapper(object):
    def __init__(self, pa_gpu, c_type_src, c_type, varnames):
        self.c_type = c_type
        self.c_type_src = c_type_src
        self.varnames = varnames
        self._allocate_memory(pa_gpu)
        self.sync(pa_gpu)

    def _gpu_copy(self, pa_gpu):
        copy_kernel = get_copy_kernel(get_context(), self.c_type_src,
                                      self.c_type, self.varnames)
        args = [getattr(pa_gpu, v) for v in self.varnames]
        args += [getattr(self, v) for v in self.varnames]
        copy_kernel(*args)

    def _allocate_memory(self, pa_gpu):
        shape = getattr(pa_gpu, self.varnames[0]).shape
        for v in self.varnames:
            setattr(self, v,
                    cl.array.zeros(get_queue(), shape,
                                   ctype_to_dtype(self.c_type))
                    )

    def _gpu_sync(self, pa_gpu):
        v0 = self.varnames[0]

        if getattr(self, v0).shape != getattr(pa_gpu, v0).shape:
            self._allocate_memory(pa_gpu)
        self._gpu_copy(pa_gpu)

    def sync(self, pa_gpu):
        self._gpu_sync(pa_gpu)


class ParticleArrayWrapper(object):
    """A loose wrapper over Particle Array

    Objective is to transparently maintain a copy of
    the original particle array's position properties
    (x, y, z, h)
    """

    def __init__(self, pa, c_type_src, c_type, varnames):
        self._pa = pa
        # If data types are different, then make a copy of the
        # underlying data stored on the device
        if c_type_src != c_type:
            self._pa_gpu_is_copy = True
            self._gpu = GPUParticleArrayWrapper(pa.gpu, c_type_src,
                                                c_type, varnames)
        else:
            self._pa_gpu_is_copy = False
            self._gpu = None

    def get_number_of_particles(self):
        return self._pa.get_number_of_particles()

    @property
    def gpu(self):
        if self._pa_gpu_is_copy:
            return self._gpu
        else:
            return self._pa.gpu

    def sync(self):
        if self._pa_gpu_is_copy:
            self._gpu.sync(self._pa.gpu)
