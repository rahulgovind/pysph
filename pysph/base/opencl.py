"""Common OpenCL related functionality.
"""

from __future__ import print_function
import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: 401
import pyopencl.algorithm
import pyopencl.tools
from pyopencl.scan import GenericScanKernel
from pyopencl.elementwise import ElementwiseKernel
from collections import defaultdict
from operator import itemgetter
from mako.template import Template
from pyopencl.tools import (dtype_to_ctype, VectorArg, ScalarArg,
                            KernelTemplateBase, dtype_to_c_struct)

import logging

logger = logging.getLogger()

from .config import get_config
import time
from pysph.base.particle_array import ParticleArray

_ctx = None
_queue = None
_profile_info = defaultdict(float)
_cpu_profile_info = defaultdict(float)
# args: uint* indices, dtype array, int length
REMOVE_KNL = Template(r"""//CL//
        unsigned int idx = indices[n - 1 - i];
        array[idx] = array[length - 1 - i];
""")

# args: tag_array, tag, indices, head
REMOVE_INDICES_KNL = Template(r"""//CL//
        if(tag_array[i] == tag)
            indices[atomic_inc(&head[0])] = i;
""")

# args: tag_array, num_real_particles
NUM_REAL_PARTICLES_KNL = Template(r"""//CL//
        if(i != 0 && tag_array[i] != tag_array[i-1])
        {
            num_real_particles[0] = i;
            return;
        }
""")


def get_context():
    global _ctx
    if _ctx is None:
        _ctx = cl.create_some_context()
    return _ctx


def set_context(ctx):
    global _ctx
    _ctx = ctx


def get_queue():
    global _queue
    if _queue is None:
        properties = None
        if get_config().profile:
            properties = cl.command_queue_properties.PROFILING_ENABLE
        _queue = cl.CommandQueue(get_context(), properties=properties)
    return _queue


def set_queue(q):
    global _queue
    _queue = q


def profile(name, event):
    global _profile_info
    event.wait()
    time = (event.profile.end - event.profile.start) * 1e-9
    _profile_info[name] += time


def cpu_profile(name, total_time):
    global _cpu_profile_info
    _cpu_profile_info[name] += total_time


def print_profile(device_only=False):
    global _profile_info
    profile_info = sorted(_profile_info.items(), key=itemgetter(1),
                          reverse=True)
    if len(profile_info) == 0:
        print("No profile information available")
        return
    print("{:<30} {:<30}".format('Kernel', 'Time'))
    tot_time = 0
    for kernel, time in profile_info:
        print("{:<30} {:<30}".format(kernel, time))
        tot_time += time
    print("Total profiled time: %g secs" % tot_time)

    if not device_only:
        global _cpu_profile_info
        profile_info = sorted(_cpu_profile_info.items(), key=itemgetter(1),
                              reverse=True)
        if len(profile_info) == 0:
            print("No profile information available")
            return
        print("{:<30} {:<30}".format('Kernel', 'Time'))
        tot_time = 0
        for kernel, time in profile_info:
            print("{:<30} {:<30}".format(kernel, time))
            tot_time += time
        print("Total profiled wall-clock time: %g secs" % tot_time)


def reset_profile():
    global _profile_info
    global _cpu_profile_info
    _profile_info = defaultdict(float)
    _cpu_profile_info = defaultdict(float)

def profile_kernel(kernel, name):
    def _profile_knl(*args, **kwargs):
        start = time.time()
        event = kernel(*args, **kwargs)
        profile(name, event)
        end = time.time()
        cpu_profile(name, end - start)
        return event

    if get_config().profile:
        return _profile_knl
    else:
        return kernel


def profile_with_name(name):
    def _decorator(f):
        if name is None:
            n = f.__name__
        else:
            n = name

        def _profiled_kernel_generator(*args, **kwargs):
            kernel = f(*args, **kwargs)
            return profile_kernel(kernel, n)

        return _profiled_kernel_generator

    return _decorator


def get_elwise_kernel(kernel_name, args, src, preamble=""):
    ctx = get_context()
    knl = ElementwiseKernel(
        ctx, args, src,
        kernel_name, preamble=preamble
    )
    return profile_kernel(knl, kernel_name)


class DeviceArray(object):
    def __init__(self, dtype, n=0):
        self.queue = get_queue()
        self.ctx = get_context()
        self.dtype = dtype
        length = n
        if n == 0:
            n = 16
        data = cl.array.empty(self.queue, n, dtype)
        self.minimum = 0
        self.maximum = 0
        self.set_data(data)
        self.length = length
        self._update_array_ref()

    def _update_array_ref(self):
        self.array = self._data[:self.length]

    def resize(self, size):
        self.reserve(size)
        self.length = size
        self._update_array_ref()

    def reserve(self, size):
        if size > self.alloc:
            new_data = cl.array.empty(self.queue, size, self.dtype)
            new_data[:self.alloc] = self._data
            self._data = new_data
            self.alloc = size
            self._update_array_ref()

    def set_data(self, data):
        self._data = data
        self.length = data.size
        self.alloc = data.size
        self.dtype = data.dtype
        self._update_array_ref()

    def get_data(self):
        return self._data

    def copy(self):
        arr_copy = DeviceArray(self.dtype)
        arr_copy.set_data(self.array.copy())
        return arr_copy

    def update_min_max(self):
        self.minimum = float(cl.array.min(self.array).get())
        self.maximum = float(cl.array.max(self.array).get())

    def fill(self, value):
        self.array.fill(value)

    def append(self, value):
        if self.length >= self.alloc:
            self.reserve(2 * self.length)
        self._data[self.length] = value
        self.length += 1
        self._update_array_ref()

    def extend(self, cl_arr):
        if self.length + len(cl_arr) > self.alloc:
            self.reserve(self.length + len(cl_arr))
        self._data[-len(cl_arr):] = cl_arr
        self.length += len(cl_arr)
        self._update_array_ref()

    def remove(self, indices, input_sorted=False):
        if len(indices) > self.length:
            return

        if not input_sorted:
            radix_sort = cl.algorithm.RadixSort(
                self.ctx,
                "unsigned int* indices",
                scan_kernel=GenericScanKernel, key_expr="indices[i]",
                sort_arg_names=["indices"]
            )

            (sorted_indices,), event = radix_sort(indices)

        else:
            sorted_indices = indices

        args = "uint* indices, %(dtype)s* array, uint length" % \
               {"dtype": cl.tools.dtype_to_ctype(self.dtype)}
        src = REMOVE_KNL.render()
        remove = get_elwise_kernel("remove", args, src)

        remove(sorted_indices, self.array, self.length)
        self.length -= len(indices)
        self._update_array_ref()

    def align(self, indices):
        self.set_data(cl.array.take(self.array, indices))

    def squeeze(self):
        self.set_data(self._data[:self.length])

    def copy_values(self, indices, dest):
        dest[:len(indices)] = cl.array.take(self.array, indices)


class DeviceHelper(object):
    """Manages the arrays contained in a particle array on the device.

    Note that it converts the data to a suitable type depending on the value of
    get_config().use_double. Further, note that it assumes that the names of
    constants and properties do not clash.

    """

    def __init__(self, particle_array):
        self._particle_array = pa = particle_array
        self._queue = get_queue()
        self._ctx = get_context()
        use_double = get_config().use_double
        self._dtype = np.float64 if use_double else np.float32
        self._data = {}
        self.properties = []
        self.constants = []

        for prop, ary in pa.properties.items():
            self.add_prop(prop, ary)
        for prop, ary in pa.constants.items():
            self.add_const(prop, ary)

    def _get_array(self, ary):
        ctype = ary.get_c_type()
        if ctype in ['float', 'double']:
            return ary.get_npy_array().astype(self._dtype)
        else:
            return ary.get_npy_array()

    def _get_prop_or_const(self, prop):
        pa = self._particle_array
        return pa.properties.get(prop, pa.constants.get(prop))

    def _add_prop_or_const(self, name, carray):
        """Add a new property or constant given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        np_array = self._get_array(carray)
        g_ary = DeviceArray(np_array.dtype, n=carray.length)
        g_ary.array.set(np_array)
        self._data[name] = g_ary
        setattr(self, name, g_ary.array)

    def _check_property(self, prop):
        """Check if a property is present or not """
        if prop in self.properties:
            return
        else:
            raise AttributeError('property %s not present' % (prop))

    def get_number_of_particles(self, real=False):
        if real:
            return self.num_real_particles
        else:
            if len(self.properties) > 0:
                prop0 = self._data[self.properties[0]]
                return len(prop0.array)
            else:
                return 0

    def align(self, indices):
        for prop in self.properties:
            self._data[prop].align(indices)
            setattr(self, prop, self._data[prop].array)

    def add_prop(self, name, carray):
        """Add a new property given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        self._add_prop_or_const(name, carray)
        if name in self._particle_array.properties:
            self.properties.append(name)

    def add_const(self, name, carray):
        """Add a new constant given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        self._add_prop_or_const(name, carray)
        if name in self._particle_array.constants:
            self.constants.append(name)

    def update_prop(self, name, dev_array):
        """Add a new property to DeviceHelper. Note that this property
        is not added to the particle array itself"""
        self._data[name] = dev_array
        setattr(self, name, dev_array.array)
        if name not in self.properties:
            self.properties.append(name)

    def update_const(self, name, dev_array):
        """Add a new constant to DeviceHelper. Note that this property
        is not added to the particle array itself"""
        self._data[name] = dev_array
        setattr(self, name, dev_array.array)
        if name not in self.constants:
            self.constants.append(name)

    def get_device_array(self, prop):
        if prop in self.properties or prop in self.constants:
            return self._data[prop]

    def max(self, arg):
        return float(cl.array.max(getattr(self, arg)).get())

    def update_min_max(self, props=None):
        """Update the min,max values of all properties """
        if props:
            for prop in props:
                array = self._data[prop]
                array.update_min_max()
        else:
            for prop in self.properties:
                array = self._data[prop]
                array.update_min_max()

    def pull(self, *args):
        if len(args) == 0:
            args = self._data.keys()
        for arg in args:
            self._get_prop_or_const(arg).set_data(
                getattr(self, arg).get()
            )

    def push(self, *args):
        if len(args) == 0:
            args = self._data.keys()
        for arg in args:
            getattr(self, arg).set(
                self._get_array(self._get_prop_or_const(arg))
            )

    def remove_prop(self, name):
        if name in self.properties:
            self.properties.remove(name)
        if name in self._data:
            del self._data[name]
            delattr(self, name)

    def resize(self, new_size):
        for prop in self.properties:
            self._data[prop].resize(new_size)
            setattr(self, prop, self._data[prop].array)

    def align_particles(self):
        tag_arr = self._data['tag'].array

        num_particles = self.get_number_of_particles()
        indices = cl.array.arange(self._queue, 0, num_particles, 1,
                                  dtype=np.uint32)

        radix_sort = cl.algorithm.RadixSort(
            self._ctx,
            "unsigned int* indices, unsigned int* tags",
            scan_kernel=GenericScanKernel, key_expr="tags[i]",
            sort_arg_names=["indices"]
        )

        (sorted_indices,), event = radix_sort(indices, tag_arr, key_bits=2)
        self.align(sorted_indices)

        tag_arr = self._data['tag'].array

        num_real_particles = cl.array.zeros(self._queue, 1, np.uint32)
        args = "uint* tag_array, uint* num_real_particles"
        src = NUM_REAL_PARTICLES_KNL.render()
        get_num_real_particles = get_elwise_kernel(
            "get_num_real_particles", args, src)

        get_num_real_particles(tag_arr, num_real_particles)
        self.num_real_particles = int(num_real_particles.get())

    def remove_particles(self, indices):
        """ Remove particles whose indices are given in index_list.

        We repeatedly interchange the values of the last element and
        values from the index_list and reduce the size of the array
        by one. This is done for every property that is being maintained.

        Parameters
        ----------

        indices : array
            an array of indices, this array can be a list, numpy array
            or a LongArray.

        Notes
        -----

        Pseudo-code for the implementation::

            if index_list.length > number of particles
                raise ValueError

            sorted_indices <- index_list sorted in ascending order.

            for every every array in property_array
                array.remove(sorted_indices)

        """
        if len(indices) > self.get_number_of_particles():
            msg = 'Number of particles to be removed is greater than'
            msg += 'number of particles in array'
            raise ValueError(msg)

        radix_sort = cl.algorithm.RadixSort(
            self._ctx,
            "unsigned int* indices",
            scan_kernel=GenericScanKernel, key_expr="indices[i]",
            sort_arg_names=["indices"]
        )

        (sorted_indices,), event = radix_sort(indices)

        for prop in self.properties:
            self._data[prop].remove(sorted_indices, 1)
            setattr(self, prop, self._data[prop].array)

        if len(indices) > 0:
            self.align_particles()

    def remove_tagged_particles(self, tag):
        """ Remove particles that have the given tag.

        Parameters
        ----------

        tag : int
            the type of particles that need to be removed.

        """
        tag_array = getattr(self, 'tag')

        remove_places = tag_array == tag
        num_indices = int(cl.array.sum(remove_places).get())

        if num_indices == 0:
            return

        indices = cl.array.empty(self._queue, num_indices, np.uint32)
        head = cl.array.zeros(self._queue, 1, np.uint32)

        args = "uint* tag_array, uint tag, uint* indices, uint* head"
        src = REMOVE_INDICES_KNL.render()

        # find the indices of the particles to be removed.
        remove_indices = get_elwise_kernel("remove_indices", args, src)

        remove_indices(tag_array, tag, indices, head)

        # remove the particles.
        self.remove_particles(indices)

    def add_particles(self, **particle_props):
        """
        Add particles in particle_array to self.

        Parameters
        ----------

        particle_props : dict
            a dictionary containing cl arrays for various particle
            properties.

        Notes
        -----

         - all properties should have same length arrays.
         - all properties should already be present in this particles array.
           if new properties are seen, an exception will be raised.
           properties.

        """
        pa = self._particle_array

        if len(particle_props) == 0:
            return

        # check if the input properties are valid.
        for prop in particle_props:
            self._check_property(prop)

        num_extra_particles = len(list(particle_props.values())[0])
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        for prop in self.properties:
            arr = self._data[prop]

            if prop in particle_props.keys():
                s_arr = particle_props[prop]
                arr.extend(s_arr)
            else:
                arr.resize(new_num_particles)
                # set the properties of the new particles to the default ones.
                arr.array[old_num_particles:] = pa.default_values[prop]

            self.update_prop(prop, arr)

        if num_extra_particles > 0:
            # make sure particles are aligned properly.
            self.align_particles()

    def extend(self, num_particles):
        """ Increase the total number of particles by the requested amount

        New particles are added at the end of the list, you may
        have to manually call align_particles later.
        """
        if num_particles <= 0:
            return

        old_size = self.get_number_of_particles()
        new_size = old_size + num_particles

        for prop in self.properties:
            arr = self._data[prop]
            arr.resize(new_size)
            arr.array[old_size:] = self._particle_array.default_values[prop]
            self.update_prop(prop, arr)

    def append_parray(self, parray):
        """ Add particles from a particle array

        properties that are not there in self will be added
        """
        if parray.gpu is None:
            parray.set_device_helper(DeviceHelper(parray))

        if parray.gpu.get_number_of_particles() == 0:
            return

        num_extra_particles = parray.gpu.get_number_of_particles()
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        # extend current arrays by the required number of particles
        self.extend(num_extra_particles)

        for prop_name in parray.gpu.properties:
            if prop_name in self.properties:
                arr = self._data[prop_name]
                source = parray.gpu.get_device_array(prop_name)
                arr.array[old_num_particles:] = source.array
            else:
                # meaning this property is not there in self.
                dtype = self._data[prop_name].dtype
                arr = DeviceArray(dtype, n=new_num_particles)
                arr.fill(parray.gpu._particle_array.default_values[prop_name])
                self.update_prop(prop_name, arr)

                # now add the values to the end of the created array
                dest = self._data[prop_name]
                source = parray.gpu.get_device_array(prop_name)
                dest.array[old_num_particles:] = source.array

        for const in parray.gpu.constants:
            if const not in self.constants:
                arr = parray.gpu.get_device_array(const)
                self.update_const(const, arr.copy())

        if num_extra_particles > 0:
            self.align_particles()

    def extract_particles(self, indices, props=None):
        """Create new particle array for particles with given indices

        Parameters
        ----------

        indices : cl.array.Array
            indices of particles to be extracted.

        props : list
            the list of properties to extract, if None all properties
            are extracted.

        """
        result_array = ParticleArray()
        result_array.set_device_helper(DeviceHelper(result_array))

        if props is None:
            prop_names = self.properties
        else:
            prop_names = props

        if len(indices) == 0:
            return result_array

        for prop_name in prop_names:
            src_arr = self._data[prop_name]
            dst_arr = DeviceArray(src_arr.dtype, n=len(indices))
            src_arr.copy_values(indices, dst_arr.array)

            prop_type = cl.tools.dtype_to_ctype(src_arr.dtype)
            prop_default = self._particle_array.default_values[prop_name]
            result_array.add_property(name=prop_name,
                                      type=prop_type,
                                      default=prop_default)

            result_array.gpu.update_prop(prop_name, dst_arr)

        for const in self.constants:
            result_array.gpu.update_const(const, self._data[const].copy())

        result_array.gpu.align_particles()
        result_array.set_name(self._particle_array.name)

        if props is None:
            output_arrays = list(self._particle_array.output_property_arrays)
        else:
            output_arrays = list(
                set(props).intersection(
                    self._particle_array.output_property_arrays
                )
            )

        result_array.set_output_arrays(output_arrays)
        return result_array


def get_simple_elwise_program(context, arguments, operation,
                              name="elwise_kernel", options=[],
                              preamble="", **kwargs):
    source = ("""//CL//
        %(preamble)s
        #define PYOPENCL_ELWISE_CONTINUE continue
        __kernel void %(name)s(%(arguments)s)
        {
          int lid = get_local_id(0);
          int gsize = get_global_size(0);
          int work_group_start = get_local_size(0)*get_group_id(0);
          int i = get_global_id(0);
          %(body)s
        }
        """ % {
        "arguments": ", ".join(arg.declarator() for arg in arguments),
        "name": name,
        "preamble": preamble,
        "body": operation,
    })

    from pyopencl import Program
    return Program(context, source).build(options)


def get_simple_elwise_kernel_and_types(context, arguments, operation,
                                       name="elwise_kernel", options=[],
                                       preamble="",
                                       **kwargs):
    from pyopencl.tools import parse_arg_list

    parsed_args = parse_arg_list(arguments, with_offset=True)

    pragmas = []
    includes = []

    if pragmas or includes:
        preamble = "\n".join(pragmas + includes) + "\n" + preamble

    parsed_args.append(ScalarArg(np.intp, "n"))

    prg = get_simple_elwise_program(
        context, parsed_args, operation,
        name=name, options=options, preamble=preamble,
        **kwargs)

    from pyopencl.tools import get_arg_list_scalar_arg_dtypes

    kernel = getattr(prg, name)
    kernel.set_scalar_arg_dtypes(get_arg_list_scalar_arg_dtypes(parsed_args))

    return kernel, parsed_args


def get_simple_elwise_kernel(context, arguments, operation,
                             name="elwise_kernel", options=[], **kwargs):
    func, arguments = get_simple_elwise_kernel_and_types(
        context, arguments, operation,
        name=name, options=options, **kwargs)

    return func


class SimpleElementwiseKernel(object):
    """PyOpenCL's elementwise kernel without any of its magic.

    PyOpenCL's elementwise kernel without most of its magic except for:
    1) Default variables set (i)
    2) Size determined from 1st parameter
    """

    def __init__(self, context, arguments, operation,
                 name="elwise_kernel", options=[], **kwargs):
        self.context = context
        self.arguments = arguments
        self.operation = operation
        self.name = name
        self.options = options
        self.kwargs = kwargs

    def get_kernel(self):
        knl, arg_descrs = get_simple_elwise_kernel_and_types(
            self.context, self.arguments, self.operation,
            name=self.name, options=self.options,
            **self.kwargs)

        for arg in arg_descrs:
            if isinstance(arg, VectorArg) and not arg.with_offset:
                from warnings import warn
                warn("ElementwiseKernel '%s' used with VectorArgs that do not "
                     "have offset support enabled. This usage is deprecated. "
                     "Just pass with_offset=True to VectorArg, everything should "
                     "sort itself out automatically." % self.name,
                     DeprecationWarning)

        if not [i for i, arg in enumerate(arg_descrs)
                if isinstance(arg, VectorArg)]:
            raise RuntimeError(
                "ElementwiseKernel can only be used with "
                "functions that have at least one "
                "vector argument")
        return knl, arg_descrs

    def __call__(self, *args, **kwargs):
        repr_vec = None

        kernel, arg_descrs = self.get_kernel()

        # {{{ assemble arg array

        invocation_args = []
        for arg, arg_descr in zip(args, arg_descrs):
            if isinstance(arg_descr, VectorArg):
                if not arg.flags.forc:
                    raise RuntimeError("ElementwiseKernel cannot "
                                       "deal with non-contiguous arrays")

                if repr_vec is None:
                    repr_vec = arg

                invocation_args.append(arg.base_data)
                if arg_descr.with_offset:
                    invocation_args.append(arg.offset)
            else:
                invocation_args.append(arg)

        # }}}

        queue = kwargs.pop("queue", None)
        wait_for = kwargs.pop("wait_for", None)
        if kwargs:
            raise TypeError("unknown keyword arguments: '%s'"
                            % ", ".join(kwargs))

        if queue is None:
            queue = repr_vec.queue

        max_wg_size = kernel.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE,
            queue.device)

        # Last arg: n
        invocation_args.append(repr_vec.size)
        gs, ls = repr_vec.get_sizes(queue, max_wg_size)

        kernel.set_args(*invocation_args)
        return cl.enqueue_nd_range_kernel(queue, kernel,
                                          gs, ls, wait_for=wait_for)
