from mako.template import Template
import os
import sys

from pysph.cpy.opencl import get_context, profile_kernel


def get_elwise_kernel(kernel_name, args, src, preamble=""):
    ctx = get_context()
    from pyopencl.elementwise import ElementwiseKernel
    knl = ElementwiseKernel(
        ctx, args, src,
        kernel_name, preamble=preamble
    )
    return profile_kernel(knl, kernel_name)


class GPUNNPSHelper(object):
    def __init__(self, tpl_filename, backend=None, use_double=False):
        disable_unicode = False if sys.version_info.major > 2 else True
        self.src_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                tpl_filename),
            disable_unicode=disable_unicode
        )

        self.data_t = "double" if use_double else "float"

        helper_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "gpu_helper_functions.mako"),
            disable_unicode=disable_unicode
        )

        helper_preamble = helper_tpl.get_def("get_helpers").render(
            data_t=self.data_t
        )
        preamble = self.src_tpl.get_def("preamble").render(
            data_t=self.data_t
        )
        self.preamble = "\n".join([helper_preamble, preamble])
        self.cache = {}
        self.backend = backend

    def _get_code(self, kernel_name, **kwargs):
        arguments = self.src_tpl.get_def("%s_args" % kernel_name).render(
            data_t=self.data_t, **kwargs)

        src = self.src_tpl.get_def("%s_src" % kernel_name).render(
            data_t=self.data_t, **kwargs)

        return arguments, src

    def get_kernel(self, kernel_name, **kwargs):
        data = kernel_name, tuple(kwargs.items())
        if data in self.cache:
            return self.cache[data]
        else:
            args, src = self._get_code(kernel_name, **kwargs)
            knl = get_elwise_kernel(kernel_name, args, src,
                                    preamble=self.preamble)
            self.cache[data] = knl
            return knl
