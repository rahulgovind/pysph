"""
PySPH
=====

A general purpose Smoothed Particle Hydrodynamics framework.

This package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.  See here:

    http://pysph.googlecode.com

for more information.
"""

import numpy
import commands
import os
import sys
from os import path

from setuptools import find_packages, setup
from Cython.Distutils import build_ext, Extension

Have_MPI = True
try:
    import mpi4py
except ImportError:
    Have_MPI = False

USE_ZOLTAN=True

compiler = 'gcc'
#compiler = 'intel'
if compiler == 'intel':
    extra_compile_args = ['-O3']
else:
    extra_compile_args = []


mpi_inc_dirs = []
mpi_compile_args = []
mpi_link_args = []

def get_deps(*args):
    """Given a list of basenames, this checks if a .pyx or .pxd exists
    and returns the list.
    """
    result = []
    for basename in args:
        for ext in ('.pyx', '.pxd'):
            f = basename + ext
            if path.exists(f):
                result.append(f)
    return result

def get_openmp_flags():
    """Returns any OpenMP related flags if OpenMP is avaiable on the system.
    """
    if sys.platform == 'win32':
        omp_flags = ['/openmp']
    else:
        omp_flags = ['-fopenmp']

    env_var = os.environ.get('USE_OPENMP', '')
    if env_var.lower() in ("0", 'false', 'n'):
        print("-"*70)
        print("OpenMP disabled by environment variable (USE_OPENMP).")
        return [], []

    from textwrap import dedent
    from pyximport import pyxbuild
    from distutils.errors import CompileError, LinkError
    import shutil
    import tempfile
    test_code = dedent("""
    from cython.parallel import parallel, prange, threadid
    cimport openmp
    def n_threads():
        with nogil, parallel():
            openmp.omp_get_num_threads()
    """)
    tmp_dir = tempfile.mkdtemp()
    fname = path.join(tmp_dir, 'check_omp.pyx')
    with open(fname, 'w') as fp:
        fp.write(test_code)
    extension = Extension(
        name='check_omp', sources=[fname],
        extra_compile_args=omp_flags,
        extra_link_args=omp_flags,
    )
    has_omp = True
    try:
        mod = pyxbuild.pyx_to_dll(fname, extension, pyxbuild_dir=tmp_dir)
        print("-"*70)
        print("Using OpenMP.")
        print("-"*70)
    except CompileError:
        print("*"*70)
        print("Unable to compile OpenMP code. Not using OpenMP.")
        print("*"*70)
        has_omp = False
    except LinkError:
        print("*"*70)
        print("Unable to link OpenMP code. Not using OpenMP.")
        print("*"*70)
        has_omp = False
    finally:
        shutil.rmtree(tmp_dir)

    if has_omp:
        return omp_flags, omp_flags, True
    else:
        return [], [], False


def get_zoltan_directory(varname):
    global USE_ZOLTAN
    d = os.environ.get(varname, '')
    if ( len(d) == 0 ):
        USE_ZOLTAN=False
        return ''
    if not path.exists(d):
        print("*"*80)
        print("%s incorrectly set to %s, not using ZOLTAN!"%(varname, d))
        print("*"*80)
        USE_ZOLTAN=False
        return ''
    return d


openmp_compile_args, openmp_link_args, openmp_env = get_openmp_flags()

if Have_MPI:
    mpic = 'mpic++'
    if compiler == 'intel':
        link_args = commands.getoutput(mpic + ' -cc=icc -link_info')
        link_args = link_args[3:]
        compile_args = commands.getoutput(mpic +' -cc=icc -compile_info')
        compile_args = compile_args[3:]
    else:
        link_args = commands.getoutput(mpic + ' --showme:link')
        compile_args = commands.getoutput(mpic +' --showme:compile')
    mpi_link_args.append(link_args)
    mpi_compile_args.append(compile_args)
    mpi_inc_dirs.append(mpi4py.get_include())

    # First try with the environment variable 'ZOLTAN'
    zoltan_base = get_zoltan_directory('ZOLTAN')
    inc = lib = ''
    if len(zoltan_base) > 0:
        inc = path.join(zoltan_base, 'include')
        lib = path.join(zoltan_base, 'lib')
        if not path.exists(inc) or not path.exists(lib):
            inc = lib = ''

    # try with the older ZOLTAN include directories
    if len(inc) == 0 or len(lib) == 0:
        inc = get_zoltan_directory('ZOLTAN_INCLUDE')
        lib = get_zoltan_directory('ZOLTAN_LIBRARY')

    if (not USE_ZOLTAN):
        print("*"*80)
        print("Zoltan Environment variable" \
              "not set, not using ZOLTAN!")
        print("*"*80)
        Have_MPI = False
    else:
        print('-'*70)
        print("Using Zoltan from:\n%s\n%s"%(inc, lib))
        print('-'*70)
        zoltan_include_dirs = [ inc ]
        zoltan_library_dirs = [ lib ]

        # PyZoltan includes
        zoltan_cython_include = [ os.path.abspath('./pyzoltan/czoltan') ]
        zoltan_include_dirs += zoltan_cython_include

include_dirs = [numpy.get_include()]

cmdclass = {'build_ext': build_ext}

ext_modules = [
    Extension(
        name="pyzoltan.core.carray",
        sources=["pyzoltan/core/carray.pyx"],
        include_dirs = include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++"
    ),

    Extension(
        name="pysph.base.particle_array",
        sources=["pysph/base/particle_array.pyx"],
        depends=get_deps("pyzoltan/core/carray"),
        extra_compile_args=extra_compile_args,
        language="c++"
    ),

    Extension(
        name="pysph.base.point",
        sources=["pysph/base/point.pyx"],
        extra_compile_args=extra_compile_args,
        language="c++"
    ),

    Extension(
        name="pysph.base.nnps",
        sources=["pysph/base/nnps.pyx"],
        depends=get_deps(
            "pyzoltan/core/carray", "pysph/base/point",
            "pysph/base/particle_array",
        ),
        extra_compile_args=extra_compile_args + openmp_compile_args,
        extra_link_args=openmp_link_args,
        cython_compile_time_env={'OPENMP': openmp_env},
        language="c++"
    ),

    # kernels used for tests
    Extension(
        name="pysph.base.c_kernels",
        sources=["pysph/base/c_kernels.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++"
    ),

    # Eigen decomposition code
    Extension(
        name="pysph.sph.solid_mech.linalg",
        sources=["pysph/sph/solid_mech/linalg.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++"
    ),
]

# add the include dirs for the extension modules
for ext in ext_modules:
    ext.include_dirs = include_dirs

if Have_MPI:
    zoltan_modules = [
        Extension(
            name="pyzoltan.core.zoltan",
            sources=["pyzoltan/core/zoltan.pyx"],
            depends=get_deps(
                "pyzoltan/core/carray",
                "pyzoltan/czoltan/czoltan",
                "pyzoltan/czoltan/czoltan_types",
            ),
            include_dirs = include_dirs+zoltan_include_dirs+mpi_inc_dirs,
            library_dirs = zoltan_library_dirs,
            libraries=['zoltan', 'mpi'],
            extra_link_args=mpi_link_args,
            extra_compile_args=mpi_compile_args+extra_compile_args,
            language="c++"
        ),

        Extension(
            name="pyzoltan.core.zoltan_dd",
            sources=["pyzoltan/core/zoltan_dd.pyx"],
            depends=get_deps(
                "pyzoltan/czoltan/czoltan_dd",
                "pyzoltan/czoltan/czoltan_types"
            ),
            include_dirs = include_dirs + zoltan_include_dirs + mpi_inc_dirs,
            library_dirs = zoltan_library_dirs,
            libraries=['zoltan', 'mpi'],
            extra_link_args=mpi_link_args,
            extra_compile_args=mpi_compile_args+extra_compile_args,
            language="c++"
        ),

        Extension(
            name="pyzoltan.core.zoltan_comm",
            sources=["pyzoltan/core/zoltan_comm.pyx"],
            depends=get_deps("pyzoltan/czoltan/zoltan_comm"),
            include_dirs = include_dirs + zoltan_include_dirs + mpi_inc_dirs,
            library_dirs = zoltan_library_dirs,
            libraries=['zoltan', 'mpi'],
            extra_link_args=mpi_link_args,
            extra_compile_args=mpi_compile_args+extra_compile_args,
            language="c++"
        ),
    ]

    parallel_modules = [

        Extension(
            name="pysph.parallel.parallel_manager",
            sources=["pysph/parallel/parallel_manager.pyx"],
            depends=get_deps("pyzoltan/core/carray",
                "pyzoltan/core/zoltan", "pyzoltan/core/zoltan_comm",
                "pysph/base/point", "pysph/base/particle_array",
                "pysph/base/nnps"
            ),
            include_dirs = include_dirs + mpi_inc_dirs + zoltan_include_dirs,
            library_dirs = zoltan_library_dirs,
            libraries = ['zoltan', 'mpi'],
            extra_link_args=mpi_link_args,
            extra_compile_args=mpi_compile_args+extra_compile_args,
            language="c++"
        ),
    ]

    ext_modules += zoltan_modules + parallel_modules

if 'build_ext' in sys.argv or 'develop' in sys.argv or 'install' in sys.argv:
    for pth in (path.join('pyzoltan', 'core'), path.join('pysph', 'base')):
        generator = path.join( path.abspath('.'), path.join(pth, 'generator.py'))
        d = {'__file__': generator }
        execfile(generator, d)
        d['main'](None)


setup(name='PySPH',
      version = '1.0alpha',
      author = 'PySPH Developers',
      author_email = 'pysph-dev@googlegroups.com',
      description = "A general purpose Smoothed Particle Hydrodynamics framework",
      long_description = __doc__,
      url = 'http://pysph.googlecode.com',
      license = "BSD",
      keywords = "SPH simulation computational fluid dynamics",
      test_suite = "nose.collector",
      packages = find_packages(),
      # include Cython headers in the install directory
      package_data={'' : ['*.pxd', '*.mako']},

      ext_modules = ext_modules,

      include_package_data = True,
      cmdclass=cmdclass,
      #install_requires=['mpi4py>=1.2', 'numpy>=1.0.3', 'Cython>=0.14'],
      #setup_requires=['Cython>=0.14', 'setuptools>=0.6c1'],
      #extras_require={'3D': 'Mayavi>=3.0'},
      zip_safe = False,
      entry_points = """
          [console_scripts]
          pysph_viewer = pysph.tools.mayavi_viewer:main
          """,
      platforms=['Linux', 'Mac OS-X', 'Unix', 'Windows'],
      classifiers = [c.strip() for c in """\
        Development Status :: 3 - Alpha
        Environment :: Console
        Intended Audience :: Developers
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Natural Language :: English
        Operating System :: MacOS :: MacOS X
        Operating System :: Microsoft :: Windows
        Operating System :: POSIX
        Operating System :: Unix
        Programming Language :: Python
        Topic :: Scientific/Engineering
        Topic :: Scientific/Engineering :: Physics
        Topic :: Software Development :: Libraries
        """.splitlines() if len(c.split()) > 0],
      )
