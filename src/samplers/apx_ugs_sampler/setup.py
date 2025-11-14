from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Get the include directory
include_dir = os.path.join(os.path.dirname(__file__), 'include')

setup(
    name='apx_ugs_sampler',
    ext_modules=[
        CppExtension(
            name='apx_ugs_sampler',
            sources=['src/apx_ugs_sampler.cpp'],
            include_dirs=[include_dir],
            extra_compile_args=['-O3', '-std=c++17', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
