from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Get include directory
include_dir = os.path.join(os.path.dirname(__file__), 'include')

setup(
    name='uniform_sampler',
    version='1.0.0',
    author='Aniruddha Mandal',
    description='Truly uniform connected subgraph sampler via exhaustive enumeration',
    ext_modules=[
        CppExtension(
            name='uniform_sampler',
            sources=['src/uniform_sampler.cpp'],
            include_dirs=[include_dir],
            extra_compile_args=['-O3', '-std=c++17'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
