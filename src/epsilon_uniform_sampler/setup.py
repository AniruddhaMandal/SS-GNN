from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Get include directory
include_dir = os.path.join(os.path.dirname(__file__), 'include')

setup(
    name='epsilon_uniform_sampler',
    version='1.0.0',
    author='Aniruddha Mandal',
    description='Epsilon-uniform connected subgraph sampler via random walk with rejection sampling',
    ext_modules=[
        CppExtension(
            name='epsilon_uniform_sampler',
            sources=['src/epsilon_uniform_sampler.cpp'],
            include_dirs=[include_dir],
            extra_compile_args=['-O3', '-std=c++17', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
