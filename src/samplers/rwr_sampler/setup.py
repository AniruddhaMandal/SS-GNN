from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

include_dir = os.path.join(os.path.dirname(__file__), 'include') if os.path.isdir('include') else None
sources = ['src/rwr_sampler.cpp']

ext_kwargs = {
    'name': 'rwr_sampler',
    'sources': sources,
    'extra_compile_args': ['-O3', '-std=c++17', '-fopenmp'],
    'extra_link_args': ['-fopenmp'],
}
if include_dir:
    ext_kwargs['include_dirs'] = [include_dir]

setup(
    name='rwr_sampler',
    version='0.1.0',
    author='Aniruddha Mandal',
    description='RWR connected induced subgraph sampler for PyTorch (cpp/pybind11)',
    ext_modules=[CppExtension(**ext_kwargs)],
    cmdclass={'build_ext': BuildExtension},
)
