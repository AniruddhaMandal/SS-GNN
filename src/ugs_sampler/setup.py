# setup.py -- build C++ pybind11 extension using PyTorch's build tooling
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

# Sources to compile (adjust if you split files or change paths)
sources = [
    "src/extension.cpp",
    "src/preproc.cpp",
    "src/sampler.cpp",
    "src/ugs_sampler_batch_extension.cpp",
]

# Optionally include vendored pybind11 headers (if you put pybind11 in third_party/)
extra_include_dirs = []
# If you vendor pybind11 at third_party/pybind11, add its include dir
pybind11_vend = os.path.join("third_party", "pybind11", "include")
if os.path.isdir(pybind11_vend):
    extra_include_dirs.append(pybind11_vend)

extra_include_dirs.append("include")

ext_modules = [
    CppExtension(
        name="ugs_sampler",
        sources=sources,
        include_dirs=extra_include_dirs,
        extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    ),
]

setup(
    name="ugs_sampler",
    version="2.0.1",
    description="UGS sampler C++ extension (pybind11 + PyTorch cpp extension)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
