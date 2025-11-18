# setup.py
import os
from setuptools import setup, find_packages

# Read long description from README if present
this_dir = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(this_dir, "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="synthetic_dataset",
    version="0.1.0",
    description="Synthetic graph datasets (triangle parity, K4/clique generators) compatible with PyG.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Aniruddha Mandal",
    author_email="ani96dh@gmail.com",
    url="https://github.com/AniruddhaMandal/SS-GNN/tree/master/src/synthetic-dataset", 
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8",
        "torch-geometric>=2.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="graph datasets pytorch pytorch-geometric synthetic graphs",
    zip_safe=False,
)
