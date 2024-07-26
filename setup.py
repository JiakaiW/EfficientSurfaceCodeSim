import os

from setuptools import setup, find_namespace_packages

DIST_NAME = "surface_erasure_sim"
PACKAGE_NAME = "surface_erasure_sim"

REQUIREMENTS = [
    "numpy",
    "stim",
    "pymatching",
]
EXTRA_REQUIREMENTS = [

]
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), PACKAGE_NAME, "VERSION.txt")
)

with open(version_path, "r") as fd:
    version_str = fd.read().rstrip()

setup(
    name=DIST_NAME,
    version=version_str,
    description=DIST_NAME,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/JiakaiW/surface_erasure_decoding",
    author="Jiakai Wang",
    author_email="jwang2648@wisc.edu",
    license="Apache 2.0",
    packages=find_namespace_packages(exclude=None),
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    keywords="surface code erasure decoding",
    python_requires=">=3.7",
    project_urls={
        # "Documentation": "https://github.com/JiakaiW/surface_erasure_decoding",
        "Source Code": "https://github.com/JiakaiW/surface_erasure_decoding",
        # "Tutorials": "https://github.com/EQuS/jaxquantum/tree/main/tutorials",
    },
    include_package_data=True,
)