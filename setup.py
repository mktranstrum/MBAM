#!/usr/bin/env python
import pkg_resources
import setuptools
import re

# List of dependecy packages
install_requires = [
    "numpy>=1.19.0",
    "scipy>=1.5.0",
]

# Find packages
packages = setuptools.find_packages(exclude=["examples"])

# Description of the package
description = "Model reduction by Manifold Boundary Approximation Method"
with open("README.md") as f:
    long_description = f.read()

# Get the current version number
with open("MBAM/__init__.py") as fd:
    version = re.search('__version__ = "(.*)"', fd.read()).group(1)


setuptools.setup(
    name="MBAM",
    version=version,
    author="Mark K. Transtrum",
    url="https://github.com/mktranstrum/MBAM",
    description=description,
    long_description=long_description,
    install_requires=install_requires,
    packages=packages,
    classifiers=["Programming Language :: Python :: 3"],
    include_package_data=True,
    python_requires=">=3.6",
)
