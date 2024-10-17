from io import open
from os import path

from setuptools import find_packages, setup

import versioneer

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="alfabet-lite",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A library to estimate bond dissociation energies (BDEs) of organic molecules",
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kangqiao-ctrl/alfabet_lite",  # Optional
    author="Peter St. John, Qiao Kang",
    author_email="peter.stjohn@nrel.gov, kangqiao@gmail.com",  # Optional
    classifiers=[
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13"
    ],
    packages=find_packages(exclude=["docs", "tests"]),  # Required
    install_requires=[
        "pandas",
        "nfp>=0.3.6",
        "tqdm",
        "pooch",
        "joblib",
        "rdkit",
    ],
    extras_require={
        "tensorflow": ["tensorflow==2.15"],
        "apple_silicon": ["tensorflow-metal"],
    },
    project_urls={
        "Source": "https://github.com/kangqiao-ctrl/alfabet_lite",
    },
)
