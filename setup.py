from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent

setup(
    name="nomad_projection", 
    version="0.0.1", 
    description="",
    long_description="",
    author="",
    author_email="",
    url="", 
    packages=find_packages(), 
    classifiers=[
    ],
    python_requires=">=3.6",
    install_requires=[
        'torch',
        'click',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'tqdm',
        'nomic',
    ],
    extras_require={
    },
    include_package_data=True,  # Include package data specified in MANIFEST.in
)
