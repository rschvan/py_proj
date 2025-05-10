import os
from setuptools import setup, find_packages

setup(
    name="pypf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'flask',
        'flask-cors',
        'numpy'
    ],
    author="Roger Schvaneveldt",
    author_email="schvan@yahoo.com",
    description="A package for Pathfinder Network creation",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)