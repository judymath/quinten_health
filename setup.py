"""
This is the setup.py script for the 'health' package.

It defines the package metadata and requirements for installation.

Author: Quinten_team_1
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="health",
    version="0.0.1",
    author="Quinten_team_1",
    author_email="Quinten_team_1@Quinten_team_1.com",
    description="Data analysis of health comments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
