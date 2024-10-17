#!/usr/bin/env python3
import os
import platform
import re
import subprocess

from setuptools import find_packages, setup

# TODO
setup(
    name="cpse",
    version="0.1",
    description="",
    author="",
    author_email="",
    install_requires=["unified_planning"],
    packages=find_packages(include=["cpse"]),
    package_data={"": ["cpse.py"]},
    license="MIT",
)
