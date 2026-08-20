from __future__ import annotations

import sys

import numpy as np
from setuptools import Extension, setup

if sys.platform == "win32":
    compile_args = ["/O2", "/std:c++17", "/EHsc"]
else:
    compile_args = ["-O3", "-std=c++17"]

setup(
    package_dir={"": "src"},
    packages=["fastlabelcontours"],
    package_data={"fastlabelcontours": ["py.typed", "_core.pyi"]},
    ext_modules=[
        Extension(
            "fastlabelcontours._core",
            ["src/fastlabelcontours/_core.cpp"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=compile_args,
        )
    ],
)
