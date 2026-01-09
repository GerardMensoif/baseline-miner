from __future__ import annotations

import sys

from setuptools import Extension, find_packages, setup


def _ext_modules() -> list[Extension]:
    extra_args: list[str] = []
    if sys.platform == "win32":
        extra_args.append("/O2")
    else:
        extra_args.extend(["-O3", "-funroll-loops"])
    return [
        Extension(
            "baseline_miner._sha256d",
            sources=["baseline_miner/native/sha256d.c"],
            extra_compile_args=extra_args,
        )
    ]


setup(
    packages=find_packages(),
    ext_modules=_ext_modules(),
)
