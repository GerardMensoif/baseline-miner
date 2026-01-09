from __future__ import annotations

import os
import sys

import platform
import sysconfig

from setuptools import Extension, find_packages, setup


def _ext_modules() -> list[Extension]:
    extra_args: list[str] = []
    native = os.environ.get("BASELINE_MINER_NATIVE", "").strip() not in ("", "0", "false", "False", "no", "NO")
    if sys.platform == "win32":
        extra_args.append("/O2")
        if native:
            extra_args.extend(["/arch:AVX2", "/favor:INTEL64"])
    else:
        extra_args.extend(["-O3", "-funroll-loops"])
        if native:
            extra_args.extend(["-march=native", "-mtune=native"])
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
