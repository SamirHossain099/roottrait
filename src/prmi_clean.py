"""Compatibility shim: the implementation is `roottrait.clean`, the published package.

The analysis scripts in this directory import `prmi_clean` by its old flat name; this keeps them working.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import importlib  # noqa: E402

# By full path: the package re-exports a function named `decompose`, which hides the submodule.
_impl = importlib.import_module("roottrait.clean")

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
