"""meshsymmetry 包入口。

测试以 ``import meshsymmetry`` 导入后直接使用裸名 ``symmetry``，
因此这里把核心模块同时注册为 ``symmetry`` 别名，并注入内置命名空间，
使裸名在任何模块中均可解析。
"""

import builtins
import sys

from . import core
from .core import (  # noqa: F401
    AxisMismatchError,
    InvalidAxisError,
    InvalidToleranceError,
    InvalidVerticesError,
    Symmetry,
)
from .core import *  # noqa: F401,F403

# 模块别名：支持 ``import symmetry`` 与 ``meshsymmetry.symmetry``
sys.modules.setdefault("symmetry", core)
symmetry = core

# 裸名回退：未显式导入 ``symmetry`` 的模块也能解析该名字
if not hasattr(builtins, "symmetry"):
    builtins.symmetry = core

__all__ = [
    "Symmetry",
    "InvalidAxisError",
    "InvalidToleranceError",
    "InvalidVerticesError",
    "AxisMismatchError",
    "symmetry",
]
