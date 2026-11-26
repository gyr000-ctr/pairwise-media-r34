"""异常类型的兼容入口。

规范实现位于 :mod:`bevel.core`；这里重新导出全部公开异常，使
``from bevel.errors import InvalidWidthError`` 等引用方式同样可用。
"""

from .core import (  # noqa: F401
    BevelError,
    BevelException,
    InvalidMeshError,
    EmptyMeshError,
    InvalidVertexError,
    InvalidFaceError,
    IndexOutOfRangeError,
    NonFiniteInputError,
    InvalidWidthError,
    NonPositiveWidthError,
    WidthTooLargeError,
    NonFiniteWidthError,
    WidthError,
    BadWidthError,
    IllegalWidthError,
    BevelWidthError,
    NonFiniteError,
    NotFiniteError,
    NaNInputError,
    InfiniteInputError,
)
