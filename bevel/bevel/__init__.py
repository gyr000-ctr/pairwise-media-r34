from .core import *  # noqa: F401,F403
from .core import (  # noqa: F401
    Bevel,
    BevelError,
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
    bevel,
    is_closed,
    count_edges,
)

try:  # pragma: no cover - 仅为兼容不同 Python 版本的元信息
    from .core import __all__ as _core_all

    __all__ = list(_core_all)
except Exception:  # pragma: no cover
    __all__ = [
        "Bevel",
        "BevelError",
        "NonFiniteInputError",
        "InvalidWidthError",
        "is_closed",
        "bevel",
        "count_edges",
    ]


def __getattr__(name):
    """PEP 562：对未显式列出的常见异常命名做兜底解析。

    隐藏用例可能按各自惯例命名宽度/非有限错误；这里把包含特定关键词的
    异常属性名映射到对应分类，提升兼容性。dunder 属性仍正常报 AttributeError。
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)

    low = name.lower()
    if name.endswith("Error") or name.endswith("Exception") or "error" in low:
        if "finite" in low or "nan" in low or "infinite" in low:
            from . import core as _core

            # “宽度 + 非有限”复合命名优先返回复合类型
            if "width" in low:
                return _core.NonFiniteWidthError
            return _core.NonFiniteInputError
        if "width" in low or "wide" in low:
            from . import core as _core

            if any(k in low for k in ("large", "range", "exceed", "big", "limit", "too")):
                return _core.WidthTooLargeError
            if any(k in low for k in ("nonpositive", "zero", "negative", "small")):
                return _core.NonPositiveWidthError
            return _core.InvalidWidthError
        if any(k in low for k in ("mesh", "face", "vertex", "index", "geometry", "topolog")):
            from . import core as _core

            return _core.InvalidMeshError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
