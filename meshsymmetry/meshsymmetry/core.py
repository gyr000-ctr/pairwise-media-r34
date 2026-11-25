"""meshsymmetry 核心 API（初始未实现）。"""


class Symmetry:
    def pair(self, verts, axis, tol):
        """配对"""
        raise NotImplementedError

    def residual(self):
        """残差"""
        raise NotImplementedError
