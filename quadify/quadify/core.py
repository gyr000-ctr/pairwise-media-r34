"""quadify 核心 API：三角网格四边形化。

将共享边、绕向一致且法线夹角在阈值 (0, 180] 内的相邻三角形成对合并
为四边形；未配对的三角形原样保留；会产生三点共线等退化四边形的配对
不合并。输出面数不超过输入面数。

错误分类（均为 ValueError 子类，可区分）：
- InvalidThresholdError：阈值非法（非实数、非有限、不在 (0, 180]）。
- NonFiniteInputError：顶点坐标含 NaN/Inf 等非有限值。
- InvalidMeshError：网格结构非法（空输入、非三角形面、索引越界等）。

并发安全：quadify 的全部中间状态均为局部变量，仅在成功返回前一次性
更新实例的 _quads 计数；不存在模块级可变状态。
"""

from __future__ import annotations

import math

__all__ = [
    "Quadify",
    "QuadifyError",
    "InvalidThresholdError",
    "NonFiniteInputError",
    "InvalidMeshError",
]

_MAX_THRESHOLD = 180.0
# 三点共线判定：|sin(夹角)| 小于该值视为共线（退化）。
_COLLINEAR_SIN_EPS = 1e-9


class QuadifyError(ValueError):
    """quadify 输入非法的基类。"""


class InvalidThresholdError(QuadifyError):
    """法线夹角阈值非法（须为 (0, 180] 内的有限实数）。"""


class NonFiniteInputError(QuadifyError):
    """输入包含 NaN 或无穷等非有限数值。"""


class InvalidMeshError(QuadifyError):
    """网格结构非法（空输入、面片非三角形、索引越界等）。"""


def _validate_threshold(angle_thresh):
    if isinstance(angle_thresh, bool) or not isinstance(angle_thresh, (int, float)):
        raise InvalidThresholdError(
            "法线夹角阈值必须是实数，got %r" % (angle_thresh,)
        )
    t = float(angle_thresh)
    if not math.isfinite(t):
        raise InvalidThresholdError("法线夹角阈值必须是有限数值，got %r" % (t,))
    if not 0.0 < t <= _MAX_THRESHOLD:
        raise InvalidThresholdError(
            "法线夹角阈值必须在 (0, 180] 区间内，got %r" % (t,)
        )
    return t


def _validate_verts(verts):
    try:
        raw = list(verts)
    except TypeError:
        raise InvalidMeshError("顶点必须是可迭代的三元坐标序列") from None
    if not raw:
        raise InvalidMeshError("顶点列表为空")
    out = []
    for i, v in enumerate(raw):
        try:
            comps = tuple(v)
        except TypeError:
            raise InvalidMeshError("顶点 %d 不是三元坐标序列" % i) from None
        if len(comps) != 3:
            raise InvalidMeshError(
                "顶点 %d 应有 3 个坐标分量，实际 %d 个" % (i, len(comps))
            )
        pt = []
        for c in comps:
            if isinstance(c, bool) or not isinstance(c, (int, float)):
                raise InvalidMeshError("顶点 %d 含非数值坐标 %r" % (i, c))
            f = float(c)
            if not math.isfinite(f):
                raise NonFiniteInputError("顶点 %d 含非有限坐标 %r" % (i, c))
            pt.append(f)
        out.append((pt[0], pt[1], pt[2]))
    return out


def _validate_faces(faces, nverts):
    try:
        raw = list(faces)
    except TypeError:
        raise InvalidMeshError("面必须是可迭代的三角形索引序列") from None
    if not raw:
        raise InvalidMeshError("面列表为空")
    out = []
    for i, f in enumerate(raw):
        try:
            idx = tuple(f)
        except TypeError:
            raise InvalidMeshError("面 %d 不是索引序列" % i) from None
        if len(idx) != 3:
            raise InvalidMeshError(
                "面 %d 应为三角形（3 个索引），实际 %d 个" % (i, len(idx))
            )
        for j in idx:
            if isinstance(j, bool) or not isinstance(j, int):
                raise InvalidMeshError("面 %d 含非整数索引 %r" % (i, j))
            if j < 0 or j >= nverts:
                raise InvalidMeshError("面 %d 的索引 %d 越界" % (i, j))
        out.append((idx[0], idx[1], idx[2]))
    return out


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _face_normal(pts, face):
    e1 = _sub(pts[face[1]], pts[face[0]])
    e2 = _sub(pts[face[2]], pts[face[0]])
    return _cross(e1, e2)


def _apex(face, u, v):
    """三角形中不在共享边 (u, v) 上的顶点；不存在（面含重复索引）时返回 None。"""
    for x in face:
        if x != u and x != v:
            return x
    return None


def _quad_degenerate(pts, quad):
    """四边形退化判定：顶点重复或任意连续三点（近似）共线。"""
    if len(set(quad)) != 4:
        return True
    p = [pts[i] for i in quad]
    for k in range(4):
        e1 = _sub(p[(k + 1) % 4], p[k])
        e2 = _sub(p[(k + 2) % 4], p[(k + 1) % 4])
        c = _cross(e1, e2)
        # |e1 x e2| <= eps * |e1| * |e2| 视为共线（含零长边）。
        if _norm(c) <= _COLLINEAR_SIN_EPS * _norm(e1) * _norm(e2):
            return True
    return False


class Quadify:
    """三角网格四边形化器。实例可复用；quads() 返回最近一次合并的四边形数。"""

    def __init__(self):
        self._quads = 0

    def quadify(self, verts, faces, angle_thresh):
        """合并相邻三角形为四边形，返回 (顶点列表, 面列表) 元组。

        不修改输入；同一输入总是产生同一输出。
        """
        thresh = _validate_threshold(angle_thresh)
        pts = _validate_verts(verts)
        tris = _validate_faces(faces, len(pts))

        n = len(tris)
        normals = [_face_normal(pts, f) for f in tris]

        # 无向共享边 -> [(面索引, 该面内的有向边)]，按面序插入，确定性。
        edge_map = {}
        for fi, (a, b, c) in enumerate(tris):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                edge_map.setdefault(key, []).append((fi, u, v))

        candidates = []
        for key, inc in edge_map.items():
            if len(inc) != 2:
                continue  # 边界边或非流形边
            f1, u1, v1 = inc[0]
            f2, u2, v2 = inc[1]
            if f1 == f2:
                continue
            if not (u1 == v2 and v1 == u2):
                continue  # 绕向不一致，合并会产生自交四边形
            n1 = normals[f1]
            n2 = normals[f2]
            l1 = _norm(n1)
            l2 = _norm(n2)
            if l1 == 0.0 or l2 == 0.0:
                continue  # 零面积三角形不参与合并
            dot = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (l1 * l2)
            dot = max(-1.0, min(1.0, dot))
            ang = math.degrees(math.acos(dot))
            if not 0.0 < ang <= thresh:
                continue
            p = _apex(tris[f1], u1, v1)
            q = _apex(tris[f2], u1, v1)
            if p is None or q is None or p == q:
                continue
            # 沿并集边界的有向环：p -> u1 -> q -> v1，绕向与 f1 一致。
            quad = (p, u1, q, v1)
            if _quad_degenerate(pts, quad):
                continue
            candidates.append((ang, key, f1, f2, quad))

        # 贪心配对：法线夹角小者优先；排序键唯一，结果确定。
        candidates.sort(key=lambda c: (c[0], c[1]))
        paired = [False] * n
        quad_at = {}
        for _ang, _key, f1, f2, quad in candidates:
            if paired[f1] or paired[f2]:
                continue
            paired[f1] = paired[f2] = True
            quad_at[min(f1, f2)] = quad

        out_faces = []
        for i in range(n):
            if i in quad_at:
                out_faces.append(quad_at[i])
            elif not paired[i]:
                out_faces.append(tris[i])

        self._quads = len(quad_at)
        return (pts, out_faces)

    def quads(self):
        """最近一次 quadify 调用生成的四边形数（未调用过则为 0）。"""
        return self._quads
