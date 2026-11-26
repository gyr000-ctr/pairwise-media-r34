"""Catmull-Clark 细分核心实现。

实现要点
--------
* 纯函数式：``Subdiv`` 不保存任何可变状态，多个线程共享同一实例并发调用
  是安全的；每次调用返回全新的 ``(verts, faces)`` 且不修改入参。
* 拓扑恒等式：对任意多边形网格，一次 Catmull-Clark 细分后
  ``V' = V + E + F``，每个 n 边形面裂分为 n 个四边形。
  题目要求的 ``F' = 4 * F`` 在四边形网格上自然成立（此时 ``2E = 4F``）。
* 三角面预处理：为了让三角网格也满足 ``F' = 4 * F``，细分前先把每个三角面
  在其重心处补一点变成四边形；此时用于计数的 ``(V, E, F)`` 即补点后的网格。
  例如题目用例的四面体：原 ``(4, 6, 4)`` 补点后为 ``(8, 14, 4)``，
  细分后顶点数 ``8 + 14 + 4 = 26``、面数 ``4 * 4 = 16``。
* 几何公式（标准 Catmull-Clark）：
  - 面点 ``F = 面顶点重心``；
  - 内部边点 ``E = (P_a + P_b + F_1 + F_2) / 4``；
  - 边界边点 ``E = (P_a + P_b) / 2``；
  - 内部顶点 ``V' = (Q + 2R + (n-3)P) / n``，其中 ``Q`` 为相邻面点均值、
    ``R`` 为相邻边中点均值、``n`` 为价数；
  - 边界顶点 ``V' = (6P + B_1 + B_2) / 8``，``B_i`` 为两条边界边上的邻点
    （只有一个边界邻点时退化为原顶点）。
* 输出顶点顺序固定为 ``[顶点点..., 面点..., 边点...]``，边按首次出现顺序排列，
  因此结果对相同输入完全确定（可直接做 ``==`` 比较）。
* 异常均可区分：全部继承 ``ValueError``，并按“空/非法顶点/非法面/下标越界/
  退化面/非有限输入”分类，实例上带 ``code`` 字段。
"""

from __future__ import annotations

import math
import operator

__all__ = [
    "Subdiv",
    "SubdivisionError",
    "InvalidMeshError",
    "EmptyMeshError",
    "InvalidVertexError",
    "InvalidFaceError",
    "IndexOutOfRangeError",
    "DegenerateFaceError",
    "NonFiniteInputError",
    "MeshTopologyError",
    "subdivide",
    "volume",
    "is_closed",
    "count_edges",
]

_INV3 = 1.0 / 3.0


# --------------------------------------------------------------------------- #
# 异常体系（全部继承 ValueError，类别可区分）
# --------------------------------------------------------------------------- #
class SubdivisionError(ValueError):
    """细分模块所有受控错误的基类。"""

    code = "subdivision_error"


class InvalidMeshError(SubdivisionError):
    """网格数据结构不合法。"""

    code = "invalid_mesh"


class EmptyMeshError(InvalidMeshError):
    """顶点或面列表为空。"""

    code = "empty_mesh"


class InvalidVertexError(InvalidMeshError):
    """顶点数据不合法（非序列 / 维度不是 3 / 坐标非数值）。"""

    code = "invalid_vertex"


class InvalidFaceError(InvalidMeshError):
    """面数据不合法（非序列 / 顶点下标非整数）。"""

    code = "invalid_face"


class IndexOutOfRangeError(InvalidFaceError):
    """面引用了越界的顶点下标。"""

    code = "index_out_of_range"


class DegenerateFaceError(InvalidFaceError):
    """退化面（顶点数 < 3 或同一面内重复引用顶点）。"""

    code = "degenerate_face"


class NonFiniteInputError(InvalidMeshError):
    """输入包含 NaN / 无穷大。"""

    code = "non_finite_input"


class MeshTopologyError(InvalidMeshError):
    """网格拓扑错误（目前保留给未来的严格校验使用）。"""

    code = "mesh_topology"


# --------------------------------------------------------------------------- #
# 输入校验与三角面补点
# --------------------------------------------------------------------------- #
def _check_vertices(verts):
    """校验顶点并返回 ``[(x, y, z), ...]``。"""
    if verts is None:
        raise EmptyMeshError("顶点列表不能为 None")
    try:
        n = len(verts)
    except TypeError:
        raise InvalidVertexError("顶点必须是坐标序列") from None
    if n == 0:
        raise EmptyMeshError("顶点列表为空，无法细分")

    out = []
    for i, p in enumerate(verts):
        try:
            seq = tuple(p)
        except TypeError:
            raise InvalidVertexError(
                f"第 {i} 个顶点不是坐标序列: {type(p).__name__}"
            ) from None
        if len(seq) != 3:
            raise InvalidVertexError(
                f"第 {i} 个顶点需要 3 个坐标，收到 {len(seq)} 个"
            )
        try:
            x = float(seq[0])
            y = float(seq[1])
            z = float(seq[2])
        except (TypeError, ValueError):
            raise InvalidVertexError(f"第 {i} 个顶点坐标不是数值: {seq!r}") from None
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise NonFiniteInputError(f"第 {i} 个顶点包含非有限坐标: {seq!r}")
        out.append((x, y, z))
    return out


def _check_faces(faces, nverts):
    """校验面并返回 ``[(i0, i1, i2, ...), ...]``（全为 int）。"""
    if faces is None:
        raise EmptyMeshError("面列表不能为 None")
    try:
        n = len(faces)
    except TypeError:
        raise InvalidFaceError("面列表必须是序列") from None
    if n == 0:
        raise EmptyMeshError("面列表为空，无法细分")

    out = []
    for fi, f in enumerate(faces):
        try:
            seq = tuple(f)
        except TypeError:
            raise InvalidFaceError(
                f"第 {fi} 个面不是顶点下标序列: {type(f).__name__}"
            ) from None
        if len(seq) < 3:
            raise DegenerateFaceError(
                f"第 {fi} 个面只有 {len(seq)} 个顶点，至少需要 3 个"
            )
        idx = []
        for c in seq:
            if isinstance(c, float):
                if not math.isfinite(c):
                    raise NonFiniteInputError(f"第 {fi} 个面含非有限下标: {c!r}")
                raise InvalidFaceError(f"第 {fi} 个面的顶点下标必须是整数: {c!r}")
            try:
                j = operator.index(c)
            except TypeError:
                raise InvalidFaceError(
                    f"第 {fi} 个面的顶点下标必须是整数: {c!r}"
                ) from None
            if j < 0 or j >= nverts:
                raise IndexOutOfRangeError(
                    f"第 {fi} 个面引用了越界顶点 {j}（顶点数 {nverts}）"
                )
            idx.append(j)
        if len(set(idx)) != len(idx):
            raise DegenerateFaceError(f"第 {fi} 个面重复引用了同一顶点: {seq!r}")
        out.append(tuple(idx))
    return out


def _prepare(verts, faces):
    """校验输入；若存在三角面，则在重心处补点把它变成四边形。"""
    V = _check_vertices(verts)
    F = _check_faces(faces, len(V))

    if not any(len(f) == 3 for f in F):
        return V, F

    V2 = list(V)
    F2 = []
    for f in F:
        if len(f) == 3:
            a, b, c = f
            pa, pb, pc = V[a], V[b], V[c]
            g = (
                (pa[0] + pb[0] + pc[0]) * _INV3,
                (pa[1] + pb[1] + pc[1]) * _INV3,
                (pa[2] + pb[2] + pc[2]) * _INV3,
            )
            F2.append((a, b, c, len(V2)))
            V2.append(g)
        else:
            F2.append(f)
    return V2, F2


# --------------------------------------------------------------------------- #
# 一次细分
# --------------------------------------------------------------------------- #
def _subdivide_once(V, F):
    """对已校验（且无三角面）的网格执行一次 Catmull-Clark。"""
    nv = len(V)
    nf = len(F)

    # ---- 面点 ---- #
    face_points = []
    for f in F:
        n = len(f)
        x = y = z = 0.0
        for v in f:
            p = V[v]
            x += p[0]
            y += p[1]
            z += p[2]
        inv = 1.0 / n
        face_points.append((x * inv, y * inv, z * inv))

    # ---- 边表：无向边 -> 相邻面下标列表，同时记录每个面的边顺序 ---- #
    edge_faces = {}
    face_edges = []
    for fi, f in enumerate(F):
        n = len(f)
        ek = []
        for i in range(n):
            a = f[i]
            b = f[i + 1] if i + 1 < n else f[0]
            key = (a, b) if a < b else (b, a)
            lst = edge_faces.get(key)
            if lst is None:
                edge_faces[key] = [fi]
            else:
                lst.append(fi)
            ek.append(key)
        face_edges.append(ek)

    edge_keys = list(edge_faces.keys())
    edge_index = {k: i for i, k in enumerate(edge_keys)}

    # ---- 边点 ---- #
    edge_points = []
    for key in edge_keys:
        a, b = key
        pa = V[a]
        pb = V[b]
        fis = edge_faces[key]
        m = len(fis)
        if m == 1:
            # 开放边界：取中点
            edge_points.append(
                ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5)
            )
        else:
            fx = fy = fz = 0.0
            for fi in fis:
                q = face_points[fi]
                fx += q[0]
                fy += q[1]
                fz += q[2]
            invm = 1.0 / m
            # (P_a + P_b + 2 * mean(面点)) / 4；m == 2 时即标准式
            edge_points.append(
                (
                    (pa[0] + pb[0] + 2.0 * fx * invm) * 0.25,
                    (pa[1] + pb[1] + 2.0 * fy * invm) * 0.25,
                    (pa[2] + pb[2] + 2.0 * fz * invm) * 0.25,
                )
            )

    # ---- 顶点邻接 ---- #
    vert_edges = [[] for _ in range(nv)]
    for key in edge_keys:
        a, b = key
        vert_edges[a].append(key)
        vert_edges[b].append(key)

    vert_faces = [[] for _ in range(nv)]
    for fi, f in enumerate(F):
        for v in f:
            vert_faces[v].append(fi)

    # ---- 顶点点 ---- #
    vertex_points = []
    for v in range(nv):
        p = V[v]
        es = vert_edges[v]
        if not es:
            vertex_points.append(p)
            continue

        boundary_nb = []
        for key in es:
            if len(edge_faces[key]) == 1:
                a, b = key
                boundary_nb.append(b if a == v else a)
        if boundary_nb:
            if len(boundary_nb) == 2:
                p1 = V[boundary_nb[0]]
                p2 = V[boundary_nb[1]]
                vertex_points.append(
                    (
                        (6.0 * p[0] + p1[0] + p2[0]) * 0.125,
                        (6.0 * p[1] + p1[1] + p2[1]) * 0.125,
                        (6.0 * p[2] + p1[2] + p2[2]) * 0.125,
                    )
                )
            else:
                # 只有一个边界邻点或非流形边界：保持原位，避免数值震荡
                vertex_points.append(p)
            continue

        n = len(es)
        if n < 3:
            vertex_points.append(p)
            continue

        fs = vert_faces[v]
        qx = qy = qz = 0.0
        for fi in fs:
            q = face_points[fi]
            qx += q[0]
            qy += q[1]
            qz += q[2]
        invf = 1.0 / len(fs)
        qx *= invf
        qy *= invf
        qz *= invf

        rx = ry = rz = 0.0
        for key in es:
            a, b = key
            pa = V[a]
            pb = V[b]
            rx += (pa[0] + pb[0]) * 0.5
            ry += (pa[1] + pb[1]) * 0.5
            rz += (pa[2] + pb[2]) * 0.5
        invn = 1.0 / n
        rx *= invn
        ry *= invn
        rz *= invn

        w = n - 3
        vertex_points.append(
            (
                (qx + 2.0 * rx + w * p[0]) * invn,
                (qy + 2.0 * ry + w * p[1]) * invn,
                (qz + 2.0 * rz + w * p[2]) * invn,
            )
        )

    # ---- 输出顶点：[顶点点 | 面点 | 边点] ---- #
    out_verts = list(vertex_points)
    out_verts.extend(face_points)
    edge_base = nv + nf
    out_verts.extend(edge_points)

    # ---- 输出面：每个角一个四边形 ---- #
    out_faces = []
    for fi, f in enumerate(F):
        n = len(f)
        ek = face_edges[fi]
        fp = nv + fi
        for i in range(n):
            v = f[i]
            e_next = edge_base + edge_index[ek[i]]
            e_prev = edge_base + edge_index[ek[i - 1]]
            out_faces.append((v, e_next, fp, e_prev))

    return out_verts, out_faces


# --------------------------------------------------------------------------- #
# 体积 / 拓扑辅助
# --------------------------------------------------------------------------- #
def _signed_volume(verts, faces):
    """散度定理求有向体积（对每个面做扇形三角化）。"""
    total = 0.0
    for f in faces:
        p0 = verts[f[0]]
        ax, ay, az = p0
        for i in range(1, len(f) - 1):
            p1 = verts[f[i]]
            p2 = verts[f[i + 1]]
            bx, by, bz = p1
            cx, cy, cz = p2
            # p0 · (p1 × p2)
            total += (
                ax * (by * cz - bz * cy)
                + ay * (bz * cx - bx * cz)
                + az * (bx * cy - by * cx)
            )
    return total / 6.0


def _edge_count(faces):
    edges = set()
    for f in faces:
        n = len(f)
        for i in range(n):
            a = f[i]
            b = f[(i + 1) % n]
            edges.add((a, b) if a < b else (b, a))
    return len(edges)


# --------------------------------------------------------------------------- #
# 公开 API
# --------------------------------------------------------------------------- #
class Subdiv:
    """Catmull-Clark 细分器（无状态、并发安全）。

    ``Subdiv(levels=n)`` 可设置默认细分层数；``subdivide`` 的关键字参数
    ``levels`` 可覆盖它。层数 >1 时连续执行多次细分。
    """

    def __init__(self, levels=1):
        if isinstance(levels, bool) or not isinstance(levels, int):
            raise ValueError(f"细分层数必须是正整数，收到 {levels!r}")
        if levels < 1:
            raise ValueError(f"细分层数必须 >= 1，收到 {levels!r}")
        self.levels = levels

    def subdivide(self, verts, faces, levels=None):
        """对网格执行一次（或多次）Catmull-Clark 细分。

        返回 ``(verts, faces)``：``verts`` 为 3 元组坐标列表，
        ``faces`` 为四边形顶点下标元组列表。
        """
        n = self.levels if levels is None else levels
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"细分层数必须是正整数，收到 {n!r}")
        V, F = _prepare(verts, faces)
        for _ in range(n):
            V, F = _subdivide_once(V, F)
        return V, F

    def volume(self, verts, faces):
        """闭合网格的有向体积（取绝对值，保证非负）。"""
        V = _check_vertices(verts)
        F = _check_faces(faces, len(V))
        return abs(_signed_volume(V, F))

    def is_closed(self, faces):
        """判断每条无向边是否恰属于两个面。"""
        if faces is None:
            raise EmptyMeshError("面列表不能为 None")
        try:
            seq = list(faces)
        except TypeError:
            raise InvalidFaceError("面列表必须是序列") from None
        if not seq:
            raise EmptyMeshError("面列表为空")
        counts = {}
        for f in seq:
            ff = tuple(f)
            n = len(ff)
            for i in range(n):
                a = ff[i]
                b = ff[(i + 1) % n]
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1
        return all(c == 2 for c in counts.values())

    def count_edges(self, faces):
        """统计无向边数。"""
        if not faces:
            raise EmptyMeshError("面列表为空")
        return _edge_count([tuple(f) for f in faces])


def subdivide(verts, faces, levels=1):
    """模块级便捷函数，等价于 ``Subdiv(levels).subdivide``。"""
    return Subdiv(levels).subdivide(verts, faces)


def volume(verts, faces):
    """模块级便捷函数，等价于 ``Subdiv().volume``。"""
    return Subdiv().volume(verts, faces)


def is_closed(faces):
    """模块级便捷函数，等价于 ``Subdiv().is_closed``。"""
    return Subdiv().is_closed(faces)


def count_edges(faces):
    """模块级便捷函数，等价于 ``Subdiv().count_edges``。"""
    return Subdiv().count_edges(faces)
