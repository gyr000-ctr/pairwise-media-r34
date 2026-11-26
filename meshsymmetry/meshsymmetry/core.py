"""网格平面对称检测与顶点配对（最小二乘对称平面拟合）。

流程：

1. 输入校验（轴 / 容差 / 顶点有限性），非法输入分类型报错。
2. 以指定轴为法线、最小二乘估计偏移，得到初始对称平面，做镜像最近点配对。
3. 用配对中点（含落在平面上的自镜像点）做最小二乘平面拟合（法线 + 偏移），
   并校验拟合法线与指定轴夹角 ≤ 1°。
4. 用精修平面重新配对；非对称顶点计入 unpaired，数量守恒：
   ``2 * len(pairs) + len(unpaired) == len(verts)``。
5. 残差为精修平面下配对点的 RMS 距离。

仅依赖 Python 标准库。最近邻查询用带 AABB 剪枝的 KD-tree，
平局规则与暴力 argmin 完全一致（等距时取最小索引），
因此配对结果与暴力最近镜像点逐项相同。
"""

import heapq
import math
import threading

__all__ = [
    "Symmetry",
    "InvalidAxisError",
    "InvalidToleranceError",
    "InvalidVerticesError",
    "AxisMismatchError",
]

_AXES = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}

_ANGLE_LIMIT_DEG = 1.0
_EPS = 1e-12
# 平面精修最多迭代次数（中点拟合 -> 重配对）；通常 2 次内收敛
_MAX_REFINE_ITERATIONS = 4


class InvalidAxisError(ValueError):
    """非法对称轴（未知标识 / 零向量 / 维度错误 / 非有限）。"""


class InvalidToleranceError(ValueError):
    """非法配对容差。"""


class InvalidVerticesError(ValueError):
    """非法顶点输入（空集合 / 维度错误 / 非有限数值）。"""


class AxisMismatchError(ValueError):
    """拟合平面法线与指定轴夹角超过 1°。"""


class _KDTree:
    """三维精确最近邻 KD-tree（扁平数组 + 节点 AABB 包围盒剪枝）。

    每个节点保存其子树的轴对齐包围盒；查询时用查询点到包围盒的
    最小平方距离剪枝。平局时返回最小索引，与暴力 argmin 规则完全一致
    （等距子树两侧均探查）。
    """

    __slots__ = (
        "pts",
        "node_axis",
        "node_point",
        "node_left",
        "node_right",
        "node_minidx",
        "box_min",
        "box_max",
        "_capacity",
    )

    def __init__(self, pts):
        self.pts = pts
        n = len(pts)
        cap = max(1, 2 * n)
        self.node_axis = [0] * cap
        self.node_point = [0] * cap
        self.node_left = [-1] * cap
        self.node_right = [-1] * cap
        self.node_minidx = [0] * cap
        self.box_min = [None] * cap
        self.box_max = [None] * cap
        self._capacity = cap
        if n:
            self._build(0, list(range(n)), 0)

    def _ensure(self, slot):
        if slot >= self._capacity:
            grow = slot + 1 - self._capacity
            self.node_axis.extend([0] * grow)
            self.node_point.extend([0] * grow)
            self.node_left.extend([-1] * grow)
            self.node_right.extend([-1] * grow)
            self.node_minidx.extend([0] * grow)
            self.box_min.extend([None] * grow)
            self.box_max.extend([None] * grow)
            self._capacity = slot + 1

    def _build(self, slot, ids, depth):
        axis = depth % 3
        pts = self.pts
        ids.sort(key=lambda i, a=axis: pts[i][a])
        m = len(ids) // 2
        idx = ids[m]
        self.node_axis[slot] = axis
        self.node_point[slot] = idx
        left_ids, right_ids = ids[:m], ids[m + 1:]
        if left_ids:
            child = slot * 2 + 1
            self._ensure(child)
            self.node_left[slot] = child
            self._build(child, left_ids, depth + 1)
        if right_ids:
            child = slot * 2 + 2
            self._ensure(child)
            self.node_right[slot] = child
            self._build(child, right_ids, depth + 1)

        # 子树最小点索引（堆平局时的确定性次级键）
        minidx = idx
        for child in (self.node_left[slot], self.node_right[slot]):
            if child >= 0 and self.node_minidx[child] < minidx:
                minidx = self.node_minidx[child]
        self.node_minidx[slot] = minidx

        # 子树包围盒 = 本节点 ∪ 左右子树包围盒
        p = pts[idx]
        bx0, bx1 = p[0], p[0]
        by0, by1 = p[1], p[1]
        bz0, bz1 = p[2], p[2]
        for child in (self.node_left[slot], self.node_right[slot]):
            if child < 0:
                continue
            cmin, cmax = self.box_min[child], self.box_max[child]
            if cmin[0] < bx0:
                bx0 = cmin[0]
            if cmax[0] > bx1:
                bx1 = cmax[0]
            if cmin[1] < by0:
                by0 = cmin[1]
            if cmax[1] > by1:
                by1 = cmax[1]
            if cmin[2] < bz0:
                bz0 = cmin[2]
            if cmax[2] > bz1:
                bz1 = cmax[2]
        self.box_min[slot] = (bx0, by0, bz0)
        self.box_max[slot] = (bx1, by1, bz1)

    def nearest(self, q):
        """返回 (最近点索引, 平方距离)。best-first：下界最小的子树优先。"""
        qx, qy, qz = q
        pts = self.pts
        points = self.node_point
        lefts = self.node_left
        rights = self.node_right
        minidxs = self.node_minidx
        box_mins = self.box_min
        box_maxs = self.box_max
        best_d2 = math.inf
        best_i = -1
        # 堆元素：(子树下界平方距离, 子树最小点索引, 节点 slot)。
        # 等下界时最小索引子树先出，配合点级 (d2, idx) 比较，
        # 保证与暴力 argmin 的平局规则完全一致。
        heap = [(0.0, 0, 0)]
        heappop = heapq.heappop
        heappush = heapq.heappush
        while heap:
            lb, _, slot = heappop(heap)
            # 浮点 1 ULP 误差下，包围盒下界可能略大于数值相等的点距离，
            # 用容差比较保证等距子树不被漏扫（平局取最小索引的前提）
            if lb > best_d2 and lb - best_d2 > 1e-12 * max(1.0, abs(lb)):
                break
            if slot < 0:
                continue
            idx = points[slot]
            p = pts[idx]
            d2 = (
                (qx - p[0]) * (qx - p[0])
                + (qy - p[1]) * (qy - p[1])
                + (qz - p[2]) * (qz - p[2])
            )
            if d2 < best_d2 or (d2 == best_d2 and idx < best_i):
                best_d2 = d2
                best_i = idx

            for child in (lefts[slot], rights[slot]):
                if child < 0:
                    continue
                bmin, bmax = box_mins[child], box_maxs[child]
                clb = 0.0
                if qx < bmin[0]:
                    clb += (bmin[0] - qx) ** 2
                elif qx > bmax[0]:
                    clb += (qx - bmax[0]) ** 2
                if qy < bmin[1]:
                    clb += (bmin[1] - qy) ** 2
                elif qy > bmax[1]:
                    clb += (qy - bmax[1]) ** 2
                if qz < bmin[2]:
                    clb += (bmin[2] - qz) ** 2
                elif qz > bmax[2]:
                    clb += (qz - bmax[2]) ** 2
                # 剪枝规则与平局规则精确一致：
                #   clb < best_d2            -> 可能严格更近，必须探查；
                #   clb ≈ best_d2 且子树最小索引 < best_i
                #                            -> 只可能平局，但能以更小索引胜出；
                #   其余（clb == best_d2 且最小索引更大，或 clb 更大）
                #                            -> 既不能更近也不能平局取胜，剪掉。
                # 后者让大量重合点的退化点云保持 O(log n) 级查询；
                # 前者保留相对容差，避免浮点舍入漏掉等距子树。
                if clb < best_d2:
                    heappush(heap, (clb, minidxs[child], child))
                elif (
                    clb - best_d2 <= 1e-12 * max(1.0, abs(clb))
                    and minidxs[child] < best_i
                ):
                    heappush(heap, (clb, minidxs[child], child))
        return best_i, best_d2


def _jacobi_eigen3(a):
    """对 3x3 对称矩阵做循环 Jacobi 特征分解，返回 (特征值, 对应特征向量)。"""
    m = [row[:] for row in a]
    v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def rotate(p, q):
        if abs(m[p][q]) <= 1e-300:
            return
        app, aqq, apq = m[p][p], m[q][q], m[p][q]
        tau = (aqq - app) / (2.0 * apq)
        t = (1.0 if tau >= 0.0 else -1.0) / (
            abs(tau) + math.sqrt(1.0 + tau * tau)
        )
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        m[p][p] = app - t * apq
        m[q][q] = aqq + t * apq
        m[p][q] = m[q][p] = 0.0
        r = 3 - p - q
        arp, arq = m[r][p], m[r][q]
        m[r][p] = m[p][r] = c * arp - s * arq
        m[r][q] = m[q][r] = s * arp + c * arq
        for k in range(3):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq

    for _ in range(100):
        if abs(m[0][1]) + abs(m[0][2]) + abs(m[1][2]) <= 1e-300:
            break
        rotate(0, 1)
        rotate(0, 2)
        rotate(1, 2)

    eigvals = [m[0][0], m[1][1], m[2][2]]
    # V 的第 k 列是对应 m[k][k] 的特征向量
    eigvecs = [(v[0][k], v[1][k], v[2][k]) for k in range(3)]
    return eigvals, eigvecs


def _fit_plane(points, axis):
    """对点集做最小二乘平面拟合，返回 (单位法线, 偏移 d, RMS 残差)。

    平面方程 n·x + d = 0；法线符号与 axis 对齐。退化（共线/重合）时
    在最小特征方向中选取与 axis 最接近者。
    """
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    cz = sum(p[2] for p in points) / n

    sxx = syy = szz = sxy = sxz = syz = 0.0
    for x, y, z in points:
        dx, dy, dz = x - cx, y - cy, z - cz
        sxx += dx * dx
        syy += dy * dy
        szz += dz * dz
        sxy += dx * dy
        sxz += dx * dz
        syz += dy * dz

    cov = [
        [sxx / n, sxy / n, sxz / n],
        [sxy / n, syy / n, syz / n],
        [sxz / n, syz / n, szz / n],
    ]
    eigvals, eigvecs = _jacobi_eigen3(cov)
    scale = max(abs(v) for v in eigvals)
    if scale <= _EPS:
        # 所有点重合：法线不可辨识，直接采用指定轴
        normal = axis
    else:
        # 在数值上接近最小特征值的方向里选与轴夹角最小的，处理共线退化
        ordered = sorted(range(3), key=lambda k: eigvals[k])
        smallest = eigvals[ordered[0]]
        candidates = [
            k for k in ordered
            if eigvals[k] - smallest <= _EPS * scale
        ]
        best_k = max(candidates, key=lambda k: abs(
            eigvecs[k][0] * axis[0]
            + eigvecs[k][1] * axis[1]
            + eigvecs[k][2] * axis[2]
        ))
        nv = eigvecs[best_k]
        norm = math.sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2])
        normal = (nv[0] / norm, nv[1] / norm, nv[2] / norm)
        if (
            normal[0] * axis[0]
            + normal[1] * axis[1]
            + normal[2] * axis[2]
        ) < 0.0:
            normal = (-normal[0], -normal[1], -normal[2])

    d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz)
    rms = math.sqrt(
        sum(
            (normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] + d) ** 2
            for p in points
        )
        / n
    )
    return normal, d, rms


def _plane_close(n1, d1, n2, d2):
    """两套 (单位法线, 偏移) 是否在数值上视为同一平面。"""
    dn = (
        (n1[0] - n2[0]) ** 2
        + (n1[1] - n2[1]) ** 2
        + (n1[2] - n2[2]) ** 2
    )
    return dn <= _EPS and abs(d1 - d2) <= _EPS


class Symmetry:
    """平面对称检测与顶点配对器。

    拟合结果保存在类级别共享状态中（测试约定：新实例可读取最近一次
    :meth:`pair` 的结果）；类级锁串行化所有配对调用，实例方法仅做
    只读访问，因此并发安全。单次 :meth:`pair` 的返回值在锁内计算，
    调用方各自拿到与自身输入对应的结果。
    """

    _state_lock = threading.RLock()
    _pairs = []
    _unpaired = []
    _residual = 0.0
    normal = None
    offset = None

    # ------------------------------------------------------------------
    # 输入校验
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_axis(axis):
        if isinstance(axis, str):
            vec = _AXES.get(axis.lower())
            if vec is None:
                raise InvalidAxisError(f"unknown axis label: {axis!r}")
            return vec
        if isinstance(axis, (tuple, list)) and len(axis) == 3:
            try:
                vals = [float(c) for c in axis]
            except (TypeError, ValueError):
                raise InvalidAxisError(f"invalid axis vector: {axis!r}")
            if not all(math.isfinite(v) for v in vals):
                raise InvalidAxisError(f"non-finite axis vector: {axis!r}")
            norm = math.sqrt(sum(v * v for v in vals))
            if norm == 0.0:
                raise InvalidAxisError("axis vector must be non-zero")
            return (vals[0] / norm, vals[1] / norm, vals[2] / norm)
        # 其他序列（numpy 数组等）只要恰好 3 个可转 float 的元素也接受
        if not isinstance(axis, (str, bytes)):
            try:
                seq = list(axis)
            except TypeError:
                seq = None
            if seq is not None and len(seq) == 3:
                try:
                    vals = [float(c) for c in seq]
                except (TypeError, ValueError):
                    raise InvalidAxisError(
                        f"invalid axis vector: {axis!r}"
                    )
                if not all(math.isfinite(v) for v in vals):
                    raise InvalidAxisError(
                        f"non-finite axis vector: {axis!r}"
                    )
                norm = math.sqrt(sum(v * v for v in vals))
                if norm == 0.0:
                    raise InvalidAxisError("axis vector must be non-zero")
                return (vals[0] / norm, vals[1] / norm, vals[2] / norm)
        raise InvalidAxisError(f"invalid axis: {axis!r}")

    @staticmethod
    def _parse_verts(verts):
        try:
            raw = list(verts)
        except TypeError:
            raise InvalidVerticesError("verts must be an iterable of vertices")
        if not raw:
            raise InvalidVerticesError("verts must be non-empty")
        pts = []
        for i, v in enumerate(raw):
            if isinstance(v, (str, bytes)):
                raise InvalidVerticesError(
                    f"vertex {i} must be a 3-coordinate sequence"
                )
            try:
                if len(v) != 3:
                    raise InvalidVerticesError(
                        f"vertex {i} must have exactly 3 coordinates"
                    )
            except TypeError:
                raise InvalidVerticesError(
                    f"vertex {i} must have exactly 3 coordinates"
                )
            try:
                coords = (float(v[0]), float(v[1]), float(v[2]))
            except (TypeError, ValueError, IndexError, KeyError):
                raise InvalidVerticesError(
                    f"vertex {i} has non-numeric coordinates"
                )
            if not all(math.isfinite(c) for c in coords):
                raise InvalidVerticesError(
                    f"vertex {i} contains non-finite value"
                )
            pts.append(coords)
        return pts

    @staticmethod
    def _parse_tol(tol):
        if isinstance(tol, (bool, str, bytes)):
            raise InvalidToleranceError("tol must be a real number")
        try:
            as_float = float(tol)
        except (TypeError, ValueError):
            raise InvalidToleranceError("tol must be a real number")
        if not math.isfinite(as_float) or as_float <= 0.0:
            raise InvalidToleranceError(
                "tol must be a finite positive number"
            )
        return as_float

    # ------------------------------------------------------------------
    # 核心流程
    # ------------------------------------------------------------------
    @staticmethod
    def _mirror_all(pts, nx, ny, nz, d):
        return [
            (
                x - 2.0 * (nx * x + ny * y + nz * z + d) * nx,
                y - 2.0 * (nx * x + ny * y + nz * z + d) * ny,
                z - 2.0 * (nx * x + ny * y + nz * z + d) * nz,
            )
            for (x, y, z) in pts
        ]

    @staticmethod
    def _edges_for_plane(pts, tree, normal, d, tol2):
        """对给定平面生成镜像点并查询最近邻，返回 (边列表, 自镜像索引)。"""
        nx, ny, nz = normal
        mirrors = Symmetry._mirror_all(pts, nx, ny, nz, d)
        edges = []
        self_mirror = []
        for i, q in enumerate(mirrors):
            j, d2 = tree.nearest(q)
            if j < 0 or d2 > tol2:
                continue
            if j == i:
                self_mirror.append(i)
            else:
                edges.append((d2, i, j))
        return edges, self_mirror

    @staticmethod
    def _assemble(pts, n, edges, self_mirror):
        """把候选边确定性贪心匹配为配对，并构造中点与未配对索引。"""
        edges.sort(key=lambda e: (e[0], e[1], e[2]))
        used = [False] * n
        pairs = []
        for d2, i, j in edges:
            if used[i] or used[j]:
                continue
            used[i] = used[j] = True
            pairs.append((i, j) if i < j else (j, i))

        midpoints = []
        for i, j in pairs:
            pi, pj = pts[i], pts[j]
            midpoints.append(
                ((pi[0] + pj[0]) * 0.5, (pi[1] + pj[1]) * 0.5,
                 (pi[2] + pj[2]) * 0.5)
            )
        for i in self_mirror:
            if not used[i]:
                midpoints.append(pts[i])

        unpaired = [i for i in range(n) if not used[i]]
        pairs.sort()
        return pairs, unpaired, midpoints

    def pair(self, verts, axis, tol):
        """检测平面对称并返回顶点索引配对列表 ``[(i, j), ...]``。"""
        cls = type(self)
        with cls._state_lock:
            axis_vec = self._parse_axis(axis)
            pts = self._parse_verts(verts)
            tol = self._parse_tol(tol)
            n = len(pts)

            # 初始平面：法线固定为指定轴，偏移由最小二乘 d = -n·centroid
            cx = sum(p[0] for p in pts) / n
            cy = sum(p[1] for p in pts) / n
            cz = sum(p[2] for p in pts) / n
            d0 = -(
                axis_vec[0] * cx + axis_vec[1] * cy + axis_vec[2] * cz
            )

            # 点云在整个流程中不变，KD-tree 只建一次，各次精修复用
            tree = _KDTree(pts)
            tol2 = tol * tol
            normal, offset = axis_vec, d0
            edges, self_mirror = self._edges_for_plane(
                pts, tree, normal, offset, tol2
            )
            pairs, unpaired, midpoints = self._assemble(
                pts, n, edges, self_mirror
            )

            if midpoints:
                # 迭代精修，不变式：``(pairs, unpaired)`` 始终对应
                # ``(normal, offset)`` 这套平面。每一步用中点做最小二乘
                # 拟合（法线 + 偏移），校验与指定轴夹角 ≤ 1°，再用新的
                # 平面重新配对；平面稳定后配对也不会再变，提前结束。
                for _ in range(_MAX_REFINE_ITERATIONS):
                    fitted, fitted_d, _ = _fit_plane(midpoints, axis_vec)
                    cosang = abs(
                        fitted[0] * axis_vec[0]
                        + fitted[1] * axis_vec[1]
                        + fitted[2] * axis_vec[2]
                    )
                    angle = math.degrees(
                        math.acos(min(1.0, cosang))
                    )
                    if angle > _ANGLE_LIMIT_DEG:
                        raise AxisMismatchError(
                            f"fitted plane normal deviates {angle:.3f}° "
                            f"from axis {axis!r}, limit is "
                            f"{_ANGLE_LIMIT_DEG}°"
                        )
                    plane_stable = _plane_close(
                        fitted, fitted_d, normal, offset
                    )
                    normal, offset = fitted, fitted_d
                    # 用精修平面重新配对，使报告结果与对报告平面做
                    # 暴力最近镜像点完全一致（含等距平局规则）
                    edges, self_mirror = self._edges_for_plane(
                        pts, tree, normal, offset, tol2
                    )
                    pairs, unpaired, midpoints = self._assemble(
                        pts, n, edges, self_mirror
                    )
                    if not midpoints:
                        # 精修后不再有点可拟合：视为无对称信息
                        unpaired = list(range(n))
                        break
                    if plane_stable:
                        break
            else:
                # 没有任何可用对称信息：退回到轴约束平面
                normal, offset = axis_vec, d0
                unpaired = list(range(n))

            # 残差：最终配对点相对最终平面的最小二乘 RMS 距离
            if midpoints:
                residual = _fit_plane(midpoints, normal)[2]
            else:
                residual = 0.0

            cls.normal = normal
            cls.offset = offset
            cls._pairs = pairs
            cls._unpaired = unpaired
            cls._residual = float(residual)
            return list(pairs)

    def residual(self):
        """返回最近一次配对的对称平面最小二乘 RMS 残差（float）。"""
        return float(type(self)._residual)

    def unpaired(self):
        """返回最近一次配对中未配对（非对称 / 自镜像）顶点索引列表。"""
        return list(type(self)._unpaired)
