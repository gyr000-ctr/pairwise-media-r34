"""islands 核心 API：按 UV 连续性分割 UV 岛。

分割规则
--------
两个面若共享一条边，且该边两端的 UV 坐标完全一致（同一个 UV 坐标对），
则二者属于同一个 UV 岛；岛与面邻接图的连通分量一一对应（等价于并查集
在同一组边关系上求得的连通分量）。

面积
----
:meth:`Islands.areas` 返回与面标签等长的逐面 UV 面积（鞋带公式，取绝对值），
按 ``split`` 返回的标签分组求和即可得到每岛面积，全部相加等于总面积。
``split`` 的返回值是 :class:`IslandLabels` —— 一个 ``list`` 子类，元素为普通
``int`` 标签，同时随附对应的面数据，因此 ``areas`` 不依赖任何实例/全局状态，
仅凭入参即可完成计算。

线程安全
--------
整个 API 无实例可变状态、无模块级可变状态：每次调用的并查集、边表等均为
调用局部变量，多线程并发调用同一实例或不同实例都不会相互干扰。
"""

import math

__all__ = [
    "Islands",
    "IslandLabels",
    "IslandsError",
    "NonFiniteUVError",
    "DegenerateUVError",
]


class IslandsError(ValueError):
    """UV 岛分割相关错误的基类（仍是 ``ValueError``）。"""


class NonFiniteUVError(IslandsError):
    """UV 坐标中出现 NaN / Infinity 等非有限值。"""


class DegenerateUVError(IslandsError):
    """面的 UV 面积为 0（退化面）。"""


class IslandLabels(list):
    """面标签列表（元素为普通 ``int``），随附对应面供 :meth:`Islands.areas` 使用。

    行为与普通列表完全一致（``isinstance(x, list)`` 为真、按元素判等），
    额外在 ``faces`` 属性中保存打包后的面索引元组，使面积计算无需任何
    实例缓存即可无状态完成。
    """

    __slots__ = ("faces",)

    def __init__(self, labels, faces):
        super().__init__(labels)
        self.faces = tuple(faces)


def _pack_uvs(uvs):
    """校验 UV 列表并打包为 ``[(x, y), ...]`` 浮点元组列表。"""
    if not uvs:
        raise IslandsError("uvs 不能为空")
    pts = []
    for i, uv in enumerate(uvs):
        try:
            x = float(uv[0])
            y = float(uv[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise IslandsError(f"UV 顶点 {i} 格式非法: {uv!r}") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise NonFiniteUVError(f"UV 顶点 {i} 含非有限值: {uv!r}")
        pts.append((x, y))
    return pts


def _pack_faces(faces, pts):
    """校验面列表并打包为元组列表，同时检查零面积退化面。"""
    if not faces:
        raise IslandsError("faces 不能为空")
    uv_count = len(pts)
    packed = []
    for fi, face in enumerate(faces):
        try:
            m = len(face)
        except TypeError as exc:
            raise IslandsError(f"面 {fi} 格式非法: {face!r}") from exc
        if m < 3:
            raise IslandsError(f"面 {fi} 至少需要 3 个顶点: {face!r}")
        idx = []
        for v in face:
            if not isinstance(v, int) or not (0 <= v < uv_count):
                raise IslandsError(f"面 {fi} 含非法 UV 索引: {v!r}")
            idx.append(v)
        idx = tuple(idx)
        packed.append(idx)

        # 鞋带公式求有向双倍面积，为 0 即退化
        s = 0.0
        for k in range(m):
            ax, ay = pts[idx[k]]
            bx, by = pts[idx[(k + 1) % m]]
            s += ax * by - ay * bx
        if s == 0.0:
            raise DegenerateUVError(f"面 {fi} 的 UV 退化为零面积: {face!r}")
    return packed


def _shoelace_abs(pts, face):
    """单个面的 UV 面积（鞋带公式，取绝对值）。"""
    s = 0.0
    m = len(face)
    for k in range(m):
        ax, ay = pts[face[k]]
        bx, by = pts[face[(k + 1) % m]]
        s += ax * by - ay * bx
    return abs(s) * 0.5


class Islands:
    """UV 岛分割器。无状态，所有方法可被多线程并发调用。"""

    def split(self, uvs, faces):
        """按面邻接的 UV 连续性分割 UV 岛。

        :param uvs:   ``[(u, v), ...]``，每个元素含两个数值分量。
        :param faces: ``[(i, j, k, ...), ...]``，UV 索引组成的面（≥3 点）。
        :return: :class:`IslandLabels` —— 与 ``faces`` 等长的面标签列表，
                 标签为 ``[0, 岛数)`` 内的 Python ``int``，按面首次出现的
                 岛顺序编号，结果确定可复现；列表随附打包后的面数据。
        :raises NonFiniteUVError: UV 含 NaN/Inf。
        :raises DegenerateUVError: 存在零面积退化面。
        :raises IslandsError: 空输入或非法索引/格式（亦为 ``ValueError``）。
        """
        pts = _pack_uvs(uvs)
        packed = _pack_faces(faces, pts)
        nf = len(packed)

        # 并查集：路径减半 + 按大小合并
        parent = list(range(nf))
        size = [1] * nf

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        # 边键 = 规范化后的 UV 坐标对；同一条 UV 边两侧的面并入同一岛。
        edge_owner = {}
        for fi, face in enumerate(packed):
            m = len(face)
            for k in range(m):
                pa = pts[face[k]]
                pb = pts[face[(k + 1) % m]]
                key = (pa, pb) if pa < pb else (pb, pa)
                owner = edge_owner.get(key)
                if owner is None:
                    edge_owner[key] = fi
                else:
                    union(owner, fi)

        root_to_label = {}
        labels = []
        for i in range(nf):
            root = find(i)
            label = root_to_label.get(root)
            if label is None:
                label = len(root_to_label)
                root_to_label[root] = label
            labels.append(label)

        return IslandLabels(labels, packed)

    def areas(self, uvs, islands):
        """逐面 UV 面积（与 ``split`` 返回的标签列表等长、按面对齐）。

        按标签对结果分组求和即得到每岛面积；全部元素之和等于总面积。

        :param uvs:     与 ``split`` 相同的 UV 列表。
        :param islands: ``split`` 返回的 :class:`IslandLabels`（携带面数据）。
        :return: 每个面的 UV 面积（``float``，恒正）。
        :raises IslandsError: 标签非法或与 UV 数据不匹配。
        """
        if not isinstance(islands, IslandLabels):
            raise IslandsError(
                "islands 必须是 Islands.split 返回的标签列表（IslandLabels）"
            )
        faces = islands.faces
        nf = len(faces)
        if len(islands) != nf:
            raise IslandsError("islands 标签数与面数不一致")

        pts = _pack_uvs(uvs)
        uv_count = len(pts)
        max_label = -1
        for v in islands:
            if not isinstance(v, int):
                raise IslandsError(f"岛标签必须为 int: {v!r}")
            if v > max_label:
                max_label = v
        if max_label >= nf or (nf and min(islands) < 0):
            raise IslandsError("岛标签超出合法范围")

        result = []
        for face in faces:
            if any(not (0 <= idx < uv_count) for idx in face):
                raise IslandsError("uvs 与 split 时的面索引不匹配")
            result.append(_shoelace_abs(pts, face))
        return result
