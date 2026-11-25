# islands

## 题面（可直接粘贴）

子目录：islands。UV 岛分割。约束：按面邻接的 UV 连续性分割（共享边 UV 一致则同岛）；岛数等于连通分量数（与并查集对照 ≤1e-12）；每岛面积可统计且总和等于总面积；退化 UV 可区分报错；1 万面不超过 700 ms；并发安全。验收：cd islands && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd islands && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
