# meshsymmetry

## 题面（可直接粘贴）

子目录：meshsymmetry。网格对称检测与配对。约束：按平面对称（轴可配）寻找顶点配对，容差内视为对称；配对数必须等于镜像命中数（与暴力最近点对照 ≤1e-9）；非对称顶点单独统计；1 万顶点不超过 800 ms；非法轴可区分报错。验收：cd meshsymmetry && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd meshsymmetry && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
