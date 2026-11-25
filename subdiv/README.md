# subdiv

## 题面（可直接粘贴）

子目录：subdiv。Catmull-Clark 细分。约束：一次细分后面数 = 原面数×4、顶点数 = V+E+F；新顶点为相邻面点与边点加权组合（与参考式 ≤1e-9）；闭合网格体积收敛（两步后变化 ≤1%）；开放边界用边界规则；1 万面不超过 900 ms；非法面可区分报错。验收：cd subdiv && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd subdiv && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
