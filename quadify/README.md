# quadify

## 题面（可直接粘贴）

子目录：quadify。三角网格四边形化。约束：成对合并相邻三角形（共享边且法线夹角阈值可配）；输出面数 ≤ 原面数；未配对三角形保留；退化四边形（三点共线）要避免；1 万面不超过 800 ms；非法阈值可区分报错。验收：cd quadify && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd quadify && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
