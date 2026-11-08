# 去掉 bottom 预备操作的动画对照

仅在动画和结构依赖模型中删除初始化的 `bottom y=6`。
Julia 数值算法保留原流程，服务器任务未改变；本目录不是新的数值结果。

- [离线逐步播放器](ctmrg_without_bottom_interactive.html)
- [55 个 move 的 GIF](ctmrg_without_bottom_55.gif)
- [原流程说明与播放器](../README.md)
- [逐步依赖集合对照](schedule_comparison.json)
- [动画核查记录](animation_validation.json)

预备只剩 `right x=7`，然后连续做两轮：

```
left   x=5,6,…,11
right  x=8,7,…,2
top    y=5,6,…,11
bottom y=7,6,…,2
```

总共 55 个 move，比原版少 1 个。默认停在 #2，即第一次 left x=5 写回后。
粉色只表示可能依赖 h0，包括投影器路径；蓝色边框表示本次写入。
没有更新的是“预备 bottom”；每轮末尾的正常 bottom 扫描仍然保留。

## 对照观察

将两种流程按同一轮、同一方向、同一坐标对齐，不按相同 move 编号对齐。

- 第一次 left x=5 后：原版有 28 条环境记录可能依赖 h0，新版有 20 条。
- 第一轮正常 bottom y=6 写回后：两版各有 643 条，且坐标及分量集合完全相同。
- 此后所有对齐的阶段，可能依赖 h0 的环境集合均相同。
- 新版 #16 top y=5 仍会使 C1(4,5) 通过投影器依赖 h0；
  这一依赖不需要最初额外的 bottom 预备操作。

这些观察仅比较布尔依赖集合，不能推出投影器、张量数值或最终关联相同，
也不能证明额外操作有益、无害或必要。判断精度仍需固定态的数值对照。
这里固定演示的是水平源 (6,6)—(7,6)，不包含 kagome 对角线源的独立验证。

compare_without_bottom.py 使用上一层 source/trace_ctm_dependencies.py 的独立
Python 依赖图，对照原 56 步和新 55 步；不运行张量收缩。
source.html 是可编辑播放器片段，完整 HTML 可直接离线打开。

本次离线核查：55 个 move、56 个依赖快照、165 个阶段均与 Python 模型一致；
736/320 像素、浅色/深色布局及播放/暂停/前进/后退通过。
