# Kagome U1 的 large-cell energy variance

## 实现状态补充（2026-11-08）

当前代码先做 right(xmax)、bottom(ymax) 两次预备更新。
论文的水平源示例只先做一次 right；竖直源则旋转整套扫描。
额外的 bottom 更新及 kagome 对角线源扫描是当前实现的选择，
尚未用一般跨格纠缠态的独立远距离关联收缩充分验证。
本次归档保留这一数值流程；现有测试不能证明该选择更好或无害。
完整说明、逐帧动画及离线网页见
[算法说明目录](energy_variance_algorithm/README.md)。

纯 Julia/TensorKit；独立测量模块，复用原来的 Q/R 双层、角/边收缩、CTM
投影器和单列更新。没有修改 SU 或已有能量测量函数。CPU 为默认后端。

算法参考 Cortés Estay、Kamar、Corboz，PRB 113, 245114 (2026)，
https://arxiv.org/html/2511.22669v2 ，II.3。

## 功能与定义

计算交换 Hamiltonian 的有限求和窗口方差密度。对每个不等价最近邻键
分别构造中心 `h0 + source_shift*I` 作用后的 ket；bra 保持原态。中心键
跨三角形时，用精确 U1 算符 Schmidt 分解扩充两条中心虚腿，不截断 ket。
重算混合 Q/R 层，复制收敛的周期 CTM 环境，再从中心向外更新投影器。
使用正的恒等偏移是为了避免中心重叠为零；连通关联中该偏移严格抵消。
应额外比较 source_shift=1、2，检查有限 chi 的敏感性。

对所有中心键求和并除以物理元胞自旋数：sqrt3xsqrt3 有9个 site、18个
中心键。没有使用近似镜面关系来减少中心键数。普通 Q/R 是共享引用，
没有构造整胞 rank-4 AA；只有中心附近的局域混合因子需要改变。
每个位置的混合 CTM 环境独立更新。扫描不把中心插入绕周期边界传播。

`radii` 是以中心原胞原点为参照的原胞坐标 Chebyshev 半径，包含半径内
全部 up/down 三角形的6条键。它不是物理距离或关联长度。每次最大半径
计算同时生成所有较小半径的累计和与逐壳增量。有限窗口的部分和不强制
为正；不得把它截成0来掩盖收缩误差。

在严格固定总 Sz 的 U1 态中，Hex+h*Sz_total 与 Hex 的方差相同。
对于只有平均磁化接近1/9的 dense 态，不能这样推断；本模块始终明确报告
exchange variance，不自动将其称作带磁场 Hamiltonian 的方差。

## 使用

入口：`multicell/examples/run_energy_variance.jl`。所有设置放在该文件中。
运行 `julia run_energy_variance.jl`（按入口实际位置指定路径）。
默认读取已保存的 D4 U1 态，不进行 SU，也不自动运行 D10–15。
输出目录必须是新的，现有数据不会被覆盖。

输出：

- `variance_by_radius.csv`：累计方差密度、虚部、壳增量、逐壳绝对值和及最大关联。
- `connected_terms.csv`：每个中心键和目标键的复连通关联、坐标及半径。
- `variance.json` / `variance.toml`：配置、norm CTM 收敛记录、来源 SHA256 和完整结果。

先固定小 D，比对不同 chi、L、sweeps、source_shift。随后再考虑大 D。
收缩区域应是原显示周期的整数倍，半径与中心的边界距离须足够。
norm CTM 的迭代收敛不代表 LC 方差已经对 chi、区域或扫描次数收敛。
小 chi 下很小的远距离偏差会随面积累积，须看逐壳增量和累计和是否稳定。
还应看逐壳绝对值和及最大关联，避免正负项抵消使累计和看似稳定。

## 实现位置

功能函数放在 `src/bosonic/kagome/native_measurement/energy_variance/`。
大区域的测量环境和逐列投影器使用数组存储（`variance_array_sweeps.jl`）。
没有对方差测量做自动微分，不需要沿用为 AD 设计的嵌套 tuple。
旧 tuple 方法在较大 L 下，填充投影器/临时角张量会反复改变 tuple 类型，
触发大量 Julia 编译；这不是张量乘法的必要成本。
数组路径保持原来的投影器、SVD 截断、归一化及整列同步更新顺序。
`large_cell_sweeps(...; storage=:tuple)` 保留原路径供小 D 对照。
原来的周期 norm CTMRG 与 SU 不变。

日志会报告 norm CTMRG、原态键均值、各中心键混合扫描和目标键测量进度。
结果 JSON/TOML 的 `timing` 包含分阶段耗时；其中可能包含首次 Julia 编译。
它不是峰值内存测量；Julia 的累计分配量也不等于同时占用的内存。

旧模块不需要 include 新模块；只有新入口加载它，避免改变现有计算路径。
新模块只复用现有 TensorKit 版本兼容的 `native_svd` 等函数。
目前是正确性优先的初版；大 D 的逐键重复测量尚需单独评估运行成本。

## 测试

只使用 D1 / D2。不运行生产态优化。
对 Sz 乘积态和含量子纠缠的独立三角形乘积态，逐项与独立的局域8维
矩阵乘法计算的二维连通关联比较，覆盖全部18个中心键和整个半径1区域。
再使用一般复数 D2 张量检查自关联满足
`(S_i.S_j)^2 = 3/16 - (S_i.S_j)/2`。

必须通过这些检验后才将初版用于大 D 结果的外推。零方差也可对应激发
本征态；能量—方差拟合还需要比较同一低能分支及拟合区间稳定性。

2026-11-06 验证（Julia 1.11.6）：TensorKit 0.14.11 和 0.17.0 均通过
2068项断言。每个版本覆盖972个Sz乘积态关联、972个量子三角形乘积态
关联、反向扫描坐标、一般复数D2自关联，以及保存的U1 D4态的18条键、
混合环境和临时checkpoint/CSV/TOML/JSON保存。量子三角形逐项最大误差
2.36e-16，D2有限chi混合CTM自关联最大误差约4.70e-8；U1 D4在未进行
额外LC截断前的局域算符作用误差低于4.2e-16。

这是正确性优先的初版：没有测量D15的方差，也没有完成真实平台态对
chi/L/sweeps的系统收敛检查。中心扩大虚腿及逐键重复测量的大D成本
仍需评估。当前入口做一次完整测量，中断后需重新测量；没有自动重启
或后台监控进程。测试日志和检查数值在tests/energy_variance/validation。


## 服务器批量任务

多 D / 多 chi 的任务见 [ENERGY_VARIANCE_BATCH.md](ENERGY_VARIANCE_BATCH.md)。
根目录 run_energy_variance_server.jl / .sh 已改成批量入口：一次最大区域收缩提取多个 L，不设置独立 R。
本文中 examples/run_energy_variance.jl 的 radius 接口只保留用于历史结果兼容与小规模诊断。
