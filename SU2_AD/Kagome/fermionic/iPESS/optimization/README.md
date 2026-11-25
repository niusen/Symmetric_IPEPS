# Kagome Z2 iPESS：Full Update 与变分梯度优化

实现位于 `src/fermionic/kagome/optimization/`，模块名为
`KagomeFermionicOptimization`。沿用已有 kagome 的 `T_u/T_d`、三个 `B`
和检查点格式；三角晶格、旧 CTMRG 及旧优化源文件均未修改。
旧 CTMRG 函数由 kagome 适配器的独立模块加载。

## Full Update 保留的结构

参考 `src/fermionic/simple_update/fermi_triangle_FullUpdate_iPESS.jl` 和
`Full_Update_lib.jl` 的局部三角形更新：

1. 对当前 iPESS 构建 CTMRG 环境，在三角形的三个外侧虚腿上提取完整环境度量。
   外侧环境由 CTMRG 给出，SU 的 lambda 不参与这个度量。
2. 三站点虚时门作用于 `T × B_a × B_b × B_c`。
3. 用直接 HOSVD 和环境度量边缘矩阵产生的 gauge/HOSVD 分别初始化。
4. 固定其余三个因子，依次优化 `B_a → B_b → B_c → T`，用带截断和微小
   正则项的 SVD 求解局部线性最小二乘。保留 CTM 加权保真度较高的一组解。
5. 写回的仍是四个三阶张量。下三角形通过反转两个 Bravais 坐标和带费米子
   符号的虚腿交换使用同一更新；每次局部更新都重新计算 CTMRG。

这是对旧三角晶格 FU 思路的 kagome 适配，不是逐行移植。kagome 的 B 原本
就是三阶张量，所以直接优化三个完整 B 和中心 simplex；没有照搬旧实现中
更高阶站点张量的 residual/keep 拆分。环境度量先 Hermitian 化，再投影到
半正定部分，并返回原始非 Hermitian 误差、负谱权重和 CTMRG 收敛信息。
还用独立的完整混合重叠收缩核验度量提取，失败即报错。

方格 iPEPS 仅用于环境收缩和测量。优化变量一直是 iPESS 的三角形因子。
演化按 `dn(dt/2) → up(dt) → dn(dt/2)` 顺序进行。

这版使用显式环境矩阵和局部设计矩阵，优先保证结构、符号及可检查性。
若三个外腿均为 D，环境矩阵规模为 `D³ × D³`。利用物理恒等因子，将同宇称
的不同列装入不同物理基矢，一次反向收缩最多提取 64 列；D=4 通常只需 1–2 次。
局部 ALS 另需保存设计矩阵。大 D 的计算成本尚未优化。

## 自动微分变分优化

参考 `src/fermionic/triangle_iPESS_Hubbard_AD_cell.jl` 的能量链路：

```text
T_u, T_d, B_a, B_b, B_c 的 Z2 块参数
    → 费米子 iPESS 阻塞
    → 固定 N 次完整 CTMRG sweep
    → Hubbard 每站点能量
    → Zygote 梯度
    → L-BFGS / 最速下降 + Armijo 线搜索
```

`memory > 0` 为 L-BFGS，`memory = 0` 为最速下降。内积取复向量内积的实部。
所有元胞张量独立优化；优化期间固定各 Z2 扇区维数。
AD 包含 CTMRG 环境响应，是有限次迭代展开的导数，不是隐式固定点求导。
每次目标函数计算都从同样规则的 PBC 环境初始化，执行恰好 N 次 sweep，
避免基于误差的提前退出使线搜索和差分检查采用不同迭代次数。
`ctm.tol` 用于最后的独立测量，不控制 AD 的迭代次数。

`check_gradient` 比较随机复方向的中心差分与 `real(dot(g,direction))`。
正式计算须增加 chi 和 N 检查能量、梯度的收敛；截断附近的谱简并也会影响
导数稳定性。默认配置是起始参数，不代表已验证的基态收敛设置。

## 运行与续算

在 `SU2_AD` 目录中，使用原 Julia 环境运行：

```sh
julia --startup-file=no Kagome/fermionic/iPESS/optimization/FullUpdate_iPESS_Z2_ob.jl
julia --startup-file=no Kagome/fermionic/iPESS/optimization/Gradient_iPESS_Z2_ob.jl
```

分别读取同目录的 `full_update.toml`、`variational.toml`。
也可在命令末尾指定配置路径，例如：

```sh
julia --startup-file=no Kagome/fermionic/iPESS/optimization/FullUpdate_iPESS_Z2_ob.jl Kagome/fermionic/iPESS/optimization/smoke_full_update.toml
julia --startup-file=no Kagome/fermionic/iPESS/optimization/Gradient_iPESS_Z2_ob.jl Kagome/fermionic/iPESS/optimization/smoke_variational.toml
```

smoke 配置只核对流程，环境迭代次数很少，不能用于物理结论。
首次编译 CTMRG 反向传播可能需要数分钟。

- `run.resume`：SU、FU 或 AD 的 kagome 检查点；路径相对配置文件。
  续算恢复张量，L-BFGS 历史会重新开始。
- `run.simple_update_tau`：可选 SU 预热；从 FU/AD 检查点继续时通常设为 0。
- `run.D_max`：SU/FU 截断上限；AD 使用加载状态的现有扇区维数。
- FU 的 `schedule`：额外虚时演化阶段，`tau` 必须为 `dt` 的整数倍。
- `optimization.n_sweep/use_gauge/solve_rtol/ridge`：FU 的 ALS 和线性求解参数。
- `optimization.maxiter/memory/step_size/grad_tol/check_gradient`：AD 参数。
- `ctm.chi/iterations`：环境维数及 FU 最大迭代次数 / AD 固定迭代次数。
- `ctm.projector_tol`：FU/AD 环境投影中逆平方根的相对奇异值截断，默认 `1e-8`。
  接近低秩的态可能需提高到 `1e-6`；应同时检查方向导数和能量对阈值的敏感性。
- `ctm.measure/measurement_iterations`：最后是否独立测量及测量迭代上限。

输出为 JLD2 检查点，包含模型参数、配置、优化历史和终止状态；最后测量另存
`observables.jld2`，记录是否收敛。FU 每个演化阶段保存一次，AD 每个接受步保存
一次。FU 写回时将已更新的内部 lambda 设为单位阵；AD 输出的全部 lambda
均为单位阵。这些只是兼容保存格式，不是优化后态的 Schmidt 谱，因此应使用
CTMRG 测量该态，不能把此时的 `su_observables` 当作可靠的环境估计。

CTMRG 沿用模块级元胞/chi 设置，不支持多个优化任务在同一进程中并发调用。
元胞尺寸受旧 CTMRG 的 tuple/rotation 实现支持范围限制。

## API 与检查

算法测试目前包含 84 项检查（含完整三聚体环路符号），入口/续算另有 14 项检查。
`D=2, chi=2` 的随机态方向导数检查相对误差约为 `3.6e-11`；这只验证求导
与实现的有限迭代目标一致，不代表 CTMRG 或基态已收敛。

按顺序加载 `KagomeFermionicZ2.jl`、`ctmrg.jl` 和
`optimization/KagomeFermionicOptimization.jl` 后：

```julia
using .KagomeFermionicOptimization
history = full_update!(state, parameters, 0.1, 0.02, 2;
                       chi=8, ctm_iterations=30, n_sweep=4)
result = optimize_variational(state, parameters;
                              chi=8, ctm_iterations=8, maxiter=10)
optimized_state = result.state
```

`full_update!` 原地更新；`optimize_variational` 返回新状态。
测试命令：

```sh
julia --startup-file=no Kagome/fermionic/iPESS/tests/test_optimization.jl
julia --startup-file=no Kagome/fermionic/iPESS/tests/test_optimization_runner.jl
```

覆盖上下三角形几何、复跃迁精确二聚体、CTM 度量提取、半正定处理、四因子
ALS、无截断门重构、恒等更新、二阶演化顺序、AD 有限差分及能量下降。
另直接比较完整 AD 与冻结环境梯度，验证环境响应确实参与求导；双元胞测试
比较独立 CTMRG 测量和优化目标能量。入口测试覆盖保存、续算及模型参数检查。
