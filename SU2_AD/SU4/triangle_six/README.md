# 三角晶格 SU(4) 六维表示：每格点两粒子

模型对应 [SciPost Phys. 8, 076 (2020)](https://doi.org/10.21468/SciPostPhys.8.5.076)
第 2 节，参见 [arXiv:1911.03492](https://arxiv.org/abs/1911.03492) 的公式 (2)–(12)。
只有最近邻 J 项；不包含 Hund、三体或 ring exchange 项。

局部空间是 SU(4) 的反对称二粒子表示 `6`，不是 SU(6) 的基本表示。
dense TensorMap 采用 occupation 基 `|12>, |13>, |14>, |23>, |24>, |34>`。
对称 TensorMap 采用 `Rep[SU{4}]((1,1,0,0)=>1)` 的 Gelfand–Tsetlin 基；
代码显式构造与 occupation 基的酉变换，并检查两种后端的算符一致。

采用论文的能量约定：

```text
C_ab = c†_a c_b - δ_ab I6/2
h_ij = J sum_ab C_ab(i) C_ba(j) = J (swap_6 - 6 P_singlet)
6 ⊗ 6 = 1 ⊕ 15 ⊕ 20'
各通道 h/J：-5、-1、+1
```

没有额外常数平移。代码中的 `generator_dot = <h/J>/2` 对应从基本表示
`tr(t_A t_B)=δ_AB/2` 诱导到六维表示的生成元，`C2(6)=5/2`。
旧四维模型的 `<Pij>-1/4` 不适用于此模型。

## 服务器运行

保留完整项目的 `src/` 与本目录，相对路径不变，使用原 Julia 环境。
需要 TensorKit、SUNRepresentations 及旧 CTMRG 的现有依赖。
建议更新完整仓库。本次共享输出文件更新了
`src/bosonic/triangle/su4_dense/Observables_iPESS.jl`（六维标签和通道概率输出）；
计时接口依赖已有更新版 `src/bosonic/CTMRG_unitcell.jl`。手动上传时不要只复制入口文件。

```sh
# dense：默认 Dmax=4
sh SU4/triangle_six/job_su4_six.sh dense

# 使用 SU(4) 分块：默认 Dmax=7
sh SU4/triangle_six/job_su4_six.sh su4
```

直接调用 Julia，不使用 nohup。普通终端自动后台启动；SLURM/PBS/LSF 作业内前台运行。
脚本不申请作业资源；需在原来的作业资源中运行。支持 `JULIA_BIN`、`JULIA_PROJECT`、
`SU4_BLAS_THREADS`、`SU4_OUTPUT_DIR`、`SU4_DMAX`、`SU4_SEED`、`SU4_FOREGROUND=1`。
两种后端有不同结果子目录，不会相互覆盖，也不会读取旧四维模型的 checkpoint。

可在 Julia 中直接配置：

```julia
include("src/bosonic/triangle/su4_six/TriangleSU4Six.jl")
using .TriangleSU4Six
cfg = SixRunConfig(backend=:su4, Dmax=7, seed=1234,
    output_dir="/scratch/your_results/su4_six")
result = run_simulation(cfg)
```

默认是 2×2 cell、J=1，SU 最小时间步 0.001、收敛阈值 `max(1e-8, dt*1e-3)`，
每 10 sweeps 保存 checkpoint。已完成 SU 的 checkpoint 会跳过演化，重算 CTMRG/观测量。
CTMRG 依次使用 chi=16、32、48，每个 chi 最多 50 次，继承旧代码的提前退出规则。
chi 收敛同时检查所有键的 `<h/J>` 和 `<swap_6>`；六维表示中这两者是独立观测量。
结果保存 SU 和 CTMRG 的收敛标记；达到 sweep 上限不等于已收敛。

对称初始虚空间是 `1 ⊕ 6`，总维度 7。独立的 `6` 虚空间不能给三条 `6` 虚腿的
simplex 构成 singlet，不能简单把 D=6 填入对称初始化。
对称 Dmax 表示总物理虚维数预算，截断保留完整不可约表示；实际维数及扇区结构记录到 MAT。
更大初始空间可通过 `SixRunConfig(virtual_space=..., Dmax=...)` 明确指定。
代码不自动将 dense D=4 扫描的整数 D 解释为某个 SU(4) 多重态组合。
`1 ⊕ 6` 只含偶数中心电荷扇区，SU 不会自动生成缺失的奇数电荷扇区；
比较不同虚空间类别时，应显式另选包含 `4`、`4̄` 的空间（例如 `1 ⊕ 4 ⊕ 4̄`，总维数 9）。

**晶胞限制：** 当前沿用旧 CTMRG 支持的 2×2 或 4×4 方形周期胞。
2×2 适合功能验证和有限候选态比较。论文讨论的 12-site VBS 不保证与这些周期胞兼容，
本实现和小胞测试不能用于排除该 VBS；正式研究需依据其平移矢量扩展合适的周期胞。

## 实际复用的数值函数

- SU：`src/bosonic/triangle/simple_update/triangle_SimpleUpdate_iPESS.jl`
  的 `initial_iPESS`、`triangle_update_iPESS`、`triangle_gate_iPESS_simplified`。
  仅六维模型的局部 Hamiltonian/gate 是新代码；gate 指标顺序与旧实现一致。
- CTMRG：`src/bosonic/CTMRG_unitcell.jl` 的 `CTMRG_cell`。
- 观测量：`src/bosonic/triangle/triangle_spin_model_cell.jl` 的
  `hopping_x_no_sign`、`hopping_y_no_sign`、`hopping_diagonala_no_sign`、`ob_onsite`，
  以及它们原来调用的 `build_double_layer_A_B`、`ob_2x2`。
- 键坐标、周期绕行信息、Fourier/空间调制诊断和 CSV 输出复用
  `src/bosonic/triangle/su4_dense/Observables_iPESS.jl`。

测量适配层显式遵循旧 SU 内核的实际连接关系：simplex `B[x,y]` 的 M 腿连接
`T[x,y]`，其余两腿连接 `T[x-1,y].R`、`T[x,y+1].D`。
配对 M 键后沿用旧 iPESS 转换，反射 y 坐标以适配 CTMRG 向下增加的行坐标。
测试直接收缩有限周期 iPESS 网络，验证转换后的波函数完全相同。

键算符做 operator-Schmidt 分解后插入上述旧收缩，
不构造包含完整 d² 物理指标的开放双层张量，不计算无关的三点密度矩阵。
dense 下完整单点密度矩阵只在最后一个 chi 测量；对称后端由 Schur 引理给出 I6/6。
对称截断可能使不同键保留不同扇区。初始环境由旧 `init_CTM_cell` 的标量模板确定边界方向，
再在每条实际虚腿上附加单位 trace 边界；保留每条键原来的空间，不补齐全胞扇区并集。
不把单个交错 dimer 站点强行周期闭合，因为该单站点闭合可能为零，而完整周期胞非零。
之后所有环境迭代均为旧 CTMRG。
对称 CTMRG 的阈值助手避开旧 Dict/eval 依赖，但保留同样的简并截断规则。
上述助手及零奇异值/阈值舍入处理位于新文件 `src/bosonic/triangle/su4_six/CTMRGCompat.jl`，
只在 `TriangleSU4Six` 模块内针对 SU(4) 对角张量分派；原 `CTMRG.jl`、`CTMRG_unitcell.jl` 不改动。
某些 TensorKit 0.17 开发版本在 Julia 1.11 中有 repartition generated-function 定义顺序问题；
模块仅检测到该特定 world-age 异常时，按依赖原始函数体刷新生成器，不改其数学运算。

## 输出

每种后端/D/seed 子目录包含 `checkpoint.jls`、`result.mat`、`run.log` 和
`observables*.csv/.txt/.jls`。主日志在结果根目录的 `logs/`，包括退出码、PID 和可用时的峰值内存。

MAT 保存每个方向、每个 simplex 的：

- `<h/J>`、`<swap_6>`、键能量和每格点/每键能量；2×2 对应全部 12 条独立键。
- `singlet_weight=(<swap_6>-<h/J>)/6`，以及 15、20' 通道概率。
- 生成元点积、连通点积、dense 的 occupation 基单点密度矩阵。
- 带周期绕行信息的键表、方向各向异性、空间调制和 Fourier 分量。
- CTMRG 各 chi 误差/能量/时间，SU 历史、收敛状态、虚空间扇区结构。

单独重测已保存状态：

```sh
julia --startup-file=no SU4/triangle_six/Evaluate_observables.jl /path/checkpoint.jls
```

## 验证

```sh
julia --startup-file=no SU4/triangle_six/tests/runtests.jl
```

测试包含解析两点谱、Casimir/守恒性、dense/SU4 gate 等价、
CTMRG 产品态和水平 singlet-dimer 产品态的精确能量/全键图、旧 SU 内核和断点恢复。
这些是正确性检查，不是充分收敛的基态结果。
