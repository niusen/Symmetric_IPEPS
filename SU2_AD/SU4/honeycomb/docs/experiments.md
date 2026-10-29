# 历史实验、物理对照与核查

旧实验脚本已按用途放入 `../experiments/{baseline,initialization,star}/`；其已有输出均移至 `../data/`，日志移至 `../logs/experiments/`。以下保留历史实验记录。

# SU(4) honeycomb：两点 simple update 与能量测量

每个原始格点一个 TensorMap，物理空间为基本表示 `4`，不合并 A/B。
采用 brick-wall：张量顺序为 `(L,D,R,U,physical)`，三条真实虚拟腿和一条
维数为 1 的平凡腿。A/B 两个子晶格的物理表示都是 `4`，不是 `4`/`4⁺` 交替。

## 运行入口

在 `SU2_AD` 下，使用已安装 TensorKit、TensorKitSectors、SUNRepresentations、
ChainRulesCore、Zygote、KrylovKit、JLD2 的 Julia 环境运行：

```sh
julia --startup-file=no SU4/honeycomb/experiments/baseline/SimpleUpdate_ob.jl dense
julia --startup-file=no SU4/honeycomb/experiments/baseline/SimpleUpdate_ob.jl SU4
julia --startup-file=no SU4/honeycomb/tests/runtests.jl
```

入口中的 `main` 负责参数、分阶段演化、JLD2 保存和多个 chi 的 CTMRG 测量；
`include` 此入口不会自动运行计算。示例参数只是启动示例，不能当作收敛结果。

## 复用的旧代码

| 内容 | 旧文件与函数 |
|---|---|
| 两点 SU：外部 lambda、局部 SVD、截断、恢复张量 | `src/bosonic/triangle/simple_update/triangle_SimpleUpdate_iPEPS.jl`：`tebd_xbond`, `tebd_ybond` |
| 两点演化门 | 同上：`trotter_gate` |
| 近零谱的逆 | `src/bosonic/AD_lib.jl`：`my_pinv` |
| TensorKit 版本兼容与设置 | `src/bosonic/Settings.jl`, `Settings_cell.jl` |
| CTMRG 环境 | `src/bosonic/CTMRG.jl`, `CTMRG_unitcell.jl`：`CTMRG_cell` |
| CTMRG 两点算符期望值 | `src/bosonic/triangle/triangle_spin_model_cell.jl`：`hopping_x_no_sign`, `hopping_y_no_sign`, `ob_2x2` |

新增函数在 `src/bosonic/honeycomb/`，由 `Honeycomb_SU4.jl` 加载为独立模块，
沿用旧代码的矩阵 cell、TensorMap 收缩和函数返回值风格。未修改上述旧文件。
模块中补充了实数奇异值的分块逆平方根方法，以避开旧 CTMRG 对字典转换的依赖。
哈密顿量直接对恒等 TensorMap 交换输出腿构造 `P_ij`；测试与旧 `SUN_spin(4)` 对照。

## Dense 与 SU(4) 虚拟量子数

```julia
include("src/bosonic/honeycomb/Honeycomb_SU4.jl")
using .HoneycombSU4

T_set,lambdax_set,lambday_set=initial_honeycomb_iPEPS(2,2;D=2,seed=1234)

V=honeycomb_virtual_space(["1"=>1,"4"=>1,"6"=>1,"4⁺"=>1])
T_set,lambdax_set,lambday_set=initial_honeycomb_iPEPS(2,2;
    symmetry=:SU4,V=V,seed=1234)
```

`=>` 右边是简并度（multiplicity），不是不可约表示的维数；上面的总维数为 15。
也可直接传旧代码风格的 `Rep[SU₄](...)`。

不同方向：`V=(x=Vx,y=Vy,z=Vz)`。逐键指定：
`V=(horizontal=Vh,vertical=Vv)`，其中两个矩阵大小都是 `(Lx,Ly)`。
`Vh[x,y]` 对应格点 `(x,y)` 左边的键，`Vv[x,y]` 对应它下边的键。
缺失竖键自动改为平凡表示。构造时检查允许的 invariant tensor 是否非零；
局部非零本身不保证任意用户指定的量子数布局都能在周期网络上形成非零态。

初始量子数决定初态；演化中的 SVD 可以改变保留的 sectors，完整保留 SU(4)
irreps。`Dmax` 是含 irrep 维数的总维数上限，不是 multiplet 数。
若要固定演化全过程中的 sector 内容，需要另加固定空间截断，本实现不作此约束。

## 几何、空间对称性与演化

SU 坐标 y 向上；全部水平键存在。`D(x,y)--U(x,y-1)` 仅在 `iseven(x+y)`
时存在。三类匹配为：

- `x`：左端 `(x,y)` 满足 `iseven(x+y)` 的水平键。
- `y`：另一类水平键。
- `z`：真实竖键。

这三个标签是蜂窝键方向，不是方格笛卡尔坐标。每层恰好覆盖全部格点一次。
周期 cell 要求 `Lx,Ly` 都为不小于 2 的偶数。`2×2` 是四个原始物理格点。

```julia
parameters=Dict("J"=>1.0)
T_set,lambdax_set,lambday_set=itebd_honeycomb(parameters,
    T_set,lambdax_set,lambday_set,1.0,0.02,16)
```

采用 `x/2,y/2,z,y/2,x/2` 二阶 Trotter，每步默认循环轮换三方向的更新顺序，
减轻方向偏差。只更新真实键，平凡腿始终为维数 1。
张量本身含每条键一端的 `sqrt(s)`，`lambda` 也存 `sqrt(s)`，完全沿用旧两点 SU
的约定。导出 CTMRG 时直接使用张量，不额外乘 lambda，并反射 y 坐标。

该网络没有预先选定二聚体方向，但**不强制 C3 空间对称性**，三类键谱独立。
`uniform=true` 仅复制两个子晶格的初始张量，也不等于 C3 投影。
是否自发产生键序应比较三方向能量，并检查步长、D、初态与 cell 的依赖。

## 能量

默认 `convention=:traceless`：

`H = J sum_<ij> (P_ij-I/4)`。

可选 `convention=:permutation`：`H = J sum_<ij> P_ij`。
后者相对于前者的能量每键增加 `J/4`、每原始格点增加 `3J/8`。
演化和测量应传相同 convention。

```julia
local_ob=honeycomb_energy_simple(parameters,T_set,lambdax_set,lambday_set)

setting=LS_CTMRG_settings(CTM_ite_nums=100,CTM_conv_tol=1e-7,
    CTM_trun_svd=false,grad_checkpoint=false)
ctm_ob=honeycomb_energy_CTM(parameters,T_set,32,setting)
```

两者均返回：

- `energy_per_site`：每原始格点能量，所有真实键之和除以 `Lx*Ly`。
- `energy_per_bond`：所有真实键的平均。
- `bond_energy=(x=...,y=...,z=...)`：各方向每键平均。
- `bonds` 与 `energies`：逐键坐标、方向和能量，一一对应。

因此 `energy_per_site=(bond_energy.x+bond_energy.y+bond_energy.z)/2`。

`honeycomb_energy_simple` 是外部 lambda 乘积环境的局部估计，不能作为 CTMRG
能量或变分上界。`honeycomb_energy_CTM` 使用真实 CTMRG 环境，并额外返回
`CTM,iterations,corner_error,corner_converged,chi`。有限 chi 不保证严格变分上界。
应检查 CTMRG 迭代收敛及增加 chi 后的能量变化。

受旧 `rotate_AA_cell` 支持范围限制，CTMRG 接口目前接受 `2×2`、`4×4` cell；
SU 与局部能量接受任意偶数矩形 cell。CTMRG 使用模块内旧全局变量，测量应串行调用。

## D=4 benchmark

```sh
julia --startup-file=no SU4/honeycomb/experiments/baseline/Benchmark_D4.jl
```

默认计算 dense `2×2` cell、`Dmax=4`，比较 seeds `1234,2025,42`。
逐步减小时间步长到 `0.001`，随后继续演化检查局部能量和键谱的变化。
对全部初态做 `chi=16` 的 CTMRG 测量，再对其中最低能量态增加 chi。
结果在 `benchmark_D4_2x2/`：`evolution.csv`、`ctm.csv`、各 seed 的 JLD2
检查点和 `best_result.jld2`。CSV 同时保存两种哈密顿量约定。

已有 SU 检查点时，可以单独继续 CTMRG，避免重跑演化：

```sh
julia --startup-file=no -e 'include("SU4/honeycomb/experiments/baseline/Benchmark_D4.jl"); measure_saved_D4()'
```

该接口将各轮测量记录到 `ctm_refined.csv`，保存环境到 `ctm_refined.jld2`，
同时检查继续 CTMRG 迭代和增加 chi 引起的能量变化；收缩使用 4 个 BLAS 线程。

比较 Corboz et al., *Phys. Rev. X* **2**, 041013 (2012) 时，用
`E_site_permutation = E_site_traceless + 3/8`（J=1）。论文 Fig. 2 的 D=4
读图参考约为 `-0.753`（2×2）和 `-0.763`（4×4），仅为约 0.001 精度的图上估计，
不是作者提供的精确原始数据。论文 Appendix A2 的优化流程是 SU 后继续 FU，
所以该文结果不能作为纯 SU 必须严格重现的数值。

论文：https://doi.org/10.1103/PhysRevX.2.041013

2026-10-28 实测（dense，2×2，D=4，最终 dt=0.001）：三个初态的
simple-lambda 能量均为 `E(P)/site=-0.7525672268`。代表态 CTMRG 的
`chi=8,16,24` 能量分别为 `-0.7533617160126682`、`-0.7533617160127664`、
`-0.7533617160127666`，角谱误差约 `8e-7`（测量容差 `1e-6`）。
这是 chi 稳定性检查，不是总误差界；仍有有限步长与 SU 优化误差。
代码默认去迹约定的 CTMRG 能量为 `-1.1283617160127666`。
三方向 `<P>` 约为 `(-0.347519,-0.347522,-0.811682)`，对应二聚化分支。
详细数据在 `benchmark_D4_2x2/ctm_refined.csv`，SU 检查点和 CTMRG 环境也保存在该目录。

## 小 D 的 SU(4) 对称测试

```sh
julia --startup-file=no -e 'include("SU4/honeycomb/experiments/initialization/Test_small_D_SU4.jl"); run_small_D_SU4(;truncation=:rank)'
julia --startup-file=no -e 'include("SU4/honeycomb/experiments/initialization/Test_small_D_SU4.jl"); run_small_D_SU4(;seeds=(1234,),truncation=:space,pilot=true,measure=false)'
```

第一条用原有总维数截断，比较 `4⊕4bar`（Dmax=8）、`1⊕4⊕4bar`（Dmax=9）、
`1⊕4⊕6⊕4bar`（Dmax=15），每组两个 seed。2×2 cell、两子晶格物理表示都是 4，
随机 SU(4) invariant tensor，不强制 C3。分阶段演化到 dt=0.002，
正常完成的五个运行总 tau=21.9，并继续同 dt 检查局部能量与谱变化。

结果保存于 `small_D_SU4/`：`evolution.csv`、逐键 irrep/中心电荷/Schmidt 权重的
`sectors.csv`、`ctm.csv`、`summary.csv`、`errors.txt` 以及 JLD2 检查点。
以下全部采用 H=sum(P_ij)、J=1 的每格点能量：

| 初始空间 / Dmax | seed | 最终实际键维数 | CTMRG E/site | 状态 |
|---|---:|---|---:|---|
| 4⊕4bar / 8 | 1234 | 4–8 | 约 -0.46327 | chi=48 能量稳定，角谱误差约 2e-4，未充分收敛 |
| 4⊕4bar / 8 | 2025 | 4–8 | 约 -0.46598 | chi=32 能量稳定，角谱误差约 1e-3，未充分收敛 |
| 1⊕4⊕4bar / 9 | 1234 | 1–4 | -0.45833333 | CTMRG 收敛 |
| 1⊕4⊕4bar / 9 | 2025 | — | — | tau=3 后下一阶段发生空虚拟空间，SU 停止 |
| 1⊕4⊕6⊕4bar / 15 | 1234 | 1–6 | -0.56250000 | CTMRG 收敛，四格点 singlet 团簇构型 |
| 1⊕4⊕6⊕4bar / 15 | 2025 | 7–8 | -0.53779652 | CTMRG 收敛 |

失败运行的 `ctm.csv` 行只是回滚到上一检查点后的诊断，不能当作完成演化的结果。
能量稳定也不等于角谱收敛；尤其 D8 的数值只作初步估计。
两次 D8 的局部能量几乎相同，但 CTMRG 结果有差异，不能只用局部能量筛选态。

第二条是固定各 irrep 简并度的短程对照，用 TensorKit `truncspace(V)` 替换
旧两点函数的 rank 截断，其余收缩和演化门仍复用原函数；不改生产库。
数据位于 `small_D_SU4_fixed/`。只跑至 tau=6、dt=0.05，**不是收敛能量测试**：

| 固定虚拟空间 | seed | 保留维数 | 最后 simple-lambda E/site |
|---|---:|---|---:|
| 4⊕4bar | 1234 | 8 | -0.45489866 |
| 1⊕4⊕4bar | 1234 | 9 | -0.45833332 |
| 1⊕4⊕6⊕4bar | 1234 | 15 | -0.56250000 |

固定空间只保留表示槽位，不强迫其奇异值有非零权重。D15 在此短跑中也出现每条键
一个中心电荷类别占据几乎全部权重，与 rank 截断的团簇态一致。
因此不能把问题仅归因于删掉 sector，也不能把这些能量当作各表示族的最优值。
这些初态和普通 SU 流程目前均不如之前 dense D=4 的 -0.75336172；
后续应比较 C3 相容初态、更温和的初始化/演化与受限变分优化，而不是单纯增大 Dmax。

补充更新审查：`tests/audit_update.jl` 对跨 singlet 团簇的水平和竖直 SU(4)
两点门做无截断对照，四个测试的相对误差均约 `4e-15`，支持门与重构收缩正确。
但本机 TensorKit 后端的 `truncrank` 按奇异值排序，在遇到第一个超过总维数预算的
multiplet 时直接停止，不会继续寻找能装下的其他表示。这不是固定物理维数预算下
整体最优的 multiplet 选择：例如预算 5 时可能只保留 4 而漏掉还能保留的 1；
预算 9 时若首个 multiplet 为 10，甚至返回空空间。审查脚本复现了这两个情况。
因此上述 rank 小 D 结果还受这一截断策略影响，不能用来判断 SU(4) ansatz 的最优能力。
这能解释空空间和部分过度丢块，但固定空间短跑也出现团簇化，尚不能把全部高能量
归因于这个后端策略。

## 修正后的总维数截断

`itebd_honeycomb` 现在默认 `truncation=:optimal`。新增
`src/bosonic/honeycomb/multiplet_truncation.jl`，在总物理维数 Dmax 内选择完整
irreps，使保留的 Frobenius 范数平方 `sum_r dim(r)*sum_k s[r,k]^2` 最大。
对每个 irrep 只选择前 k 个奇异值，使用动态规划求预算内的最优组合。
不强制每类中心电荷都存在，也不施加点群投影；只修正局部 SVD 的预算选择。

旧行为可用 `truncation=:legacy` 调用。策略类型只传给蜂窝 SU 的两点更新，
没有改动 TensorKit 自身的整数 rank 策略、旧三角晶格代码或 CTMRG 策略。
dense 的新策略与普通 rank 截断等价。新接口对不可行预算或零范数谱给出明确错误，
而不是继续对空键归一化。

```sh
julia --startup-file=no SU4/honeycomb/tests/test_multiplet_truncation.jl
julia --startup-file=no SU4/honeycomb/experiments/initialization/Test_small_D_SU4.jl
```

第二条默认使用新策略，结果写入 `small_D_SU4_optimal/`，不会覆盖旧数据。
`test_multiplet_truncation.jl` 包含预算反例、逐一枚举所有简并度组合的独立对照、
Frobenius 截断误差和 dense 对照。当前实测环境为 Julia 1.11.6 / TensorKit 0.17。
源文件也提供 TensorKit 0.14--0.16 的旧截断接口适配，该分支尚未在本轮运行中测试。

## 严格星形 tetramer 覆盖 PEPS（没有虚时演化）

`Measure_star_RVB.jl` 的 `initial_star_RVB()` 使用 `V=1⊕4⊕4bar`，D=9。
全出射约定只保留 `(4,4,4)` 的 epsilon 中心通道，以及循环对称的
`(4bar,1,1)` 叶子通道。它生成不重叠的最近邻四格点星形 singlet 覆盖叠加，
与 `Test_C3_initial.jl` 的一般随机 C3 初态不同。每个覆盖的中心/叶子数相同，
所以两种通道的非零整体相对权重不提供变分参数。

`Exact_star_RVB.py` 独立枚举周期环面的全部合法覆盖。8、16 格点用稀疏物理
波函数直接计算，并与两个覆盖间 epsilon 网络收缩交叉检查；较大系统直接收缩
覆盖间的 overlap 和算符矩阵元，没有环境截断。这些是有限系统结果，不能把小
环面的共振增益直接当作无限晶格能量。

| brick-wall 环面 | 格点数 | 覆盖数 | E(P)/site |
|---|---:|---:|---:|
| 4×2 | 8 | 4 | -0.6666666667 |
| 4×4 | 16 | 8 | -0.6610738255 |
| 8×4 | 32 | 8 | -0.5653963614 |
| 12×4 | 48 | 8 | -0.5626557364 |
| 8×8 | 64 | 40 | -0.5648167005 |
| 12×8 | 96 | 32 | -0.5626556888 |

这里 `H=ΣP`，J=1；traceless 约定的每格点能量再减 `3/8`。
单覆盖乘积态的基准严格为 `-9/16=-0.5625`。64 格点上水平键
`<P>=-0.3773146907`，竖直键 `-0.3750040196`；矩形环面未保持三方向旋转。
SU(4) 对称性严格给出单格点 `rho=I4/4`、15 个生成元期望均为零。
四格点星形 singlet 投影 `Q_star=|epsilon><epsilon|/24` 的期望为
`0.2587536673`；同一单覆盖的平移平均基准是 `33/128=0.2578125`。
64 格点上两条竖直键（起点 `(0,0)` 与 `(4,0)`）的 connected
`<P_b P_b'>-<P_b><P_b'>=0.2366695234`，提示明显的覆盖有序关联；
还不能仅凭这些有限尺寸结果证明热力学相。
同位置两点的 `sum_a<T_i^a T_j^a>=0.0011563843`，采用
`tr(T^a T^b)=delta_ab/2`，即 `( <P_ij>-1/4 )/2`。
96 格点上的 `Q_star=0.2578762194`；相隔六格的两条竖直键关联
为 `0.3906250002`。这些矩形环面的边界、距离及覆盖族不同，不能直接
把它们当作保持 C3 的单一有限尺寸标度序列。

**本次新态的 CTMRG 接入尚未通过验证，不能据此认定旧 CTMRG 有错误。**
通过新增蜂窝接入函数使用 2×2 环境，在 chi=16/32/48 下分别给出
`E(P)=-0.5625/-0.24998477/-0.5625`，不随 chi 稳定；chi=16、48 还出现
超出 swap 谱界 `[-1,1]` 的单键期望。chi=32 虽然角谱误差报零，也不能据此
判定收缩正确。原始诊断保存在 `star_RVB/results_L2.csv`。
因此目前可靠的是精确有限环面数据，它们不支持把这个严格星形覆盖构造当作
已经得到低能共振液体的证据；其能量明显高于之前 dense D=4 的约 `-0.75336`。

```sh
python SU4/honeycomb/experiments/star/Exact_star_RVB.py
python SU4/honeycomb/experiments/star/Exact_star_RVB.py --overlap
python SU4/honeycomb/experiments/star/Exact_star_RVB.py --observables
julia --startup-file=no SU4/honeycomb/experiments/star/Measure_star_RVB.jl
```

前三条计算精确有限系统；最后一条用于复现尚未通过验证的 CTMRG 接入结果，不能用来
替代有限尺寸或收缩收敛检查。数值和观测量的定义保存在 `star_RVB/*.json`。

`Audit_star_CTM.jl` 使用同样的 D=9 空间，先只选择一个合法的星形覆盖；
其内部键 `<P>=-1`、外部键 `<P>=1/4`，提供每一条键的独立精确对照。
同时检查恒等算符插入应为 1，再用该覆盖的 CTM 环境初始化完整叠加态。
它用于区分张量/测量接入错误与特定初态、边界扇区或有限 chi 的收缩问题。

2026-10-29 的 4×4 cell 复查：完整星形叠加张量在 chi=16、PBC 初始化下，
27 次迭代后给出 `E(P)/site=-0.5624999999999724`，角谱误差 `2.34e-21`。
24 条真实键都落在 swap 的谱界内，且逐键对应一个合法的星形乘积覆盖：
中心为零基坐标 `(0,0),(0,1),(2,2),(2,3)`，内部键 -1、外部键 1/4，
最大误差 `4.0e-14`。这表明此前不能将 2×2 的失败归因于旧 CTMRG 本身。
该有序覆盖不能由 2×2 cell 表示；4×4 环境选取了一个允许的覆盖边界扇区。
单次固定点结果尚不等于对不同 chi、边界扇区和完整对称混合态的全面收敛证明。
`measure_star_RVB()` 的默认测量 cell 因此改为 4×4（局域张量本身仍只有两类）。

已知乘积态对照在 chi=16/32/48 均通过：24 条键与精确值的最大误差
`4.02e-14`，恒等算符插入误差 `9.10e-15`，两次迭代达到角谱收敛。
然后保持完整叠加张量不变，仅以该乘积态 CTM 作为 `CTM0`，三个 chi 均得到
`E(P)/site=-0.5624999999999724`，恒等算符误差 `8.99e-15`，两次迭代收敛。
这验证了该覆盖扇区的收缩，以及该对照下张量、几何映射和算符插入的基本一致性。
从 PBC 初始化出发，4×4 的 chi=48 也得到同能量的另一个合法星形覆盖；
但 chi=32 在 100 次迭代后仍周期振荡，出现 `<P><-1`，不能采用其单键结果。
因此不能把问题概括为旧 CTMRG 有 bug，也不能把热启动的覆盖扇区固定点
冒充为对称叠加态的、与边界无关的唯一收缩。现有证据表明结果对 cell 和初始
边界扇区敏感；普通初始化在 chi=32 的振荡原因尚未追踪到具体截断步骤。

复现这个通过已知对照的测量流程：

```sh
julia --startup-file=no SU4/honeycomb/experiments/star/Audit_star_CTM.jl
```

实际 iPEPS TensorMap 保存于 `star_RVB/initial_tensors_L4.jld2`，
`full` 为完整覆盖叠加，`product` 为对照乘积态；各 chi 的 CTM 和观测量分别
保存于 `audit_product_chi*.jld2`、`audit_full_warm_chi*.jld2`。

## 从完整星形覆盖态出发的 simple-update 试跑

`SimpleUpdate_star_RVB.jl` 复用 `initial_star_RVB()`、原有两点 SU 更新和
`WholeMultipletBudget`。默认 cell=4×4、初始 `V=1⊕4⊕4bar`、Dmax=9；
三方向键独立，演化不额外作 C3 投影，也不把更新张量投回星形覆盖通道。
lambda 初始为单位阵，原始 T 保持不变；先做三轮恒等门扫掠整理 SU 规范，
这部分 tau=0，然后从 dt=0.01 开始，计划逐步降到 dt=0.002。

本次 Dmax=9 试跑未完成：tau=0.2 的检查点局部 lambda 估计为
`E(P)/site=-0.5660730888`；后续阶段在竖直更新处遇到空奇异谱，回滚到该检查点。
这不是已收敛的变分能量。原始张量配单位 lambda 的局部估计甚至是 `+1/6`，
而其已验证覆盖扇区的 CTMRG 能量是 `-9/16`，说明这种强覆盖约束初态对
simple 环境/规范很敏感，不能用局部能量替代物理测量。

一个独立的表示空间问题是：交换门的算符 Schmidt 表示为 `1⊕15`。
若要保留初始 `1⊕4⊕4bar` 的所有通道，再加入完整伴随 irrep 15，总维数是24。
因此 Dmax=9 不能平滑保留这组全部通道；这一计数本身既不能解释空谱错误的
全部原因，也不能证明所有更低能态都至少需要 D=24。

结果和检查点在 `star_SU_D9_L4/`；`failure.txt` 保留失败堆栈，
`stage1.jld2` 为最后成功的 tau=0.2 检查点，CTMRG 单独保存并标记
收缩是否收敛、是否通过物理范围和恒等算符检查。

```sh
julia --startup-file=no SU4/honeycomb/experiments/star/SimpleUpdate_star_RVB.jl
```

进一步的规范无关检查表明，这段 D=9 演化没有取得物理优化：
`Inspect_star_channels.jl` 检查初态、恒等门规范整理后、tau=0.2 的张量，
原星形电荷通道外的权重均为零。SU 的 SVD 会改变部分虚拟腿的对偶约定，
因此逐键使用配对的逆转置映射对齐（每条键两端映射相消，不改变 PEPS）。
对齐后，四种局域星形通道分别只是初始通道的标量倍数，最大残差 `1.11e-16`。
在16、32、64、96格点环面枚举全部覆盖后，各覆盖相对系数的最大变化分别为
`8.9e-16,5.3e-15,8.9e-15,7.1e-15`；只剩总体归一化变化。
因此这些环面上的归一化波函数及能量与初态相同，局部 lambda 估计的变化不能
作为能量降低的证据。此结论来自覆盖系数对照，不依赖未收敛的 CTMRG。

tau=0.2 检查点的普通初始化 CTMRG 在 chi=16/32/48 均未收敛，且单键值越界，
全部不作为物理能量结果。原始覆盖环境也不能不经规范映射直接用于更新后的
张量；本次直接跨规范复用环境的尝试在 SVD 处失败，未产生有效测量。

```sh
julia --startup-file=no SU4/honeycomb/experiments/star/Inspect_star_channels.jl
python SU4/honeycomb/experiments/star/Exact_star_RVB.py --compare-evolved
```

后两条只分析已保存检查点。下一步应区分两个控制：增加预算以允许伴随15通道，
或对初态加入小的 SU(4)/C3 对称扰动，离开严格星形通道子空间。本轮没有把这些
尚未运行的控制当作优化成功的结果，也没有将空谱错误全部归因于维数限制。

### 首个两点门的实际未截断奇异谱

`Inspect_star_first_gate.jl` 直接读取完整星形初态，外部 lambda=I，计算水平键
`exp(-dt*P)` 后，按两个格点分组作完整 SVD，**不使用 Dmax 截断**。
以 `1e-10*最大奇异值` 判定非零：

| dt | 非零 irreps（各一份） | 总秩 | s_15 |
|---|---|---:|---:|
| 0 | 1,4,4bar | 9 | 0 |
| 0.001 | 1,4,4bar,15 | 24 | 0.000125000020833 |
| 0.01 | 1,4,4bar,15 | 24 | 0.001250020833437 |

因此，对**这个严格星形初态的第一道门**，新增非零 irrep 确实是15。
这比仅列举交换门的算符 Schmidt 表示更强，是直接测得的张量奇异谱。
但不能推广为所有 D=9 张量更新时都只增长15：一般候选空间是
`V⊗(1⊕15)`，其中 `4⊗15=4⊕20_(0,1,1)⊕36_(2,0,1)`，
4bar 有共轭分解，还可能增加原有 irrep 的 multiplicity。
严格星形通道使这些20、36和额外 multiplicity 在首门的矩阵元为零。
24只是完整保留本次首门非零谱的维数，并非所有优化路线的最低 D。

```sh
julia --startup-file=no SU4/honeycomb/experiments/star/Inspect_star_first_gate.jl
```
