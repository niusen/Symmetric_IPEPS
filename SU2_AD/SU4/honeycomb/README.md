# SU(4) honeycomb：两点 simple update 与能量测量

每个原始格点一个 TensorMap，物理空间为基本表示 `4`，不合并 A/B。
采用 brick-wall：张量顺序为 `(L,D,R,U,physical)`，三条真实虚拟腿和一条
维数为 1 的平凡腿。A/B 两个子晶格的物理表示都是 `4`，不是 `4`/`4⁺` 交替。

## 运行入口

在 `SU2_AD` 下，使用已安装 TensorKit、TensorKitSectors、SUNRepresentations、
ChainRulesCore、Zygote、KrylovKit、JLD2 的 Julia 环境运行：

```sh
julia --startup-file=no SU4/honeycomb/SimpleUpdate_ob.jl dense
julia --startup-file=no SU4/honeycomb/SimpleUpdate_ob.jl SU4
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
julia --startup-file=no SU4/honeycomb/Benchmark_D4.jl
```

默认计算 dense `2×2` cell、`Dmax=4`，比较 seeds `1234,2025,42`。
逐步减小时间步长到 `0.001`，随后继续演化检查局部能量和键谱的变化。
对全部初态做 `chi=16` 的 CTMRG 测量，再对其中最低能量态增加 chi。
结果在 `benchmark_D4_2x2/`：`evolution.csv`、`ctm.csv`、各 seed 的 JLD2
检查点和 `best_result.jld2`。CSV 同时保存两种哈密顿量约定。

已有 SU 检查点时，可以单独继续 CTMRG，避免重跑演化：

```sh
julia --startup-file=no -e 'include("SU4/honeycomb/Benchmark_D4.jl"); measure_saved_D4()'
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
