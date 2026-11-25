# Kagome 费米子 iPESS：Z2 Hubbard

沿用 `standard_triangle_Hubbard/iPESS/simpe_update/Z2` 的流程：初始化 →
simple update → 转换到方格 iPEPS → 费米子 CTMRG → 测量 → JLD2 保存。
当前提供 spinful Hubbard、Z2 宇称、矩形 `Lx × Ly` 元胞。每个元胞有三个
独立物理张量和上、下两个 simplex；不强制 C3、SU2 或粒子数守恒。
另提供保留三角形结构的 full update 和自动微分变分优化，入口与算法说明见
[optimization/README.md](optimization/README.md)。目前没有固定粒子数求解器。

## 整理后的目录与服务器运行

功能函数位于 `src/fermionic/kagome/`，分层说明见
[源码目录说明](../../../src/fermionic/kagome/README.md)。运行脚本只负责加载与调用。
Linux 启动方式、依赖检查和初态存放位置见
[server/README.md](server/README.md)；配置入口位于 `gfpeps/configs/`，
历史 GFPEPS 报告在 `benchmarks/gfpeps_validation/reports/`。

## 运行

在仓库 `SU2_AD` 目录下使用已安装依赖的 Julia 环境运行。

```sh
julia Kagome/fermionic/iPESS/SimpleUpdate_iPESS_Z2_ob.jl
julia Kagome/fermionic/iPESS/SimpleUpdate_iPESS_Z2_ob.jl Kagome/fermionic/iPESS/smoke.toml
julia Kagome/fermionic/iPESS/compute_ob.jl Kagome/fermionic/iPESS/results/SU_kagome_iPESS_Z2_D4.jld2 40 200
```

主要依赖与旧代码一致：TensorKit、JLD2；CTMRG 另需 Zygote、ChainRulesCore、
KrylovKit、HalfIntegers。支持本仓库的 TensorKit 0.17 环境。
参数集中在 `config.toml`。输出和 `resume` 路径相对于配置文件。
`smoke.toml` 仅运行两个时间步和三次 CTMRG 迭代，用来检查安装及完整流程，
不能用它的能量判断基态。正式计算需检查虚时、时间步长、D 和 chi 的收敛。

续算时将 `run.resume` 设为保存文件路径，`schedule` 表示额外演化时间。
检查点保存两个 simplex、三个 B、两组共六个 lambda 及参数。
默认拒绝无意中改变续算模型参数；有意改变时设置 `allow_parameter_change=true`。

## 模型、基底和计数

物理空间为 `Rep[ℤ₂](0=>2,1=>2)`，基底按旧代码排列：
`[|0>, |↑↓>, |↑>, |↓>]`，且 `c↓|↑↓> = -|↑>`。

```text
H = -Σ_triangle Σ_(ij)σ (t_ij c†_iσ c_jσ + h.c.)
    + U Σ_i (n_i↑-1/2)(n_i↓-1/2) - μ Σ_i n_i
```

`centered=false` 改为未平移的 `U n↑n↓`；两种约定的 μ 数值不能直接混用。
每个 kagome site 恰好属于两个三角形，因此每个三站点门分配一半 onsite 项；
每条最近邻键只属于一个三角形，跃迁不除以二。
`t_up`、`t_dn` 可为标量或按 `(ab,bc,ca)` 排列的三元数组。
Julia API 也接受复数跃迁，反向键自动取共轭。
所有 `energy_per_site` 除以 `3LxLy`，不是按阻塞后的元胞计数。

## 张量与几何

源码在 `src/fermionic/kagome/`，封装于 `KagomeFermionicZ2` 模块。
`B[x,y,b]` 轴序为 `(down,up,physical)`，`T_u/T_d` 为 `(a,b,c)`。
元胞连接与现有 kagome 玻色子代码一致（x 向右，y 向下）：

```text
up(x,y) = [a(x,y),   b(x,y), c(x,y)]
dn(x,y) = [a(x+1,y), b(x,y), c(x,y+1)]
```

各向的索引独立取周期模数，支持 `1×1`、`2×1`、`2×2` 等矩形元胞。
每个方向的三角形覆盖每个物理 site 一次。
simple update 的 lambda 始终保存在物理 B 所属坐标；更新时吸收外侧 lambda，HOSVD 后移除。
SVD 的 Z2 偶、奇维数可自行重新分配，总维数不超过 `D_max`。

这里使用普通 Z2 TensorMap，因此 `permute` 本身没有费米子符号。
`fpermute` 为每次相邻交换显式乘 `1-2 P_odd⊗P_odd`。
三站点门在 `(a,b,c)` 的 Fock 次序中构造，包含中间 site 的 Jordan–Wigner
宇称串；聚合物理轴及分解、还原 B 时均记录交换。
局部收缩采用 `(external,physical) ← internal` 的偶映射次序。
虚时演化每一步严格为 `dn(dt/2) → up(dt) → dn(dt/2)`。

## 测量

`su_observables` 使用外侧 lambda 的乘积环境，是 simple-update 近似估计，
不是完整网络的变分能量。`measure_ctmrg` 使用已有 `Fermionic_CTMRG_cell` 和
带费米子符号的 `hopping_x/y/diagonala`。
先将三个物理 site 合并为一个 64 维物理轴，虚轴顺序为 `(L,D,R,U)`；
同元胞算符与跨元胞的奇算符均通过同一个物理融合变换。
接入旧方格 CTMRG 时还需保持其左虚腿排列约定，阻塞转换中显式包含
`(-1)^(q_L*(q_R+q_D)+q_D*q_R)` 的虚腿交叉，并匹配对角跃迁的方向负号。
完整三聚体检查覆盖了仅用独立二聚体无法发现的环路符号。
返回密度、双占据、上下三角形能量贡献及 CTMRG 收敛信息。
CTMRG 适配器保留旧实现的模块级状态，不能从多个线程同时调用。
适配器针对 Z2 使用普通秩截断，并安全处理零奇异值，无需 TensorKit
私有分支的 `truncmultiplet` 接口。它仍构造完整阻塞双层，较大 D 时应关注内存。

## 验证

已在 Julia 1.11.6 / TensorKit 0.17.0 上通过 96 项核心与 CTMRG 检查。

```sh
julia Kagome/fermionic/iPESS/tests/runtests.jl
```

加测 CTMRG（PowerShell）：

```powershell
$env:KAGOME_TEST_CTM='1'
julia Kagome/fermionic/iPESS/tests/runtests.jl
```

测试包括独立占据位基底的三站点哈密顿量、复数跃迁、宇称交换、
无截断恒等门及非零虚时门的波函数重构、多元胞几何、断点续算、
原子极限的能量和密度，以及 CTMRG 对接。
另用复数跃迁的精确二聚体分别验证元胞内、水平、竖直、对角键，并验证
非均匀 2×2 元胞的混合二聚体覆盖、密度及双占据。
CTMRG 测试也可单独运行 `julia Kagome/fermionic/iPESS/tests/test_ctmrg.jl`。

Full update 和 AD 的独立测试为
`julia Kagome/fermionic/iPESS/tests/test_optimization.jl`。

## 显式 onsite 配对

可选参数 `pairing=Delta` 加入
`Delta*c_up†*c_down† + conj(Delta)*c_down*c_up`。
Julia API 支持复数 Delta；TOML 的 `[model]` 中可直接设置实数 `pairing`。
未设置时为零，旧配置和检查点参数不增加额外键。
配对项与 U、mu 一样，在两个三角形门中各计一半，完整 CTM 能量只计一次。
物理基底仍为 `|0>, |up dn>, |up>, |dn>`，总粒子数宇称守恒。

SU 估计和 CTM 测量新增 `pair_amplitude=<c_down*c_up>` 及逐站点
`pair_amplitude_set`，复数相位不会被丢弃。
模型测试见 `tests/test_pairing.jl`；D=4 三种优化与精确 BdG 的比较见
`benchmarks/pairing_D4/README.md`。

## 从三角晶格继承的 Hofstadter flux

`benchmarks/flux_D4/README.md` 给出从每个三角形 flux 为 `pi/2` 的母晶格
删键得到 kagome 的完整规范、能带和 D=4 测试流程。保留键的 hopping 不变，
六边形 flux 为 `pi`；在本仓库的哈密顿量约定下，两个三角形均使用
`t_up=t_dn=(1,1,-im)`。每个自旋有三条带，陈数从低到高为 `(-1,0,1)`，
中间为零能平带，两侧带隙均为 `sqrt(3)`。

测试包含 SU、保留 simplex 的 CTM-overlap FU、自动微分 L-BFGS，
以及不同初态、投影截断、最终梯度和独立 CTM 测量的检查。
能量接近不等于准确；运行状态和相对精确自由费米子基态的误差需同时查看。
该模型新增的 38 项检查见 `tests/test_flux.jl`。
