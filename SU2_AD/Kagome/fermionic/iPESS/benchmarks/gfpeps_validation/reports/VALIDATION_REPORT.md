# 有自旋 Gaussian PEPS / iPESS 的普通张量验证

验证对象是 `magnetic plateau/kagome` 工作目录保存的 **同一个有限 D 的
Gaussian 变分态**。参考值由这个态的 Gaussian 收缩得到，不是 Hamiltonian
精确基态的能量。没有重新优化，也没有做 Gutzwiller 投影。

2026-11-21：D=4、D=8 的两套构造均已完成能量对照。D=4 能量绝对误差为
1e-10～1e-8 量级；D=8 在 χ=64 时小于 1e-6。这里的“一致”指这些数值精度内
的一致，不是双精度机器误差，也不是对所有优化代码的证明。
全部完成与未收敛运行均保存在
[`comparison.md`](../../results/gfpeps_validation/comparison.md)。

## 验证的态

| 构造 | D | 源文件 |
|---|---:|---|
| 原生 Gaussian iPESS | 4 | `gaussian_ipess/orig_m2_s12102.npz` |
| 原生 Gaussian iPESS | 8 | `gaussian_ipess/orig_m3_s12003.npz` |
| 直接 Gaussian iPEPS | 4 | `gfpeps_finite_D_flux/orig_m2_r0.npz` |
| 直接 Gaussian iPEPS | 8 | `gfpeps_finite_D_flux/orig_m3_r0.npz` |

这些是 manifest 中选择的 `orig` 态，SHA256 已核验并保存了只读源文件的
本地快照。D=2^m 是一条原始虚腿包括两个自旋通道的总 Fock 维数。
原生 iPESS 的物理腿维数为 4；三个物理格点合并之后的 iPEPS 物理腿维数为 64。

已保存的 Julia TensorMap checkpoint（`state.jld2`）及逐元素 Fock 张量：

| 构造 | D | Julia checkpoint | Fock tensors |
|---|---:|---|---|
| iPESS | 4 | [state.jld2](../../results/gfpeps_validation/ipess_D4/state.jld2) | [tensors.mat](../../results/gfpeps_validation/ipess_D4/tensors.mat) |
| iPESS | 8 | [state.jld2](../../results/gfpeps_validation/ipess_D8/state.jld2) | [tensors.mat](../../results/gfpeps_validation/ipess_D8/tensors.mat) |
| iPEPS | 4 | [state.jld2](../../results/gfpeps_validation/ipeps_D4/state.jld2) | [tensors.mat](../../results/gfpeps_validation/ipeps_D4/tensors.mat) |
| iPEPS | 8 | [state.jld2](../../results/gfpeps_validation/ipeps_D8/state.jld2) | [tensors.mat](../../results/gfpeps_validation/ipeps_D8/tensors.mat) |

iPESS checkpoint 的键为 `state, fields, physical_basis`，其中 `state` 保留
B/Tu/Td 三角形结构；iPEPS checkpoint 的键为 `A, U, fields, physical_basis`。
它们保存源态转换后的原始规范，测量时的可选规范平衡按参数重新生成。

## 能量与误差

所有能量均以 hopping 幅度为单位、除以物理 kagome 格点数。
`Egrand = Ekin - mu_up*n_up - mu_down*n_down`。
磁性晶胞有九个物理格点，三个基本 kagome 晶胞。

| 构造 | D | χ | Gaussian Ekin | CTMRG Ekin | ΔEkin | ΔEgrand |
|---|---:|---:|---:|---:|---:|---:|
| iPESS | 4 | 32 | -1.564326178941268 | -1.564326179477337 | -5.36e-10 | -7.90e-10 |
| iPEPS | 4 | 48 | -1.660536658090060 | -1.660536667897829 | -9.81e-9 | +6.56e-8 |
| iPESS | 8 | 64 | -1.700055656483297 | -1.700055305683102 | +3.51e-7 | -4.56e-7 |
| iPEPS | 8 | 64 | -1.723648984724453 | -1.723649704756879 | -7.20e-7 | -5.02e-7 |

D=4 iPESS 的 projector cutoff 为 1e-10，角谱误差 7.98e-9，通过所设
1e-8 收敛阈值。D=4 iPEPS 使用八轮严格可逆的虚键规范平衡、cutoff=3e-10，
角谱误差 2.98e-8，通过所设 1e-7 阈值。后者继续到 χ=64，动能仅改变
8.70e-12，grand energy 仅改变 1.68e-11；更严的 1e-9 角谱阈值没有通过，
不能把该次运行改称为角谱收敛。

同时检查了各格点的两自旋密度、各条上/下三角形键的复 hopping、物理
spin flip 和 pair creation。D=4 iPESS 的最大已检查关联误差约 1.23e-8，
D=4 iPEPS 约 2.94e-7。能量接近并不等于每个关联函数都达到同样精度。

D=8 两套 χ=64 运行的 cutoff 均为 1e-10，分别经过 17、9 次迭代，角谱误差
为 1.18e-7、1.57e-7，均通过所设 2e-7 阈值。最大已检查关联误差分别为
4.38e-6、3.94e-6。实际投影秩的最小值和最大值都是 64，cutoff 没有额外降低秩。
只读计数器逐次调用原逆平方根函数，12 项逐元素一致性测试通过。

| D=8 构造 | χ=16 的 ΔEkin | χ=32 的 ΔEkin | χ=64 的 ΔEkin |
|---|---:|---:|---:|
| iPESS | +2.21e-4（角谱未收敛） | +1.21e-5 | +3.51e-7 |
| iPEPS | -2.46e-4 | -2.59e-5 | -7.20e-7 |

三档 χ 分别使用 cutoff=1e-8、1e-9、1e-10，因此这张表同时展示了 χ 与
数值截断精度的提高，不能把它拟合成纯粹的 χ 标度律。χ=64 的实际保留秩说明
该档没有因 cutoff 额外丢掉态。数据支持剩余误差来自有限环境精度；没有发现
不随精度提高而消失的能量/fermionic 符号不一致。尚未检查 χ>64，不能宣称
D=8 已达到 1e-9 精度。

## 独立验证与实现范围

* 保留完整 Nambu 数据，经 Pfaffian 振幅转换到 Z2 Fock 张量。随机纯态的
  全 Nambu 协方差回算误差小于 2.50e-14。
* 显式 Fock 虚键投影与 Gaussian 投影相符，误差小于 8.89e-15。
* 原生 iPESS 的四张量三角形和五张量基本胞分别与独立 Gaussian 收缩对比。
  D=8 的全协方差误差分别为 1.72e-15、2.66e-15。
* Gaussian 能量既由动量空间矩阵迹计算，又由实空间逐键相关求和计算，
  两者差异小于 3.56e-15。积分网格从 192² 加密到 768² 后，四个态的能量
  都只改变几个 1e-15。
* 使用现有共享 CTMRG 的环境迭代与 onsite/x/y/diagonal observable 收缩。
  kagome 适配层补充了 3×1 晶胞旋转；六种旧尺寸与新增 3×1 共七项比较通过。
* D=8 原双层构造产生很大的中间张量，曾观察到约 20 GB 进程峰值。
  新增可选的等价收缩顺序，仅先收缩物理腿再融合虚腿，不作近似。
  19 项张量比较最大相对误差 2.45e-16；固定同一个 CTM 环境时，全套测量
  的能量/密度差异小于 3.86e-12，关联差异小于 1.95e-11。
* 规范平衡在每条虚键两端插入互逆、保持 parity 的矩阵，不截断态。
  普通及带 parity twist 的封闭三格点网络全振幅比较误差小于 6.46e-16。

源态采用 `b=(up,down-hole)` 基底。本次张量保留该基底，测量时使用相应的
变换 Hamiltonian，包括 `n_down=1-n_hole` 和化学势常数。不要把这些 checkpoint
直接交给未做粒子空穴变换的普通电子 Hubbard evaluator。完整腿序、键相位
和算符约定见 [README.md](README.md) 及
[CONVENTIONS.md](../../../../../src/fermionic/kagome/gfpeps/CONVENTIONS.md)。

原有三角晶格与共享 CTMRG 源文件没有修改。新增功能位于
`src/fermionic/kagome/gfpeps/`；benchmark、检查脚本及诊断记录位于本目录。
失败的初版腿映射、角谱振荡和被停止的运行在 [diagnostics.md](diagnostics.md)
中单独说明，未隐藏或计为通过。

本验证检查 Gaussian→普通张量、fermionic blocking、CTMRG 及这些观测量的
一致性。它本身不验证 SU/FU 更新、AD 梯度或优化算法是否找到变分最优态。

## 复现最终结果

从 `SU2_AD` 启动 Julia，先执行：

```julia
include("Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/run.jl")
```

`prepare.py` 的四条源态导出命令和独立检查命令见 README。最终采用的运行如下；
warmstart 文件是同一 family/D 目录下此前保存的环境，不是另一个态的环境。

```julia
# Native iPESS D=4：原共享双层核的结果另有保存；紧凑核与其等价。
validate_saved("ipess",4,32;cutoff=1e-10,maxiter=60,tol=1e-8,
    save_environment=true)
# Direct iPEPS D=4：同一态的可逆虚键规范，不做优化或截断。
validate_saved("ipeps",4,48;cutoff=3e-10,maxiter=40,tol=1e-7,
    gauge_sweeps=8,save_environment=true)
# D=8：顺序执行两套构造，避免同时占用较大内存。
BLAS.set_num_threads(6)
for family in ("ipeps","ipess")
    validate_saved(family,8,16;cutoff=1e-8,maxiter=20,tol=1e-7,
        save_environment=true)
    validate_saved(family,8,32;cutoff=1e-9,maxiter=30,tol=1e-7,
        warmstart="ctm_chi16_cut1.0e-8_compact.jld2",save_environment=true)
    validate_saved(family,8,64;cutoff=1e-10,maxiter=18,tol=2e-7,
        warmstart="ctm_chi32_cut1.0e-9_compact.jld2",save_environment=true)
end
```

不同 BLAS 线程数、近零 projector 模式以及收敛容差会使末尾数字略有变化，
应按报告中的误差量级和关联函数检查判断结果，而不是要求打印小数逐位相同。
