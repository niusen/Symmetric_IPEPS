# SU(4) 三角晶格：复用 src 的 TensorMap iPESS

只考虑最近邻 J 项，默认 `H = J sum_<ij> (P_ij - I/4)`，2×2 unit cell。
`dense` 指无对称分块的 `ComplexSpace`，状态和计算全程使用 TensorMap。

## 实际调用的旧代码

| 工作 | src 中的函数 |
|---|---|
| SU(4) 相互作用/MPO | `bosonic/SUN_spin_op.jl`: `SUN_spin(4; dense=true)` |
| 初态与 iPESS simple update | `triangle/simple_update/triangle_SimpleUpdate_iPESS.jl`: `initial_iPESS`, `triangle_update_iPESS`, `triangle_gate_iPESS_simplified` |
| 三角形演化门 | `triangle/simple_update/triangle_SimpleUpdate.jl`: `gate_RU_LD_RD` |
| iPESS 转换为测量用 iPEPS | `triangle_SimpleUpdate_iPESS.jl`: `convert_iPESS_to_iPEPS` |
| 环境 | `bosonic/CTMRG_unitcell.jl`: `CTMRG_cell` |
| 能量和键耦合 | `triangle/triangle_spin_model_cell.jl`: `evaluate_ob_cell` |
| 连通关联所需单点密度矩阵 | 同一文件的 `ob_onsite` |

没有另写更新或收缩算法。原文件只添加 ComplexSpace 分支及 TensorKit 接口兼容。
转换成 iPEPS 仅用于沿用旧 CTMRG；演化仍是三角晶格 iPESS simple update。

## 文件位置

可复用的 Julia 函数集中在 `src/bosonic/triangle/su4_dense/`：

- `TensorMapSU4.jl`：加载旧代码、设置参数、扩展虚空间、转换观测量坐标、检查点读写。
- `Observables_iPESS.jl`：12 条键的坐标、文本/CSV/MAT 格式辅助与 Fourier 分析，无收缩算法。
- `LongRun.jl`：多 D、多 seed 的升 D 续接或独立随机计算、断点续算、MAT 输出及检查点重新测量。
- `PlotBondPattern.py`：完整键图的绘图工具。

`SU4/simple_update_dense/` 保留简短运行入口和使用文件：

- `LongRun_D4_D16.jl`、`Evaluate_observables.jl`：加载 src 模块并启动计算。
- `job_su4_D4_D16.sh`：直接调用 Julia 的服务器启动脚本。
- `tests/runtests.jl`：数值正确性与运行流程检查。

## 观测量约定

`exchange = <P_ij>`；`coupling = <P_ij>-1/4`；生成元规范
`tr(T^a T^b)=delta_ab/2` 下 `generator_dot = coupling/2`。
`connected_generator_dot = (<P_ij> - tr(rho_i rho_j))/2`。
所有量均由旧 CTMRG 环境测得。

每格点能量是 12 条独立键能之和除以 4。三方向为 x、y、x+y，
基矢夹角 120 度。A=(1,1)、B=(2,1)、C=(1,2)、D=(2,2)。
`bond_table` 保留跨胞位移，避免把同一子格对上的不同键合并。
原始方向矩阵按 simplex 索引；`coupling_origin_*` 按实际起点索引。
还输出各方向的空间调制和 (pi,0)、(0,pi)、(pi,pi) Fourier 系数，
用于检查 plaquette/stripy 候选；不只用三个方向均值判相。

## 检查与重新测量

使用与旧 src 相同的 Julia 包环境（TensorKit、TensorKitSectors、Zygote、
ChainRulesCore、KrylovKit、MAT）：

```sh
julia --startup-file=no SU4/simple_update_dense/tests/runtests.jl
julia --startup-file=no SU4/simple_update_dense/Evaluate_observables.jl /path/to/checkpoint.jls
python src/bosonic/triangle/su4_dense/PlotBondPattern.py /path/to/observables_bonds.csv
```

批量运行见 `LONG_RUN.md`。历史 `test_runs_D4_2x2` 的数据/图像保留，
它们属于此前 Array 实现的结果；新程序不将其作为 TensorMap 检查点续算。
新格式使用独立输出根目录 `long_runs_tensormap`。默认升 D 计算保存在其 `continuation/`
子目录，可复用根目录里已有的第一个 D 的 TensorMap 检查点；独立随机模式为
`LongRunConfig(initialization=:random)`。详情见 `LONG_RUN.md`。
