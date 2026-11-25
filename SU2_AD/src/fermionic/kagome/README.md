# Kagome 费米子 Z2 iPESS 源码

从投影 GF-iPESS 出发计算 **Heisenberg + 磁场**，从 [`spin/`](spin/) 开始。Hubbard 和自旋模型共用 Kagome 的 SU、FU 和梯度优化器；模型相关的三角门与能量单独定义。修改耦合常数 `J`、`h` 或截断参数只需改 [运行配置 D4.toml](../../../Kagome/fermionic/iPESS/spin/D4.toml)。

| 要找的功能 | 源码 |
|---|---|
| Heisenberg 加磁场的三角形 Hamiltonian（SU/FU）及可微 CTM 能量（AD） | [spin/model.jl](spin/model.jl) |
| 导入投影初态、分派三种优化、测量和保存 | [spin/workflow.jl](spin/workflow.jl) |
| Simple update：三角门作用与截断、虚时演化 | [simple_update.jl](simple_update.jl) |
| Full update：三角环境、迭代拟合三个 B 与 simplex | [optimization/triangle_environment.jl](optimization/triangle_environment.jl)、[optimization/full_update.jl](optimization/full_update.jl) |
| 梯度优化：可微 CTMRG、梯度与线搜索 | [optimization/differentiable.jl](optimization/differentiable.jl)、[optimization/variational.jl](optimization/variational.jl) |
| GFPEPS 导入、单占据投影、独立测量自旋能量和磁化 | [gfpeps/import_ipess.jl](gfpeps/import_ipess.jl)、[gfpeps/projection/](gfpeps/projection/)、[gfpeps/observables/](gfpeps/observables/) |
| Z2 iPESS 态与费米符号、Hubbard 模型与测量 | [state.jl](state.jl)、[tensor_utils.jl](tensor_utils.jl)、[model.jl](model.jl)、[observables.jl](observables.jl) |
| 共享 CTMRG 适配、Hubbard 运行函数 | [ctmrg.jl](ctmrg.jl)、[workflows/](workflows/) |

D=8 parton 的算例选择、校验和 SU 调用在[任务入口 task.jl](../../../Kagome/fermionic/jobs/parton_D8_su/task.jl)。[parton_su_io.jl](workflows/parton_su_io.jl) 提供读取初态、校验测量和写对比表的函数。GFPEPS 固定态测量函数在 [measurements.jl](gfpeps/workflows/measurements.jl)。

投影态的 [D=8 验证任务](../../../Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/run_projected_D8.jl)和[存档能量比较](../../../Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/summarize_projected_runs.jl)放在 `benchmarks/`。两份海森堡报告共用 [spin_tables.jl](gfpeps/reporting/spin_tables.jl) 中的制表函数。

## 在服务器上运行

从 `SU2_AD/` 目录运行。保留完整 `src/`，因为 `ctmrg.jl` 还会加载 `src/bosonic/` 和 `src/fermionic/` 的共享实现。`Kagome/fermionic/iPESS/` 存放启动脚本、配置及数据。若从 Git 检出，另行复制被忽略的 D=4 初态到 `Kagome/fermionic/iPESS/results/gfpeps_validation/ipess_D4/tensors.mat`，或在配置中指定自己的输入。

```sh
APP=Kagome/fermionic/iPESS
bash "$APP/server/run.sh" setup
bash "$APP/server/run.sh" spin su "$APP/spin/D4.toml"
```

将 `su` 换成 `fu` 或 `ad` 可从同一输入态独立运行；三种方法分别写入 `job.output/{su,fu,ad}/`。若要从 SU 续接 FU，见[自旋运行说明](../../../Kagome/fermionic/iPESS/spin/README.md)。

也可以在 `SU2_AD/` 下启动 Julia，从脚本或 REPL 加载：

```julia
include("src/fermionic/kagome/spin/bootstrap.jl")
using .KagomeSpinOptimization
run_spin_optimization("su", "Kagome/fermionic/iPESS/spin/D4.toml")
```

仅做固定 GFPEPS 态的 CTMRG 测量时，使用 [gfpeps/bootstrap.jl](gfpeps/bootstrap.jl)；规范见 [gfpeps/CONVENTIONS.md](gfpeps/CONVENTIONS.md)。`gfpeps/ctm/`、`diagnostics/`、`reporting/` 和 `workflows/` 分别放置环境处理、扇区检查、结果汇总和固定态测量流程。`gfpeps/python/` 只用于转换外部 Gaussian 存档，读取现成的 MAT 初态和三种优化都用 Julia。共享 CTMRG 有模块级设置，同一进程内请串行运行。

原生三格点合并 iPEPS 的 simple update（含三个自旋分辨 parton 模型、Hubbard 和 Heisenberg + 纵场）见 [ipeps/README.md](ipeps/README.md)。该 Julia 入口也可选择原生 iPESS。
