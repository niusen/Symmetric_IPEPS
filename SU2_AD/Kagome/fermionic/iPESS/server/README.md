# Linux 终端运行

计算函数在 `src/fermionic/kagome/`；本目录的 `run.sh` 从源码目录启动 Julia。
脚本使用服务器当前的 Julia 环境。本机验证版本为 Julia **1.11.6** 和 TensorKit **0.17.0**。
如果依赖装在其他 Julia 项目中，可在提交前设置 `JULIA_PROJECT` 为该项目的绝对路径。

## 准备源码与初态

在服务器保留 `SU2_AD/src/` 和 `SU2_AD/Kagome/fermionic/iPESS/` 的相对位置。初态是数据文件，Git 默认忽略 `results/`。只需传这次计算的输入：

| 任务 | 配置默认读取的输入（相对于 `Kagome/fermionic/iPESS/`） |
|---|---|
| 投影 D=4 GF-iPESS 的自旋 SU/FU/AD，或该态的固定态 Heisenberg 测量 | `results/gfpeps_validation/ipess_D4/tensors.mat` |
| 其他 D 或 iPEPS 固定态测量 | 对应 `results/gfpeps_validation/<态名>/tensors.mat` |
| 未投影 Gaussian 能量对照 | 同一态目录的 `tensors.mat` 和 `reference_N192.mat` |

在服务器的 `SU2_AD/` 目录检查已有依赖和 D=4 固定态测量的输入。

```sh
APP=Kagome/fermionic/iPESS
bash "$APP/server/run.sh" setup
bash "$APP/server/run.sh" gfpeps "$APP/gfpeps/configs/heisenberg_ipess_D4.toml" --check
bash "$APP/server/run.sh" gfpeps "$APP/gfpeps/configs/heisenberg_ipess_D4.toml"
```

`setup` 加载所需包，并打印 Julia、TensorKit 版本和当前项目路径。
可通过 `JULIA_BIN=/path/to/julia` 指定可执行文件。
脚本路径和配置路径可以包含空格，也可从其他工作目录使用绝对路径启动。

## 计算内容与配置

`gfpeps/configs/` 包含 D=4、8 的 iPESS/iPEPS 配置，以及额外的 D=8 seed12103。
`heisenberg_*` 对输入态作 Gutzwiller 投影，再测量 J=1、h=0 的 Heisenberg
能量和磁化；`gaussian_*` 保持未投影，与 `reference_N192.mat` 比较。
这些都是固定态测量，不重新优化波函数。
所附配置是从小 χ 开始的服务器运行起点；历史报告中的某些数值使用不同的
χ、更新顺序或续算环境。精确复现旧记录时，应采用对应结果 TOML 中的参数。

配置中的路径相对于 TOML 文件，而不是终端当前目录。输出目录必须为空；
再次运行时请修改 `job.output`，以保留已有结果。每个 χ 默认独立从 PBC 初始化。
如需续算，显式设置 `job.warmstart` 为旧 `.jld2` 或 `_environment.jld2` 路径；
必须使用相同源态、投影方式、规范平衡参数和虚空间约定。
中断后的 `_environment.jld2` 可以作为新输出目录下任务的初始环境。
CTMRG 未收敛的能量仍会记录，需比较 χ 和迭代次数后再判断精度。

计算线程分别由 `JULIA_NUM_THREADS` 和 TOML 的 `job.blas_threads` 控制。
D=8 的内存需求明显更高，先运行 D=4，再逐步增加 χ；紧凑等价双层默认开启。
同一进程内串行计算；如同时启动多个进程，应自行分配内存和 CPU。

## SU、FU 和梯度优化

原生 iPEPS 的 SU（以及三个自旋分辨 parton 模型的 iPESS SU）用 [Julia 运行入口](../../iPEPS/SimpleUpdate.jl)。参数直接写在脚本中，也可运行 `bash "$APP/server/run.sh" ipeps-su zero "$APP/results/gfpeps_validation/ipeps_D8_zero/tensors.mat" runs/ipeps_D8_zero`。算法、初态与续算见[源码说明](../../../../src/fermionic/kagome/ipeps/README.md)。

物理腿为纯自旋、以投影 GF-iPESS 为初态时，请使用 [自旋模型运行入口](../spin/README.md)
和 `spin su|fu|ad` 子命令。下面的例子用于 Hubbard 物理空间。

原有配置与入口继续有效：

```sh
bash "$APP/server/run.sh" su "$APP/config.toml"
bash "$APP/server/run.sh" fu "$APP/optimization/full_update.toml"
bash "$APP/server/run.sh" ad "$APP/optimization/variational.toml"
```

这些入口的实现已迁入 `src/fermionic/kagome/workflows/`，算法位于同级源码及
`optimization/`；外面的 `.jl` 仅保留加载与命令行调用。

## 验证与历史资料

GFPEPS 测试在 `benchmarks/gfpeps_validation/tests/`，报告在 `reports/`，
本机历史日志在 `iPESS/results/gfpeps_validation/logs/`，重新生成的测量报告在同级 `reports/`。旧的 `run.jl`、`run_heisenberg.jl` 等常用入口保留。
其中 `test_measurement.jl` 等历史对照需要原来的 CTM 结果文件，完整扇区存档
审计需要 `charge_audit_input.mat`。要运行这些诊断，需另行提供对应的历史数据。
常规投影、导入和新运行流程的测试用相应态的输入文件，具体命令见验证记录。
重新从外部 Gaussian 工作区导出输入时，才需要 Python/NumPy/SciPy，并设置
`KAGOME_GAUSSIAN_SOURCE` 为原始 kagome Gaussian 工作区；服务器直接读取 MAT
不依赖原来 Windows 上的笔记目录。

本次整理的实际检查与平台限制见 [VALIDATION.md](VALIDATION.md)。
