# 服务器批量运行

将完整仓库传到服务器，沿用旧自旋计算使用的 Julia 包环境。从仓库根目录运行：

```sh
sh SU4/simple_update_dense/job_su4_D4_D16.sh
```

脚本直接调用 `julia --startup-file=no ...`，没有 nohup。普通 shell 下后台启动，
输出 PID 和日志位置。在 SLURM/PBS/LSF 分配的作业内自动前台运行。
需要普通 shell 前台运行时设置 `SU4_FOREGROUND=1`。
此脚本本身不申请集群资源；资源由服务器调度脚本/已分配作业负责。

可选环境变量：

```sh
export JULIA_BIN=/path/to/julia
export JULIA_PROJECT=/path/to/your/existing/environment
export SU4_BLAS_THREADS=16
export SU4_OUTPUT_DIR=/scratch/your_user/su4_tensormap
sh SU4/simple_update_dense/job_su4_D4_D16.sh
```

## 默认计算

- 2×2 cell，J=1，只有 J 项，能量规范 `P_ij-I/4`。
- D=4,5,...,16；每个 D 独立随机初始化，seeds=1234,5678,9012。
- 演化直接调用 src 的 TensorMap iPESS simple update。
- 时间步与每阶段最大 sweep 数：
  `(0.1,1000), (0.05,1000), (0.02,1500), (0.01,1500), (0.005,2000),
  (0.002,2000), (0.001,2500), (0.0002,3000), (0.0001,3000)`。
- 每阶段至少 100 sweeps；每 20 sweeps 检查三组 lambda 的变化，
  使用旧 `check_convergence`。区间内最大变化连续 4 次低于
  `max(1e-8,1e-4*dt)` 时进入下一阶段；否则达到上限后继续并记录未收敛。
- 每 10 sweeps 及检查/阶段结束时保存检查点。
- SU 结束后必须调用旧 `CTMRG_cell` 和 `evaluate_ob_cell`，
  chi=16,32,48，每个 chi 最多 50 次。没有 lambda 环境的近似能量输出。
- CTMRG 沿用旧代码的角谱收敛判据，容差 1e-7；最后两个 chi 均达到此判据且
  所有键的最大变化 <1e-5 才标记 chi 收敛。旧 CTMRG 内部也可能提前停止。
- `converged` 要求最终 SU 阶段和 CTMRG 检查同时通过。
  固定 sweep/chi 上限不保证收敛，须检查输出标记；高 D 尤其需要检查 chi 依赖。

可在 `src/bosonic/triangle/su4_dense/LongRun.jl` 修改 `LongRunConfig` 的默认参数，
或在 `LongRun_D4_D16.jl` 入口构造配置时指定参数。无本地内存预算硬限制，实际内存需求
随 D、chi 增长。2×2 允许当前考虑的候选，但不会自动排除更大周期的状态。

## 输出与续算

默认目录 `SU4/simple_update_dense/long_runs_tensormap`。
每个 D/seed 独立目录包含：

- `checkpoint.jls`：TensorMap 状态、lambda、进度及 SU 历史。
- `result.mat`：CTMRG 能量、12 条键、连通关联、方向/空间调制、Fourier 分量、
  CTMRG chi 历史、SU 收敛历史、参数和收敛标记。默认同时导出 TensorMap 的数组数据。
- `observables*.csv/.txt/.jls`：完整键表及易读摘要。
- `run.log`：该初态的 SU 进度。

总目录有 `scan_summary.mat/.csv`：status=0 待计算、1 收敛、2 完成但未收敛、-1 失败。
某任务失败保留检查点并继续其余任务，全部结束后若存在失败则以非零状态退出。

同一输出目录重启会检查参数并续算。SU 已完成时只重新测量 CTMRG；
可修改 chi/CTMRG 容差重新测量。科学参数或 SU 日程不同需使用新输出目录。
不要同时往同一输出目录启动两个任务。
旧 Array 实现的数据保留供对照，不自动转换或混入此次 TensorMap 扫描。
