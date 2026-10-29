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
- D=4,5,...,16，seeds=1234,5678,9012；每个 seed 独立形成一条升 D 路线。
- 默认 `initialization=:previous_D`：第一个 D 随机初始化；如果同一输出根目录里
  已有该 D/seed 的独立计算检查点，则直接复制到续接目录并恢复其进度，不重算已完成的 SU。
  后续 D 从前一个已完成 SU 日程的 D/seed 检查点扩维初始化，不再完全随机启动。
- 扩维保留旧张量块，仅在新增虚空间方向加入相对幅度 `expansion_noise=1e-3`
  的扰动，并给新增 lambda 分量小的非零权重。原检查点不修改。
- 演化直接调用 src 的 TensorMap iPESS simple update。
- 第一个 D 的时间步与每阶段最大 sweep 数：
  `(0.1,1000), (0.05,1000), (0.02,1500), (0.01,1500), (0.005,2000),
  (0.002,2000), (0.001,2500), (0.0002,3000), (0.0001,3000)`。
- 后续 D 使用较短的 `continuation_schedule`：
  `(0.005,500), (0.002,800), (0.001,1000), (0.0002,1200), (0.0001,1500)`。
  这些是上限，满足收敛判据会提前进入下一阶段；同样不保证每个 D 都能达到收敛阈值。
- 每阶段至少 100 sweeps；每 20 sweeps 检查三组 lambda 的变化，
  使用旧 `check_convergence`。区间内最大变化连续 4 次低于
  `max(1e-8,1e-3*dt)` 时进入下一阶段；此阈值沿用旧 `itebd_iPESS` 的 `dt*1e-3`
  （当前默认步长下均大于下限）。否则达到上限后继续并记录未收敛。
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
如需原来的独立随机计算，构造 `LongRunConfig(initialization=:random, ...)`。
升 D 路线要求 `Ds` 严格递增；若跳过中间 D，则从列表中的前一个 D 扩维。
例如已经完成 D=5，可设 `Ds=collect(5:16)`，复用已有 D=5，再从 D=6 开始扩维续接。
续接可能延续前一个 D 的局部极小值，可使用不同 seed 或独立随机计算比较能量。

## 输出与续算

默认目录 `SU4/simple_update_dense/long_runs_tensormap`。
默认续接结果放在该根目录的 `continuation/` 子目录内；原独立随机结果仍位于根目录。
如果服务器此前设置了 `SU4_OUTPUT_DIR`，请继续指定相同的根目录，以自动找到已有 D=4 检查点。
每个 D/seed 独立目录包含：

- `checkpoint.jls`：TensorMap 状态、lambda、进度及 SU 历史。
- `result.mat`：CTMRG 能量、12 条键、连通关联、方向/空间调制、Fourier 分量、
  CTMRG chi 历史、SU 收敛历史、参数和收敛标记。默认同时导出 TensorMap 的数组数据。
  同时记录 `initialization`、`source_D`、`source_checkpoint`、`expansion_noise` 和实际使用的日程。
- `observables*.csv/.txt/.jls`：完整键表及易读摘要。
- `run.log`：该初态的 SU 进度。

续接目录（独立随机模式为根目录）有 `scan_summary.mat/.csv`：
status=0 待计算、1 收敛、2 完成但未收敛、-1 失败。
某任务失败保留检查点并继续其余任务，全部结束后若存在失败则以非零状态退出。

同一输出目录重启会检查参数并续算。SU 已完成时只重新测量 CTMRG；
可修改 chi/CTMRG 容差重新测量。科学参数或 SU 日程不同需使用新输出目录。
续接的每个 D 自己保存完整状态、进度和来源；恢复该 D 的检查点时不会重新扩维或重新加扰动。
若前一个 D 的 SU 尚未完成日程，后一个 D 会报错，不会悄悄改成随机初始化。
此前默认 SU 阈值 `dt*1e-4` 的检查点可直接续算：保留已演化的张量和历史，
按恢复后的 `dt*1e-3` 阈值重新检查阶段收敛标记；已完成的 SU 不会从头重跑。
不要同时往同一输出目录启动两个任务。
旧 Array 实现的数据保留供对照，不自动转换或混入此次 TensorMap 扫描。
