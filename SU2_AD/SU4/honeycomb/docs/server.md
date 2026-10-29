# 服务器：从 D=9 星形态出发，允许增长到 Dmax=24

服务器入口 `server/run_star_simple_update.jl` 不再 include 本地测量/审查脚本。
复用的星形初态已移入 `src/bosonic/honeycomb/star/honeycomb_star_state.jl`，
由 `initial_honeycomb_star_iPEPS()` 返回 T、单位初始 lambda 和信息；
原 `initial_star_RVB()` 保留兼容入口。调度、检查点和测量位于
`src/bosonic/honeycomb/star/StarSimpleUpdateRun.jl`，两点更新仍调用原 tebd_x/ybond。
没有改动 CTMRG 核心。

**初始 D 固定为9，配置中的 Dmax 是更新上限。** 默认4×4 cell，三方向独立，
严格 SU(4)，不强迫每个键都达到24，也不强迫点群对称。完整 multiplet 截断
使用 `WholeMultipletBudget`；虚拟量子数与 multiplicity 由 SVD 更新。
默认不做恒等门预整理。所有局部/CTM 能量 `E_P` 采用 `sum(P_ij)` 每格点单位，
实际 Hamiltonian 为 `J*sum(P_ij)`；CTM 另记录 `E_H=J*E_P`。

将**完整 SU2_AD 目录**放到服务器，使用现有兼容的 Julia/TensorKit/SUNRepresentations
环境；不依赖本机生成的 JLD2 初态或 Windows 路径。脚本不安装或更新任何包。
本机验证环境为 Julia1.11.6、TensorKit0.17 与现有 SUNRepresentations；其他后端版本
仍需先跑 pilot。`JULIA_BIN`、`JULIA_PROJECT` 可选择服务器可执行文件和环境。

从 SU2_AD 目录运行：

```sh
# 可选：指向已有环境，不要照抄本机 Windows 路径。
export JULIA_BIN=/path/to/julia
export JULIA_PROJECT=/path/to/existing/environment

# 只解析配置，不载入 TensorKit、不生成 CGC、不演化。
sh SU4/honeycomb/run_star_su.sh --dry-run

# 先在服务器试三轮 Dmax=24 更新，不做能量/CTMRG测量。
sh SU4/honeycomb/run_star_su.sh --config SU4/honeycomb/config/StarSimpleUpdate_pilot.toml

# 一次 sh 后台启动：SU 总 tau=11，随后自动 CTMRG 计算能量和全部真实键观测量。
sh SU4/honeycomb/run_star_su.sh

# 中断后重复同一命令，自动从最后有效检查点续跑。
sh SU4/honeycomb/run_star_su.sh

```

启动方式沿用现有服务器 sh：脚本内部用 `nohup setsid ... &` 后台启动一个 Julia
进程，自动生成带时间戳的 `.out/.err/.pid` 并打印位置。无需手动添加 nohup。
日志默认在 `SU4/honeycomb/logs/`，可用 `SU4_STAR_LOG_DIR` 指定；
`JULIA_NUM_THREADS` 默认1，`SU4_STAR_BLAS_THREADS` 可设置计算线程数并同步 OMP/MKL。
`--help`、`--dry-run` 保持前台输出；需要前台运行时用
`sh SU4/honeycomb/run_star_su.sh --foreground`。
普通 sh 自动传递 `--auto-resume`，无检查点时创建新任务，有检查点时续跑；
仍不能同时运行两个写同一输出目录的任务。若环境没有 setsid，可使用前台方式。

也保留可选的 Slurm 提交方式（不要求使用）：

```sh
sbatch SU4/honeycomb/server/job_star_su.sbatch
sbatch SU4/honeycomb/server/job_star_su.sbatch --resume
```

SBATCH 中CPU、内存、时限只是可编辑模板，D=24完整计算的资源尚未在本机测定；
partition/account 按自己的集群添加。调度器作业内自动保持前台，退出状态传给调度器。
从其他位置提交时设置 `SU4_STAR_DIR=/absolute/path/to/SU2_AD/SU4/honeycomb`。

`config/StarSimpleUpdate.toml`：

- `[[schedule]]` 的 `steps,dt` 定义固定演化段，可在某一段另设 `Dmax`。
  完成计划不等于达到物理收敛；检查谱变化、时间步和 CTMRG 能量。
- `checkpoint_every` 控制保存间隔；正式演化第一轮结束就先保存一次。
  `keep_every` 为额外存档间隔，0关闭周期存档，段末仍存档。
- `local_energy_every` 是近似 lambda 能量测量间隔，0关闭；未测量行的 NaN
  表示“没有测量”，不是张量含NaN。它不能替代 CTMRG。
- `blas_threads` 可被 `SU4_STAR_BLAS_THREADS` 覆盖；CTMRG 用单独线程设置。
- `max_seconds` 在检查点边界干净退出，0不设限；不保证单个块在时限内完成。
- `ctm.enabled=true` 为正式配置默认值：同一入口、同一个作业内，SU 完成后自动
  执行 CTMRG。`chis`、迭代次数、角谱容差和能量容差分别可调。
  只有显式的轻量 pilot 配置关闭测量；正式任务无需第二次提交。

输出默认位于 `SU4/honeycomb/server_runs/star_D24/`（模板配置写作 `../server_runs/star_D24`，以保持整理前的位置）；`--output /scratch/...` 可覆盖。
相对输出路径按**配置文件所在目录**解析。每个检查点先写同目录临时文件，完成后
替换 `state_latest.jld2`，保存 T、lambda、累计虚时、已完成步数、当前演化段和历史。
续跑恢复三方向更新的循环位置，避免检查点分块改变 Trotter 顺序。
异常时不保存部分更新的内存张量，只保留最近一次有效磁盘检查点，退出非零状态。

`evolution.csv` 为日志，`bond_sectors_latest.csv` 记录各键 dynkin label、multiplicity
和 simple 环境权重；重启时由检查点重建日志，避免重复追加。`status.toml` 可直接读。
`state_step*.jld2` 是阶段/周期存档。中途被强制 kill 可能留下临时文件及
`.star_su_lock/`：确认旧任务已停止后，删除锁目录再 `--resume`；不要删除有效的
`state_latest.jld2`。正常退出会自动释放锁。

普通 `--resume` 要求物理参数和 schedule 不变。如果需要提高 Dmax 或换一套计划，
使用**新输出目录**和新配置，从旧服务器检查点继续；累计 tau 和方向循环不会清零：

```sh
sh SU4/honeycomb/run_star_su.sh --config /path/to/new_plan.toml \
  --output /scratch/new_run \
  --init-checkpoint /scratch/old_run/state_latest.jld2
```

CTMRG 只复用同一个测量张量/规范生成且通过检查的环境，不直接拿旧初态环境热启动。
演化和测量函数都在 `StarSimpleUpdateRun.jl`，由同一个入口顺序执行。
若中断发生在 CTMRG 阶段，`--resume` 会跳过已完成的 SU，再执行测量；
`--measure-only` 仅保留为调试选项，不是正式运行所需的第二个任务。
输出包含全部真实键的 `ctm_bonds_*.csv`、CTM、源检查点快照及 chi 对照表。
单键谱界、角谱收敛和相邻 chi 的能量变化分别记录；`chi_stable` 只检查能量，
不证明不同边界扇区的状态相同。所有 chi 都未通过时返回状态码2，保留诊断文件，
不把它当作优化成功。D=9严格星形态此前的边界敏感问题仍应留意。

本机轻量检查：配置/启动器、首个两点门 D9→D24 的无截断重构、真实SU的检查点
续跑与不中断更新对照。没有在本机执行上述完整 Dmax=24、tau=11 计划。

```sh
julia --startup-file=no SU4/honeycomb/tests/test_star_server_config.jl
julia --startup-file=no SU4/honeycomb/tests/test_star_server_runtime.jl
julia --startup-file=no SU4/honeycomb/tests/test_star_server_branch.jl
```
