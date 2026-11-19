# 服务器：从 D=9 星形态出发，允许增长到 Dmax=24

服务器入口 `server/run_star_simple_update.jl` 不再 include 本地测量/审查脚本。
复用的星形初态已移入 `src/bosonic/honeycomb/star/honeycomb_star_state.jl`，
由 `initial_honeycomb_star_iPEPS()` 返回 T、单位初始 lambda 和信息；
原 `initial_star_RVB()` 保留兼容入口。调度、检查点和测量位于
`src/bosonic/honeycomb/star/StarSimpleUpdateRun.jl`。默认局部更新位于
`src/bosonic/honeycomb/honeycomb_StarSimpleUpdate.jl`，旧两点更新保留为显式对照。
没有改动 CTMRG 核心。

**初始 D 固定为9，配置中的 Dmax 是更新上限。** 默认4×4 cell，四站点star
HOSVD；三个模式从同一张量求投影，再共同构造核心，和kagome普通hosvd一致。
不做核心二次HOSVD，不强迫每个键都达到24，也不强制三个投影矩阵相同。
完整 multiplet 截断
使用旧 simple update 的 TensorKit `truncdim(Dmax)`；虚拟量子数与 multiplicity 由 SVD 更新。
默认不做恒等门预整理。所有局部/CTM 能量 `E_P` 采用 `sum(P_ij)` 每格点单位，
实际 Hamiltonian 为 `J*sum(P_ij)`；CTM 另记录 `E_H=J*E_P`。

只复用已有的一份通用 `src/`，不导出或复制算法源码。
运行目录可以放在任意位置，不要求 `SU2_AD/SU4/honeycomb` 层级。
程序自动沿入口和当前目录的父目录查找 src，也支持
`SU4_SRC_DIR=/absolute/path/to/src` 或 `SU2_AD_ROOT=/absolute/project/root`。
显式指定的路径若无效会直接报出该路径，避免意外载入另一份源码。
使用现有兼容的 Julia/TensorKit/SUNRepresentations 环境；不依赖本机生成的
JLD2 初态或 Windows 路径。脚本不安装或更新任何包。
本机验证环境为 Julia1.11.6、TensorKit0.17 与现有 SUNRepresentations；其他后端版本
仍需先跑 pilot。`JULIA_BIN`、`JULIA_PROJECT` 可选择服务器可执行文件和环境。

原仓库布局下可从 SU2_AD 目录运行以下命令；独立运行目录直接用 `sh run_star_su.sh`：

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

输出默认位于 `SU4/honeycomb/server_runs/star_hosvd_D24/`（模板配置写作 `../server_runs/star_hosvd_D24`）；`--output /scratch/...` 可覆盖。
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

更新默认使用旧 simple update 的 TensorKit `truncdim(Dmax)` 维度截断，
不再默认调用自定义 `WholeMultipletBudget`。检查点记录截断策略；旧版自定义截断
产生的检查点不能直接 `--resume` 混接为同一次演化。对照测试应从初态和新输出目录
开始；如需接续旧张量，显式使用上面的 `--init-checkpoint` 和新输出目录。

### 默认 star HOSVD 方法及权重约定

`[run] method="star_hosvd"` 是默认值，`hosvd_tol=1e-10` 控制模式奇异谱的
相对小值清理。`Dmax=9` 也可运行，初态始终是原来的D=9张量，不预先填充到24。
HOSVD严格采用kagome普通`hosvd`的流程：先归一化cluster，分别从同一个cluster
做三次展开SVD，再共同投影得到核心并归一化；不加核心二次HOSVD。
旧的两站点更新仅在显式 `method="two_site"` 时使用。

每个cluster含中心及三个邻居。外围site先吸收两条外侧lambda，再做仅删除精确零
奇异值的外腿支持压缩；物理gate与三个HOSVD作用于保留块，最后乘回残余张量，
在原来的外腿上去除所吸收的lambda。内部三条旧lambda不重复插入。

四组互不重叠star覆盖组成一个二阶Trotter步，层顺序为a/2,b/2,c/2,d,c/2,b/2,a/2；
每轮循环首组，并通过检查点的累计步数延续。每条真实键出现在两个star中，
因此四站点gate使用 `h_star=J/2 sum_i(P_0i-I/4)`，permutation约定则不减I/4。

与旧kagome一样，投影后的核心G已含模式奇异值权重。为了与现有site张量和CTMRG
兼容，输出做精确的配对规范变换：每个外围投影乘 `R_i=sqrt(S_i/norm(S_i))`，
核心对应腿乘 `pinv(R_i)`，权重数组存R_i；物理PEPS直接收缩输出site张量，
不额外插入lambda。这里的S_i是用于HOSVD的原始模式谱，不声称是联合截断后
重新精确计算的谱。核心保持三方向关联，不能再额外乘三组S。

默认输出为 `star_hosvd_D24`，以免自动续跑旧两站点结果。更新方法、HOSVD容差
与权重约定都写入检查点；旧方法需以新输出目录显式作为初始检查点导入。

```sh
julia --startup-file=no SU4/honeycomb/tests/test_star_hosvd.jl
```

CTMRG 只复用同一个测量张量/规范生成且通过检查的环境，不直接拿旧初态环境热启动。
演化和测量函数都在 `StarSimpleUpdateRun.jl`，由同一个入口顺序执行。
若中断发生在 CTMRG 阶段，`--resume` 会跳过已完成的 SU，再执行测量；
`--measure-only` 仅保留为调试选项，不是正式运行所需的第二个任务。
输出包含全部真实键的 `ctm_bonds_*.csv`、CTM、源检查点快照及 chi 对照表。
单键谱界、角谱收敛和相邻 chi 的能量变化分别记录；`chi_stable` 只检查能量，
不证明不同边界扇区的状态相同。所有 chi 都未通过时返回状态码2，保留诊断文件，
不把它当作优化成功。D=9严格星形态此前的边界敏感问题仍应留意。

本机检查包括星形覆盖、同一张量三次展开、C3协变性、dense与SU4张量的
重构，以及非均匀lambda的吸收/移除，共32项通过；配置检查30项通过。
启动器和独立运行目录复用共享src的检查也通过。旧两点方法的重构及续跑
检查保留为历史回归。没有在本机执行上述完整 Dmax=24、tau=11 计划。

2026-10-29 新方法试跑：均从严格D=9初态开始，4×4、J=1，65轮dt=0.01，
累计tau=0.65，使用TensorKit原生维度截断和hosvd_tol=1e-10。

| Dmax | 演化结果 | 最终实际键维数 | CTMRG能量E_P/site |
| --- | --- | --- | --- |
| 9 | 完成65轮 | 1或4 | -0.5624999999999722，chi=16和24一致 |
| 24 | 完成65轮，未重现旧方法第60轮的零张量 | 16至24 | 本机测量主动中断，未得到结果 |

Dmax=9的结果退化为四站点SU4 singlet的乘积态：16站点cell中12条键P=-1，
另12条键P=1/4，故解析能量为-9/16。它不是保留完整共振叠加的优化结果。
Dmax=24末步谱变化约7.02e-4；越过旧报错位置仅说明此次短程演化通过，
不能据此宣称长时间稳定或能量收敛。测试日志位于本机忽略目录
`logs/tests/star_hosvd_D{9,24}_evolution/`，不需要上传服务器。
Dmax=24的额外CTMRG在SU4 CG系数生成阶段耗时较长，故主动中断；
日志中的InterruptException不是simple update的数值失败。完整SU检查点保留。

2026-10-30补查：从原始D=9星形叠加态重新演化，Dmax=9、dt=0.01，
在初态及每个完整扫掠后独立PBC初始化CTMRG，chi=16、24，各最多100次迭代，
角谱容差1e-7，BLAS单线程。前10轮的有效测量均为E_P/site=-0.5624999999999722，
未见可分辨的升降。初态chi=24、第一轮chi=16未收敛且单键越界，明确排除；
初态采用有效的chi=16，第一轮采用有效的chi=24。其他各步两个chi均通过。
所以初态到第一轮不是同一个chi下的双重收敛比较，仍有已知边界扇区限制。

实际键维数在第二轮tau=0.02已从9降到1或4；第2至10轮的CTMRG均2次迭代收敛。
初态chi=16首次测量224.2秒；另一进程初态chi=24首次测量308.1秒且无效，
两者均含编译/系数生成开销。第一轮chi=24有效测量20.05秒、25次迭代；
第一轮chi=16无效测量71.31秒、100次迭代。乘积态第3至10轮单次测量
0.22至0.60秒。这些短耗时不代表一般完整D=9 RVB或D=24的测量成本。
计时包含CTM初始化、迭代和24条键的能量测量，不含状态保存和SU演化。

诊断入口为`experiments/star/Trace_D9_CTM.jl`，参数为新输出目录和扫掠次数；
可加`--resume`复用已保存的逐点测量。原始表、逐步张量及CTM保存在本机
`logs/tests/D9_CTM_by_sweep/`。能量持平不能证明原始共振态未变；
初态CTM选取覆盖扇区，而第二轮后张量本身已变为乘积态。

### Dmax=9降维的逐次表示追踪（2026-10-30）

入口`experiments/star/Trace_D9_HOSVD.jl`从同一原始态运行3轮；功能函数在共享
`src/bosonic/honeycomb/star/StarHOSVDTrace.jl`。记录84次star更新、252次内部键
模式截断，并记录每个star的3条内部键及6条外部键在更新前后的表示、lambda
对角元和两端张量的表示范数占比。额外完整SVD只用于观察；每轮与关闭审计的
同起点演化重放比较，三个重放最大相对误差均严格为0。

首次降维发生在第41次star更新：第2轮、第4层、中心(1,1)，模式1对应
真实键(1,1)--(4,1)。完整归一化模式谱按奇异值排序为：

| irrep / Dynkin label | 维数 | 奇异值 |
| --- | --- | --- |
| 1 / (0,0,0) | 1 | 0.999986489125005 |
| 15 / (1,0,1) | 15 | 0.00124736804489457 |
| 4 / (1,0,0) | 4 | 0.000743654325565550 |
| 4bar / (0,0,1) | 4 | 0.000606336291801200 |
| (1,1,0) | 20 | 4.37387272050e-9 |

本机TensorKit原生维度策略按奇异值排序保留前缀：保留1之后，下一项15使
累计维数超过9，便停止，不会跳过15再考虑后面的4、4bar。因此**维度截断
本身就已把9降成1**；并不是4和4bar小于1e-10阈值。该模式丢弃范数平方
比例约2.70216e-5，包含仍非零的4及4bar。后续tol清理没有进一步改变它。
事实上本次全部252个模式的tol清理保留范数比例都为1、维数没有额外下降。

同一模式施门前数值上只有1+4+4bar非零谱；施门生成的15通道在这里超过了4和4bar，
触发上述排序前缀行为。此前多个更新中模式权重已经逐渐偏向单一表示，
这里定位的是首次降维的直接机制，不是对lambda规范/环境近似正确性的完整证明。

| 阶段结束后 | D=1键数 | D=4键数 | D=9键数 |
| --- | --- | --- | --- |
| 第1轮、以及第2轮前三层 | 0 | 0 | 24 |
| 第2轮第4层 | 8 | 0 | 16 |
| 第2轮第5层 | 12 | 4 | 8 |
| 第2轮第6层 | 12 | 12 | 0 |
| 第2轮末至第3轮末 | 12 | 12 | 0 |

本机结果在`logs/tests/D9_HOSVD_bond_audit/`：`transitions.csv`记录输入空间、
完整模式空间、dim截断、tol清理和输出空间；`spectra.csv`保存上述各阶段
每份multiplet的全部对角元；`bonds.csv`保存输入/输出lambda及两端局部表示
支持；`layer_summary.csv`汇总全cell，`first_dimension_drop.jld2`保存关键更新
前后完整张量。谱份额按dim(irrep)*s^2计算。lambda份额和两端局部范数均依赖
规范，不作为全局Schmidt概率。真实截断方式保持原生truncdim，没有改为固定表示。

原始RVB的singlet权重核对：当前规范的单格点张量是四个互相正交、等范数
通道之和：一个(4,4,4)中心通道及三个循环的(4bar,1,1)外围通道。
因此在任一真实虚拟腿上，仅按单张量Frobenius范数计算，
1、4、4bar的扇区份额为1/2、1/4、1/4（按该腿方向可能交换4与4bar）。
初始lambda则为单位阵；若只统计lambda的dim(r)*lambda_r^2份额，得到
1/9、4/9、4/9。它们都不是无限PEPS的全局Schmidt权重。

第一次施门前的四站点模式谱已经不同：1、4、4bar分别为128/193、1/193、
64/193，约0.6632124、0.00518135、0.3316062。独立dense epsilon/delta
构造验证了这个结果：规范化单格点的外部迹给出每个磁分量权重
m_1=1/2、m_4=m_4bar=1/16；三个邻居消息使中心通道的未归一化贡献为
(1/4)*(1/16)^3=1/16384，每个外围通道为
(1/4)*(1/16)*(1/2)^2=64/16384。对指定内部键，两个外围通道给singlet，
故总份额为128:1:64。这验证了该有限star、单位外部权重下的局部收缩，
不验证它等于无限RVB环境下的权重。

第41次更新施门前，该模式singlet份额已经达到0.9999963717528982；
施门后反而降到0.9999729784325895，因为门生成了15通道。
所以99.997%的数值不能解释为原始RVB本就几乎全在1扇区，也不能把偏置全部
归因于这道门；它发生在此前的局部更新、规范及lambda环境反馈过程中。
独立dense核对保存在`initial_weights_dense_check.json`。

```sh
julia --startup-file=no SU4/honeycomb/tests/test_star_server_config.jl
julia --startup-file=no SU4/honeycomb/tests/test_star_server_runtime.jl
julia --startup-file=no SU4/honeycomb/tests/test_star_server_branch.jl
```

### 旧两站点方法的空 lambda 诊断（历史）

蜂窝模块的实数 bond 逆运算直接逐对称块计算，不再先 `convert(Array,lambda)`。
每个两点门之前检查相关 lambda，之后立即检查新 bond 和两个格点张量；
全局及检查点检查也包含 brick-wall 的平凡腿。空空间、零权重和非有限值会停止
更新，而不是给空集合的 maximum 补一个默认值继续演化。
`failure.log` 会记录绝对更新步数、方向、键坐标、Dmax、各邻接 bond 的表示与权重
概况，以及后端版本；最近的有效磁盘检查点保持不变。

2026-10-29 对照：严格 D=9 星形初态、4×4、J=1、dt=0.01、Dmax=24，
不做恒等门预整理，三方向按原顺序循环，BLAS 单线程。自定义截断在第61步
x 键 `(1,1)-(2,1)` 出现空谱；恢复 TensorKit 标准维度截断，从初态重新运行，
在第60步 z 键 `(1,1)-(1,4)` 输出 D=0。标准截断的最后有效磁盘检查点为
第41步、tau=0.41；计划的100步未完成。此次未执行能量测量或CTMRG。
从标准截断的第41步检查点重复追踪到首次空输出，截断前的两站点张量
无截断SVD范数严格为0、D=0。因此这一步不是非空谱被维度预算截光，
而是形成两站点张量时已经没有非零支持。
因此不能把更换截断策略当作此初态演化问题的完整修复。
测试后端为 Julia 1.11.6、TensorKit 0.17.0、SUNRepresentations 0.4.0。
可单独运行逆运算、空腿检查及默认更新与原两点函数的一致性检查：

```sh
julia --startup-file=no SU4/honeycomb/tests/test_honeycomb_bond_validation.jl
```

### 旧两站点方法的逐次追踪

```sh
julia --startup-file=no SU4/honeycomb/tests/diagnose_star_update.jl --trace /path/to/new_trace_folder 100
```

从同一4×4严格星形初态、Dmax=24、dt=0.01开始，用标准TensorKit维度截断。
最后一个参数是最多扫掠次数，输出目录必须是新目录。原两点函数仅增加可选的
`trace` 回调；关闭时不计算额外SVD，不改变归一化或截断。
功能函数位于共享src的 `honeycomb/star/StarUpdateTrace.jl`。

- `numbers.csv`：每次局部更新在输入、吸收lambda、辅助SVD、两站点收缩、截断、
  去除lambda、归一化后的范数、最小非零元素、最大元素、有限性及块结构。
- `spectrum.csv`：两次辅助SVD、两站点无截断SVD及保留谱的全部奇异值与irrep维数。
- `metrics.csv`：辅助SVD重构误差、保留范数比例、截断误差及各邻接键的支持重叠。
- `support.csv`：每次更新后，所有受影响键两端在各irrep中的相对范数。
- `first_lost_support.jld2`：首次出现无共同irrep支持的键时，保存此次更新前后状态。
- `first_zero_or_nonfinite.jld2`、`failure.jld2`：首次零/非有限中间量及最后失败上下文。

`update` 是两站点操作的连续编号，`sweep` 是完整扫掠编号，`layer` 区分五层
二阶Trotter门。每20次扫掠另保存完整状态；异常退出状态码为2。

本机逐次追踪结果（2026-10-29）：记录到第2361次两站点操作，重现第60次
扫掠的零张量。前2360次成功操作的两个输出site tensor均已归一化，范数
最大偏差为3.33e-16；辅助SVD相对重构误差最大为6.28e-16，没有NaN/Inf，
lambda最小非零值为0.0201，最大求逆元素为36.52。

| 两站点操作编号 | 扫掠/层 | 观测 |
| --- | --- | --- |
| 2337 | 59/3 | x键(1,1)-(2,1)首次降到D=20，只保留1+4+15；局部丢弃范数平方比例1.45e-5 |
| 2355 | 59/5 | 更新y键(2,1)-(3,1)后，邻接z键(2,2)-(2,1)首次没有共同irrep支持 |
| 2361 | 60/1 | z键(1,1)-(1,4)的成对张量范数严格为0；输入两个site tensor范数仍为1 |

第2337次完整谱中，按奇异值降序依次为4(0.00649258)、1(0.000414479)、
15(3.09195e-5)、20(1.04559e-5)、4bar(8.09845e-6)。本机TensorKit0.17
标准维度策略先占用4+1+15=20维，遇到装不下的20维multiplet便停止，
因此没有再保留后面的4bar。这里的4bar仍是非零通道。
第2355次操作本身保留了完整非零谱，其局部SVD几乎无损，不能把首次检测到的
邻接支持消失简单归因于这一次截断；这些记录定位了表示空间变化及后续支持丢失
的先后顺序，尚不等同于证明所有全局PEPS约束都在第2337次之前有效。
# Two-site full update

The two-site alternative uses `config/FullUpdate.toml` and the SAME existing
shared `src`. Run from SU2_AD:

```sh
sh SU4/honeycomb/run_full_update.sh --dry-run
sh SU4/honeycomb/run_full_update.sh
```

The launcher delegates background/log/PID handling to the existing SU launcher;
the Julia job evolves and measures energies together. Repeating the command
resumes `state_latest.jld2`. `--foreground` keeps it in the terminal.
`--config FILE`, `--output DIR`, `--resume`, `--init-checkpoint FILE`, and
`--measure-only` follow the existing CLI. An explicit `SU4_SRC_DIR` selects
your existing common src. The server/ folder and both sh files stay together;
there is no second src and no copied algorithm library.

Choose exactly one virtual-space policy in `[run]`:

| space_policy | initialization of the variational bond |
| --- | --- |
| `"tensorkit"` | native `truncdim(Dmax)`; representations may change or shrink |
| `"fixed_space"` (default) | leading singular vectors within EACH input irrep, keeping its original multiplicity; no dimension/relative-threshold deletion of these sectors |

Both policies optimize against the same UNTRUNCATED gated two-site target,
using the existing square FU environment contractions and alternating local
optimizer. Fixed space fixes quantum numbers, not tensor coefficients or
singular values. Dmax is an upper bound: setting it to24 under fixed_space
does not pad the original D9 state. To fix a larger space, initialize from a
checkpoint already carrying that space. The default original RVB does not
load the collapsed SU result. Dense initial states are available with
`initial="dense"` and `dense_D`; `initial="star"` is the SU4 RVB.

Only the real honeycomb horizontal bonds and alternating vertical bonds are
updated. Missing vertical bonds keep dimension1. Coordinates inside FU and
its saved `A_set` increase downward, like the square CTMRG; incoming SU
`T_set` checkpoints are converted once. No lambda factors enter FU.

A second-order sweep has five layers, rotating the three bond groups:
a/2,b/2,c,b/2,a/2. On4x4 this is40 two-site updates. As in the existing square
FU, CTMRG is rebuilt after EVERY bond. Its corner convergence is required;
the initial and per-sweep swap observables must stay within [-1,1], and local
fidelity outside [0,1] is rejected. These checks do not establish a unique
boundary sector for the original RVB. An unconverged environment stops the
job rather than being silently used for optimization.

Checkpoints replace the last valid state after each COMPLETE measured sweep.
If interrupted halfway through a sweep, resume repeats that sweep. Output:
`energy_by_sweep.csv`, `bonds_sweep*.csv`, `bond_updates.csv`, `status.toml`,
and `state_latest.jld2`. `failure.log` preserves the failure and the last
complete checkpoint. Parameter changes require a fresh output directory;
`--init-checkpoint` starts a new schedule with tau measured from that input.

Functional code is `honeycomb_FullUpdate.jl` plus `star/FullUpdateRun.jl` and
`star/FullRunConfig.jl`. The existing square `full_update_J1_cell.jl` adds an
optional bond initializer and common density normalization; their defaults
retain the previous path. The mixed double-layer right/up fusion spaces in
`full_update_J1.jl` are also corrected for unequal, non-self-dual SU(N) bra/ket
spaces. For identical bra/ket and SU2 these fused spaces agree with the old ones.

CTM corner/edge normalization may leave a common complex phase in an
unnormalized two-site density. Each FU bond divides ALL three quantities
(target norm, candidate norm, cross overlap) by the SAME old-state trace.
This leaves normalized fidelity invariant and avoids mistaking a negative
overall CTM phase for a negative physical norm. Open-layer traces agree with
the existing closed double layer to about3e-14 on the original SU4 state.

Local verification (2026-10-30):45 geometry/factorization/mixed-space/dense
optimization checks,48 original SU4 D9 checks, and17 configuration checks pass.
The SU4 tests include both horizontal and vertical bonds, both space policies,
fixed-space identity gates preserving the original site tensors, and all24
bond energies compared by coordinate with the existing observable. When the
CTM objective cannot resolve an improvement over the input, fixed_space keeps
the original coefficients instead of adopting an unweighted SVD gauge change.
A nonproduct dense regression accepts8 optimization steps and reduces its
local loss from1.1501658e-6 to1.0565104e-6. The single-job CLI completes a
2x2 dense J=0 sweep, measurements and checkpoint after restarting from its
initial checkpoint. Restarting a completed job adds no evolution/energy rows.
These are correctness checks, not a converged SU4 FU
ground-state calculation. A full SU4 D24 sweep has not been run locally.

```sh
julia --startup-file=no SU4/honeycomb/tests/test_full_update_config.jl
julia --startup-file=no SU4/honeycomb/tests/test_honeycomb_full_update.jl
julia --startup-file=no SU4/honeycomb/tests/test_honeycomb_full_update_SU4.jl
```
