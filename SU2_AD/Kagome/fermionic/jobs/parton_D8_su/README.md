# D=8 的 Kagome parton simple update

六个任务各自从笔记 Fig. 21–26 对应的未投影 GFPEPS 张量开始。每个任务先用 CTMRG 测量输入态，再用同一个 parton Hamiltonian 做 simple update，最后从头构造 CTM 环境，用相同设置测量演化后的态。iPEPS 保留方格张量，iPESS 保留三角形与格点张量。初态 D=8，演化中每次 Z₂ 截断均取 Dmax=8。

两种 ansatz 都把 3×1 初态沿 y 平铺为 3×3，每个晶胞共 27 个 kagome 格点、54 条最近邻键。三行从相同初值开始，张量和键权重独立更新。SU 前后均调用原有的 `Fermionic_CTMRG_cell` 在整个 3×3 晶胞测量。当前三个 parton 任务保留每格点四维的费米子物理空间。作 Gutzwiller 单占据投影后，这个晶胞对应 27 个自旋 1/2。

| 模型 | 三角形相位参数 (φ↑, φ↓) | iPEPS 初态 | iPESS 初态 |
|---|---|---|---|
| `orig` | (−π/8, −π/8) | `orig_m3_r0` | `orig_m3_s12003` |
| `mild` | (−π/16, −π/8) | `mild_m3_r0` | `mild_m3_s12103` |
| `zero` | (0, −π/8) | `zero_m3_r0` | `zero_m3_s12003` |

|t↑|=|t↓|=1，跨胞 hopping 保留三列磁胞的键相位，U=0。μ↑、μ↓ 直接从各自输入读取，这六个态的数值分别为 −0.03934990911091607 和 −1.088628240676881。演化和测量均保留每个 kagome 格点的四维费米子物理空间。

## 一条命令提交六组任务

如果源码和六份初态已经在服务器上，从服务器的 `SU2_AD` 目录运行：

```bash
sh Kagome/fermionic/jobs/parton_D8_su/submit.sh
```

如果还需要上传，从本机的 `SU2_AD` 目录运行同一脚本，填入 SSH 地址和服务器上的绝对目录。此方式使用 `ssh` 和两端的 `rsync`，自动按 `transfer.list` 同步源码、任务文件和六份初态，传输成功后即提交。

```bash
sh Kagome/fermionic/jobs/parton_D8_su/submit.sh --remote user@host /path/to/SU2_AD
```

两种方式都会在服务器后台校验六份初态、顺序运行六组 SU 并生成 `Kagome/fermionic/iPESS/results/parton_D8_su/3x3/summary.tsv`。提交时终端显示节点名、后台启动进程的 PID（launcher）和日志路径。每个 Julia 进程的日志会打印 `node`、`pid`、`task` 及 `unitcell=3x3`。任务使用启动时的 Julia 环境。服务器使用 `niusen/TensorKit.jl` 的 `v0.14.9-multiplet-fix` 标签。

## 在服务器任务目录操作

**单独提交一组**时，选择对应的 `.sh` 文件。例如 `sh cases/ipess_mild.sh` 会在后台启动 iPESS 的 mild 模型。

| 模型 | iPEPS | iPESS |
|---|---|---|
| orig | `sh cases/ipeps_orig.sh` | `sh cases/ipess_orig.sh` |
| mild | `sh cases/ipeps_mild.sh` | `sh cases/ipess_mild.sh` |
| zero | `sh cases/ipeps_zero.sh` | `sh cases/ipess_zero.sh` |

用 `tail -f 日志路径` 查看进度。分别提交的任务完成后，运行 `sh run.sh report` 汇总结果。需要在前台观察单个任务时，可以运行 `sh run.sh run ipess mild`。

[settings.jl](settings.jl) 集中设置演化与测量。默认 (τ, Δτ) 依次为 (1.0, 0.05)、(0.4, 0.02)、(0.1, 0.005)，每段 20 步，共 60 步，总虚时 1.5。SU 截断容差为 10⁻¹⁰。两端 CTMRG 均用 χ=32，最多 200 次迭代，收敛阈值 10⁻⁷。这些值定义了一次固定虚时的比较，基态精度还需从更长虚时、更小步长和更大 χ 的独立计算判断。

每个 Δτ 使用对称的二阶 Trotter 分解。iPESS 依次执行下三角半步、上三角整步、下三角半步，同一层的九个三角形各自更新三个不同的 B 张量。iPEPS 将九个下三角按方格张量是否重叠分为三组，每组三个门。先按 1、2、3 组各做半步，九个独立的胞内上三角做整步，再按 3、2、1 组各做半步。跨周期边界也使用同一分组规则。

脚本默认一个 Julia 线程、四个 BLAS 线程。可用 `JULIA_BIN` 指定 Julia 可执行文件，用 `KAGOME_BLAS_THREADS` 调整 BLAS 线程数，用 `KAGOME_SEED_ROOT` 指定包含六个输入子目录的路径。初态校验检查 MAT 的 SHA-256、内嵌的原始 NPZ 指纹、张量类型、D 和化学势。换一套参数重新运行时，先将 `KAGOME_RUN_ROOT` 设为新的输出目录。

六个 `.sh` 共用本目录的 [task.jl](task.jl)，在这里选择算例、执行 SU 和汇总六组结果。[settings.jl](settings.jl) 保存初态清单和演化参数。直接用 Julia 启动单项任务的命令为 `julia task.jl run ipess mild`。

## 结果

默认输出根目录是 `Kagome/fermionic/iPESS/results/parton_D8_su/`。3×3 的总表和六组数值结果写在根目录下的 `3x3/`，运行日志写在根目录下的 `logs/`。设置 `KAGOME_RUN_ROOT` 可指定其他根目录。总表 `3x3/summary.tsv` 每个组合占一行，包含每格点能量及其变化、密度、磁化、实际最大 D 和两次 CTMRG 的迭代数、误差及收敛标记。每个组合分别保存在 `3x3/ipeps_orig/` 等子目录中。

| 文件 | 内容 |
|---|---|
| `comparison.tsv` | 各量的 SU 前、SU 后及差值（后减前）。`grand_energy_per_site` 包含 −μ↑n↑−μ↓n↓，`kinetic_energy_per_site` 只计 hopping。两者都按微观 kagome 格点归一化。|
| `sites.tsv` | 各磁胞坐标和子格上的物理 n↑、n↓、Sz。内部分量 a、b、c 对应原始输入的 B、A、C。Sz=(n↑−n↓)/2，相对单占据自旋 1/2 饱和值的磁化为 2Sz。|
| `bonds.tsv` | 每个上下三角形上有向 ab、bc、ca 六种键的分自旋 ⟨f†ᵢfⱼ⟩ 实、虚部，以及含厄米共轭的键动能。↓ 已从空穴基底换回物理电子。|
| `initial.jld2`、`final.jld2` | 两次 3×3 CTMRG 测量、模型、种子指纹和数值参数。`measurement.cell` 与 `parameters.unitcell` 都记录实际晶胞尺寸。|
| `history.tsv`、`state.jld2` | 每个 SU 步的截断误差和最近完成步骤的张量及键权重。终端输出另存在输出目录的 `logs/`。|

总表中 `complete` 表示两端 CTM 均已收敛；`ctm_unconverged` 表示已有两端数值，其中至少一次未达阈值。`not_run`、`incomplete`、`invalid` 分别表示尚未运行、未生成两端测量或校验失败。再次执行已有完整结果的任务会核对参数并重建表格。中断的任务保留现场，可选用新的输出根目录重新从 GFPEPS 开始。
