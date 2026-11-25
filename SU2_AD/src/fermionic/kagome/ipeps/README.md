# Kagome 原生 iPEPS 的 simple update

在 Linux 服务器上比较 D=8 的 iPEPS、iPESS 与三个 parton 模型时，使用[六组 GFPEPS 初态的 SU 任务](../../../../Kagome/fermionic/jobs/parton_D8_su/README.md)。该入口包含输入张量的传输清单和 SU 前后的 CTMRG 对比表。

从 [SimpleUpdate.jl](../../../../Kagome/fermionic/iPEPS/SimpleUpdate.jl) 运行，模型、场强、时间步与输出路径直接写在 Julia 里。从 SU2_AD 目录启动：

```sh
APP=Kagome/fermionic/iPESS
SEEDS="$APP/results/gfpeps_validation"
ENTRY=Kagome/fermionic/iPEPS/SimpleUpdate.jl
julia "$ENTRY" ipeps zero "$SEEDS/ipeps_D8_zero/tensors.mat" "$APP/results/ipeps_su/ipeps_D8_zero"
julia "$ENTRY" ipeps spin "$SEEDS/ipeps_D8_zero/tensors.mat" "$APP/results/ipeps_su/ipeps_D8_zero_spin"
```

第一行计算零 ↑ 三角通量的费米子模型。第二行使用**同一输入态**，先按物理电子单占据投影，再演化 Heisenberg + 纵场。改用原生 iPESS 时，首参数用 `ipess`，输入也要换成对应的 iPESS 张量文件。示例只演化一个时间步；对比能量时须检查步长、虚时、截断和 CTMRG 的 χ 及收敛。

| 模型 | Hamiltonian 与输入约定 |
|---|---|
| `orig / mild / zero` | ↑ 三角形通量依次为 −π/8、−π/16、0，↓ 固定 −π/8。磁胞三列，跨胞键保留 2π/3 磁相位。 |
| `hubbard` | 无磁通量，动能 −tσ、相互作用 U n↑n↓、化学势 −μσ nσ。U 默认为零，可在脚本里改。 |
| `spin` | 纯自旋 1/2，J Σ Sᵢ·Sⱼ − h Σ Sz；虚腿仍保持费米子 Z₂ 分级。 |

费米子模式下，`t=(t↑,t↓)` 和 `mu=(μ↑,μ↓)` 都按**物理自旋**排序；默认从 MAT 读取化学势。原张量的数据基底是 b=(f↑,f↓†)：一个物理 ↓ hopping 在这个基底中变为负共轭，n↓=1−n_hole。这些转换由 [parton.jl](../models/parton.jl) 实现，三角门和能量测量共用同一套系数。输入态的标签与所选 Hamiltonian 可以不同。

Fig. 21–26 的 D=8 输入位于上述 SEEDS 目录：原版为 `ipeps_D8`、`ipess_D8`，另两种在对应目录名后加 `_mild` 或 `_zero`。每份 `tensors.mat` 旁的 `source.json` 记录原始态和 SHA-256。迁移到服务器时须单独复制 MAT，Git 默认忽略结果目录。

原生方格张量的物理腿由三个 kagome 格点融合而成。上三角门在一个张量内部作用，下三角门连接左下 c、左上 b、右上 a 三个方格张量。实现把其余物理腿与外虚腿一起做 QR 分解，再对两个内部键作 Z₂ SVD 截断。调用 `run_native_su` 时可用 `unitcell=(3,3)` 将原生 3×1 初态平铺为九个原胞。六组 D=8 服务器任务均使用这个选项。直接调用时省略它，方格 SU 会把 3×1 输入平铺为 3×2。每一步按互不共享方格张量的下三角组依次做半步，所有上三角做整步，然后倒序对各下三角组做半步。

比较 27 自旋 VBC 时，`run_native_su(family, input, output, :spin; unitcell=(3,3), schedule, Dmax, J, h)` 会对 MAT 初态先作 Gutzwiller 投影，使用 Heisenberg 加纵场的三角门，并测量交换能、磁化和逐键自旋关联。

输出目录须为新目录。`initial.jld2`、`final.jld2` 含独立 CTMRG 能量、逐键量及收敛信息（费米子同时记录物理 ↑/↓ 密度，自旋记录磁化）；`history.tsv` 记录各步截断误差。`state.jld2` 包含每个方格张量和键权重，可替代 MAT 输入续算到新目录。D=8 的 CTMRG 测量比单步 SU 昂贵；仅做演化时，可直接在 Julia 调用 `SquareState(A,U)` 和 `ipeps_su!`。

实现核对：`Kagome/fermionic/iPESS/tests/test_native_ipeps_su.jl` 从三角门矩阵元组装磁胞 Bloch 矩阵，与原笔记的 A/B/C 电子规范逐元素比较；另外在三角外的虚腿为一维的网络上，将下三角演化与精确三站点波函数的逐键期望比较。后一项经过独立的 CTMRG 测量，检查对角键、空穴符号和未参与门的物理轨道。测试的是门及收缩的正确性，不是优化后基态的能量精度。

已有短程回归：三个模型 × 两种自旋的 9×9 Bloch 矩阵逐元素对照 6 项通过；孤立下三角（自旋、费米子、含奇宇称旁观轨道）对照 18 项通过；Hubbard 粒子–空穴参数 4 项通过。实际原版 D=4 和 D=8 的原生 iPEPS 输入均分别以投影/未投影形式完成 SU 单步。D=4 另以小 χ 验证初末测量和检查点读回；这些短运行不提供基态收敛精度。
