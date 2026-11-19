# 固定 D=9、10 的 U1 态，检查 chi 收敛

在 `Kagome/simple_update/multicell/` 运行：

```sh
sh run_u1_D9_D10_chi_check.sh
```

所有设置位于同名 `.jl` 顶部。默认读取原 U1 D4→D10 任务的
`results/u1_from_D4_D4_D10_h038_chi40/D9/final.jld2` 和 `D10/final.jld2`。
如果服务器实际目录不同，只修改 `sources` 中两个路径。
只测一个 D 时，注释掉另一个 `sources` 条目即可。

默认 chi=20、30、40、60、80，h=0.38，CPU、10 线程、CTMRG 最多 50 轮、
角张量谱收敛阈值 1e-6。每个 chi 独立初始化环境，不做 SU、不改变原态。
sh 会从 Julia 设置中的 `threads` 读取线程数，同时设置 Julia、BLAS 和
OMP/MKL 线程数；默认均为 10，只需修改 Julia 文件。

输出目录为 `results/u1_D9_D10_fixed_state_chi_check/`：

| 文件 | 内容 |
|---|---|
| `summary.csv`、`summary.txt` | 各 D、chi 的交换能、磁场能、总能量、磁化、CTMRG 是否收敛、实际保留维数 |
| `bond_energies.csv` | 每个 D、chi 的 18 条不等价键，含 site 标签、方向和相邻 chi 的能量差 |
| `site_magnetization.csv` | 9 个不等价 site 的 Sz 和归一化磁化 |
| `triangle_observables.csv` | 6 个三角形的能量及三体关联 |
| `D*/chi*/observables.toml` | 每个点的完整原始测量、输入哈希及设置 |

键的 `energy` 是 J=1 时的 `<Si.Sj>`，不含磁场能。
每 site 的交换能等于 18 条键能量之和除以 9。
总能量为 `exchange_energy_per_spin + field_energy_per_spin`，
采用原有约定 `H=Hex+h*sum(Sz)`、`m_normalized=-2*mean(Sz)`。

汇总表给出相邻已完成 chi 的最大 bond 变化和交换能变化，以及相对 chi=40
的总能量变化。所有行保留收敛状态；达到 50 轮但未收敛的点也会保存，不能把
它当作已收敛结果。达到迭代阈值只说明该 chi 下的迭代收敛，仍需比较不同 chi。

每完成一个点就更新汇总。中断后用原 sh 重跑，已完成且输入/设置相同的点会跳过。
改变 chi 列表、容差、输入 checkpoint 或依赖版本时，设置新的 `out` 目录。
任务只锁定自己的输出目录，不锁原 SU 目录，不写入原 checkpoint。

这是原测量程序的 chi 收敛检查，尚不是另一套独立收缩算法的交叉验证。

服务器保留原有包目录及依赖，增量上传：

1. `Kagome/simple_update/multicell/run_u1_D9_D10_chi_check.jl`
2. `Kagome/simple_update/multicell/run_u1_D9_D10_chi_check.sh`
3. `src/bosonic/kagome/native_measurement/chi_scan_workflow.jl`

需要原服务器的 D9、D10 `final.jld2`，不需要重新演化。
