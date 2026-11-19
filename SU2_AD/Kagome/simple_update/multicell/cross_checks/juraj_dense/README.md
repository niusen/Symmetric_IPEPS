# 服务器用 Juraj 复测 Julia U1 态

`states/D9.json`、`states/D10.json` 已由 Julia U1 最终态无截断地转换成 dense iPESS。
它们是 Juraj 原生 JSON，直接由 `ipeps.ipess_kagome.read_ipess_kagome_generic` 读取。
服务器只运行 Python/Juraj，不调用 Julia，不做 SU 或变分优化。

上传本目录，保留 `states/`。在 `run_server.py` 顶部把 `JURAJ_REPOSITORY` 改为
服务器已有、已支持 sqrt3 元胞的 `tn-torch_dev` 路径。激活 Juraj 的 Python 环境后运行：

```sh
sh run.sh
```

默认 D=9、10，chi=40，h=0.38，CPU 10 线程，CTMRG 最多 50 轮、阈值 1e-6。
全部设置在 `run_server.py`；可将 `device` 改为 `cuda:1`，或修改 `chis`。
`PYTHON_BIN` 可指定环境中的 Python，默认使用当前环境的 `python`。
`sh run.sh --foreground` 可前台运行；后台为单一进程组，没有自动重启循环。

最少上传 `run.sh`、`run_server.py`、`measure.py` 和整个 `states/`。
`run.jl` 只是保留的 Julia 导出入口，服务器测量不需要它。

输出在本目录 `results/`：

- `summary.csv`：交换能、磁场能、总能量、磁化、收敛状态，以及与 Julia 的差值。
- `bond_energies.csv`：18 条 bond，按 Julia 标签排列，包含逐 bond 对照。
- `site_magnetization.csv`：9 个 site 的 Sz。
- `D*/chi*/result.json`：逐点结果、原始观测量、输入/源代码哈希及映射检查。

`states/D*_julia_chi40.toml` 是对应最终态的 Julia chi=40 原始观测量。
修改 chi 后，输出的 `same_chi` 会标记是否仍为同一 chi。
完成点可续跑；改变脚本、输入、设置或 Juraj 代码后，设置新的 `out`。

转换采用 `(X,Y)=(1-x,1-y)`，单 site 物理顺序为 `(s0,s1,s2)=(B2,B3,B1)`。
Julia 上/下三角形对应 Juraj 下/上三角形。跨 site 三角形 RDM 的输出顺序为
`(RU.s1,LD.s2,RD.s0)`，测量脚本已显式对齐 bond 标签。
U1 输入已固定额外腿的分量，只去掉一维旁观腿，不重新投影或截断系数。

检查包括逆排列、单 site 张量恒等式、Juraj JSON 写入/读回恒等式。
CTMRG、模型和 RDM 使用 Juraj 自己的函数，本任务未修改这些计算函数。
交换能用 h=0 的模型测量，磁场项由独立测得的平均 Sz 加入一次：
`E_total/N=E_exchange/N+h*mean(Sz)`，`m_normalized=-2*mean(Sz)`。
同时核对三角形哈密顿量期望与逐 bond 能量之和。

Julia 导出函数在 `src/bosonic/kagome/native_measurement/juraj_dense_export.jl`。
