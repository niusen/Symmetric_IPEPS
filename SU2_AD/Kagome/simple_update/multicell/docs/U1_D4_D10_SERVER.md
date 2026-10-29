# U1 D4–D10 服务器任务

在本目录执行一次：

```sh
sh run_u1_D4_D10_server.sh
```

所有计算选项都在 `run_u1_D4_D10_server.jl` 顶部。不需要 Slurm 或命令行参数。
默认后台运行，终端会显示日志路径、PID 和停止整个任务的命令。
`sh run_u1_D4_D10_server.sh --foreground` 可用于前台查看启动错误。

## 初态及计算方案

- 初态为 `data/u1_D4_optimized_seed.toml`：本机 SU2 D4 投影后，已经做过
  600 步 U1 simple update 的候选态；h=0.38 时，chi=24 测得
  E/site=-0.4374196134316011，m=0.11111111057450687。
- 文本初态保留所有 B、Tu、Td、lambda、电荷重数、箭头及指标分组。
  每条腿显式保存基底电荷，读入时按电荷重排，适配不同 TensorKit 版本。
  它不是只有物理张量、丢失 lambda 的观测量导出文件。
- 固定 sqrt3xsqrt3 cell、J=1、h=0.38、1/9 磁化扇区。
- 从优化好的 U1 D4 初态继续完整演化 D4，然后 D5 使用本轮 D4 最终态，
  依次到 D10。无随机初始化、无扰动、无 SU2 多重态约束。
- 每个 D 的五阶段为 dt=[0.03,0.01,0.003,0.001,0.0005]，
  tau=[40,20,5,2,1]，实际整步数=[1333,2000,1667,2000,2000]，共 9000 步。
- CPU、10 线程、chi=40；每个 D 完成全部 SU 后只测量一次。
  CTMRG 最多 50 轮、收敛判据 1e-6。未收敛结果保留明确标记。

D 表示每条 bond 上 U1 电荷态总数的上限，实际维数可能更小。
增大 D 时，更新通过现有 U1 SVD 自行产生并选择电荷态。
总磁化由一维辅助电荷腿固定；本任务检验该扇区内的能量，不测平台宽度。
均匀磁场项在该扇区是常数，SU 使用交换项 gate，观测量中只加一次磁场能。

## 结果和中断恢复

默认目录：`results/u1_from_D4_D4_D10_h038_chi40/`。

每完成一个 D，根目录更新：

- `summary.txt`、`summary.csv`、`summary.toml`、`summary.json`：能量、
  交换/磁场能、平均磁化、实际 D/角张量维数、CTMRG 收敛与测量耗时。
- `site_magnetization.csv`：9 个独立 site 的磁化。
- `bond_energies.csv`：18 条独立 bond 的能量。
- `triangle_observables.csv`：6 个三角形的能量和原始 SzSzSz 三体关联。
- `virtual_spaces.csv`：各条腿的 U1 电荷及重数。

summary 同时保留已有 dense 参考的能量、chi、初始化来源及收敛标记，
差值列名为 `delta_E_u1_minus_dense`；不同 chi 的参考不能当成严格同条件比较。

各 `D4/`…`D10/` 子目录保存 `initial.jld2`、`latest.jld2`、
`stage_01.jld2`…`stage_05.jld2`、`final.jld2`、`progress.toml`、
`virtual_spaces.*` 和 `u1_state.toml`。`ctm/` 保存完整观测量和转换后的 U1 态。

SU 每 250 步存一次 latest 检查点，不运行中间 CTMRG。
中断后重新执行同一个 sh，会恢复当前阶段；不会重复阶段开头的半步。
已经完成的 SU 和最终 CTMRG 会跳过。CTMRG 本身没有逐轮检查点：
若恰在测量时中断，恢复后重做该 D 的最终测量，SU 不必重算。

保持设置、初态和 TensorKit/JLD2 版本不变；如需改变，修改 out 使用新目录。
同一个结果目录只允许一个进程写入。若已有任务在跑，请使用提交时显示的
停止命令先结束它；不要删除 lock 文件来绕过正在运行的任务。
任务没有自动重启循环。

## 上传哪些文件

保留 `SU2_AD` 目录结构，上传：

1. 最新 `SU2_AD/src/`。
2. 本目录 `Project.toml`、`run_u1_D4_D10_server.jl`、`run_u1_D4_D10_server.sh`。
3. `data/u1_D4_optimized_seed.toml` 和 `data/dense_reference_h038.csv`。

不用上传本机 Manifest、results、logs、tests、绘图目录或 GPU 环境。
首次启动时 Julia 会准备依赖。无需 Python，也没有 ZIP 包。

新增工作流和种子读写函数在 `src/bosonic/kagome/native_measurement/`，
SU、SVD 和 CTMRG 数值内核复用已有实现。

## 本机小检查（2026-10-29）

- 初态从原优化态导出后，34 项完整系数/权重往返检查通过。
- TensorKit 0.14.11：初态/电荷块重排 71 项、阶段内恢复 5 项、
  D4→D5 端到端 23 项，共 99 项通过。
- TensorKit 0.17.0：初态/物理量一致性 71 项、阶段内恢复 5 项，共 76 项通过。
- 端到端只做每个 D 两步 SU、两轮 chi=2 CTMRG；D5 实际最大维数达到 5。
  此测试的能量和磁化不是物理收敛结果，不用于与 dense 比较。
- 再次启动后，已完成的 SU 和最终测量被跳过，检查点及观测量哈希不变。
- sh 通过 POSIX 语法检查且为 LF 换行；正式配置已检查为 7 个 D、
  每个 D 9000 步、CPU/10线程、chi=40、CTMRG<=50、误差阈值1e-6。
  本机未启动正式 D4–D10 扫描。
