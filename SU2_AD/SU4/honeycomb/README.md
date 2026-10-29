# SU(4) honeycomb

常用入口只有 `run_star_su.sh`：从严格 D=9 星形覆盖态出发，在一个后台任务内完成
simple update 和 CTMRG 能量、键观测量计算。默认4×4 cell，更新上限 Dmax=24。

从 `SU2_AD` 目录启动：

```sh
sh SU4/honeycomb/run_star_su.sh
```

脚本自动后台运行并记录日志/PID；重复相同命令时自动读取检查点续跑。
只检查配置、不演化：

```sh
sh SU4/honeycomb/run_star_su.sh --dry-run
```

## 目录

```text
honeycomb/
├── run_star_su.sh          常用 sh 入口，路径保持不变
├── config/                 正式计划和轻量 pilot 配置
├── server/                 Julia 任务入口、可选 Slurm 模板
├── tests/                  正确性、截断和启动/续跑检查
├── experiments/
│   ├── baseline/           dense D4 基准与基本用例
│   ├── initialization/     小维数和 C3 初始化对照
│   └── star/               星形态精确计算、CTMRG 与结构核查
├── docs/                   服务器说明、历史实验记录
├── data/                   已有实验数据与检查点
├── logs/                   启动日志及已归档实验/测试日志
└── server_runs/            服务器任务输出，运行时创建
```

- [正式配置](config/StarSimpleUpdate.toml)：Dmax、时间步、演化步数、检查点和 CTMRG。
- [轻量 pilot](config/StarSimpleUpdate_pilot.toml)：只运行三轮，检查服务器环境。
- [服务器运行与续跑说明](docs/server.md)。
- [历史实验与物理核查](docs/experiments.md)。

不要求保留 `SU2_AD/SU4/honeycomb` 的目录层级。可把运行目录和已有源码分开放，
用 `SU4_SRC_DIR=/path/to/src sh run_star_su.sh` 指定源码；默认也会沿脚本位置和
当前目录的父目录查找源码。`SU2_AD_ROOT` 可指向已有项目根目录。

也可生成只含运行文件与必要源码的普通目录：

```sh
python SU4/honeycomb/server/prepare_star_run.py
```

直接上传生成的 `deploy/su4_star_run/` 文件夹即可，不需要压缩或解压。目录层次为：

```text
su4_star_run/
├── run_star_su.sh
├── server/                 Julia 运行入口
├── config/                 演化与测量配置
└── src/bosonic/            所需的原有算法库
    ├── honeycomb/          蜂窝模型与星形态流程
    └── triangle/           复用的两点更新和观测量辅助代码
```

在服务器进入该目录，执行 `sh run_star_su.sh`。不需要整个仓库，也不要求目录名。
根 `src/` 仍是唯一维护的算法源码，`deploy/` 只是供上传的拷贝，不另维护一套算法。
生成工具不会覆盖已有运行目录；更新时用 `--output /path/to/new_folder` 指定新目录，
以免覆盖服务器配置和检查点。运行目录本身不需要 Python，仍使用兼容 Julia 包环境。

正式运行不依赖 `experiments/`、本机数据或 Python。功能性代码在
`src/bosonic/honeycomb/`，相对 `SU2_AD` 的结构如下：

```text
src/bosonic/honeycomb/
├── Honeycomb_SU4.jl
├── honeycomb_SimpleUpdate.jl
├── honeycomb_spin_model.jl
├── multiplet_truncation.jl
└── star/
    ├── honeycomb_star_state.jl
    ├── StarRunConfig.jl
    └── StarSimpleUpdateRun.jl
```

本次整理只移动文件和修正路径，保留全部旧数据。配置移入 `config/` 后，正式输出
仍为 `SU4/honeycomb/server_runs/star_D24/`，旧服务器检查点可按原输出位置续跑。
