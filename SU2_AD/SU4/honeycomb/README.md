# SU(4) honeycomb

常用入口只有 `run_star_su.sh`：从严格 D=9 星形覆盖态出发，在一个后台任务内完成
四站点 star HOSVD simple update 和 CTMRG 能量、键观测量计算。默认4×4 cell，
初态仍为 D=9，更新上限 Dmax=24；三个模式从同一个 cluster 张量截断并共同投影，
沿用 kagome 的 HOSVD 方法，不做额外的核心二次 HOSVD。

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

只维护项目原有的一份通用 `src/`，不再生成带另一份源码的运行目录。
运行目录只需要入口和配置：

```text
现有通用源码/src/bosonic/honeycomb/   维护和更新算法的位置
运行目录/
├── run_star_su.sh
├── server/                         Julia 入口
└── config/                         演化与观测量配置
```

运行目录不在原项目层级时，引用服务器已经存在的通用源码即可：

```sh
SU4_SRC_DIR=/path/to/existing/src sh run_star_su.sh
```

也可把这条路径写在 sh 顶部预留的位置。无需复制源码、生成目录或解压文件。
服务器已有的通用源码需要同步本次的蜂窝模块更新；这些更新仍放在原 `src/` 中。

正式运行不依赖 `experiments/`、本机数据或 Python。功能性代码在
`src/bosonic/honeycomb/`，相对 `SU2_AD` 的结构如下：

```text
src/bosonic/honeycomb/
├── Honeycomb_SU4.jl
├── honeycomb_SimpleUpdate.jl
├── honeycomb_StarSimpleUpdate.jl
├── honeycomb_spin_model.jl
├── multiplet_truncation.jl
└── star/
    ├── honeycomb_star_state.jl
    ├── StarRunConfig.jl
    ├── StarSimpleUpdateRun.jl
    └── StarUpdateTrace.jl       旧两站点方法的诊断工具
```

默认输出为 `SU4/honeycomb/server_runs/star_hosvd_D24/`，与旧两站点结果分开。
旧方法需显式指定配置 `method="two_site"` 或函数参数 `method=:two_site`。
演化方法写入检查点，旧两站点检查点不能直接作为 HOSVD 的同一次演化自动续跑。
# Two-site full update

`sh SU4/honeycomb/run_full_update.sh --dry-run` previews
`config/FullUpdate.toml`; remove `--dry-run` to launch evolution and CTMRG
observables in the same background job. It reuses `run_star_su.sh` for logs,
PID and automatic resume, and the existing shared `src` for algorithms.

The default starts from the original D=9 star RVB and uses
`space_policy="fixed_space"`: keep the input irreps and their multiplicities
on each real bond, while optimizing tensor entries with the CTM environment.
`space_policy="tensorkit"` uses native `truncdim(Dmax)` and allows the spaces
to change. Use a separate output directory for each policy. Details and
limitations are in [the server guide](docs/server.md#two-site-full-update).
