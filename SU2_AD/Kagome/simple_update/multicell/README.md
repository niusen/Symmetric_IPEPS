# Kagome 多元胞计算

当前 U1 D4→D10 任务：

```sh
sh run_u1_D4_D10_server.sh
```

配置在 `run_u1_D4_D10_server.jl` 顶部。包内已包含优化好的 U1 D4 初态；
默认 CPU、10 线程、chi=40、每个 D 9000 步、每个 D 只做一次最终测量，
CTMRG 最多 50 轮。[运行说明及上传清单](docs/U1_D4_D10_SERVER.md)。


此前 SU2 D4→D12 任务仍可运行：

```sh
sh run_su2_D4_D12_server.sh
```

所有计算设置集中在 `run_su2_D4_D12_server.jl` 顶部，默认 CPU、10 线程、
D=4..12、h=0.38、sqrt3xsqrt3、chi=40、每个 D 9000 步、CTMRG 最多 50 轮。
Julia 入口会自动准备依赖并启动计算，没有额外的 bootstrap/driver 转接文件。

```text
multicell/
├── run_u1_D4_D10_server.jl   # 当前 U1 任务配置和入口
├── run_u1_D4_D10_server.sh   # 当前 U1 后台任务
├── run_su2_D4_D12_server.jl  # 此前 SU2 任务入口
├── run_su2_D4_D12_server.sh  # 后台启动
├── Project.toml             # CPU 依赖
├── data/                    # 初态和 dense 参考数据
├── examples/                # 其他计算配置入口
├── tests/                   # 小规模检查、兼容性检查
├── docs/                    # 详细说明和历史验证记录
├── environments/gpu/        # 可选 GPU 依赖
└── visualization/           # 之前的虚空间绘图工具
```

计算函数位于 `src/bosonic/kagome/native_measurement/`，其他工作流函数位于
`src/bosonic/kagome/workflows/`；SU 内核继续复用原有 `src` 文件。
`Manifest.toml`、`results/` 和 `logs/` 为本机依赖或运行输出，不用上传旧文件。

上传 CPU 任务时，保留 `SU2_AD` 目录结构，上传所选任务同名的 `.jl` 和 `.sh`、
`Project.toml`、该任务引用的 `data/` 文件，以及最新 `SU2_AD/src/`。测试、说明、示例、绘图和
GPU 环境文件夹均不影响默认 CPU 任务启动。

详细说明：[服务器任务](docs/SU2_D4_D12_SERVER.md)。

D4 的 SU2→U1 后继续 simple update：直接运行 `examples/run_u1_from_su2.jl`，所有选项都在文件顶部；见 [U1 说明](docs/U1_USAGE.md)。

U1 D4→D10 服务器任务：`sh run_u1_D4_D10_server.sh`。设置在同名 `.jl` 顶部，
已包含优化好的 U1 D4 文本初态；详见 [U1 D4–D10 任务和上传清单](docs/U1_D4_D10_SERVER.md)。
