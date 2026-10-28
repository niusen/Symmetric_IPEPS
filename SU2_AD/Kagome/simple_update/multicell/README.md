# Kagome 多元胞计算

正式服务器任务只需运行：

```sh
sh run_su2_D4_D12_server.sh
```

所有计算设置集中在 `run_su2_D4_D12_server.jl` 顶部，默认 CPU、10 线程、
D=4..12、h=0.38、sqrt3xsqrt3、chi=40、每个 D 9000 步、CTMRG 最多 50 轮。
Julia 入口会自动准备依赖并启动计算，没有额外的 bootstrap/driver 转接文件。

```text
multicell/
├── run_su2_D4_D12_server.jl  # 配置和正式入口
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

上传正式 CPU 任务时，保留 `SU2_AD` 目录结构，上传本目录两个运行文件、
`Project.toml`、`data/`，以及最新 `SU2_AD/src/`。测试、说明、示例、绘图和
GPU 环境文件夹均不影响默认 CPU 任务启动。

详细说明：[服务器任务](docs/SU2_D4_D12_SERVER.md)。
