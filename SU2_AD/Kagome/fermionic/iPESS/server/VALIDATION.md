# 目录整理与迁移验证（2026-11-22）

本次只整理 Kagome 工作流、路径和依赖环境。原有三角晶格与共享源码无改动。
历史物理结论保留在 `benchmarks/gfpeps_validation/reports/`；本次短运行仅作
程序回归，不作为新的收敛能量结果。

## 已完成的检查

- Julia 1.11.6 下，94 个 Kagome Julia 文件静态解析通过。
- 90 处字面量 include 路径存在，且大小写符合 Linux 文件名。
- GFPEPS 配置检查 33 项通过；覆盖相对路径、输入缺失和非法 χ 序列。
- Heisenberg 逐键/自旋方向及 singlet 测试 45 项通过。
- 投影存档/物理腿顺序测试 15 项通过。
- 紧凑双层与原共享收缩等价测试 19 项通过；最大相对差约 2.45e-16。
- Gutzwiller 代数、D=4/8 导入与投影交换性、乘积态 CTMRG 测试 60 项通过。
- 独立 Gaussian/Fock 扇区小体系测试 8 项通过。
- 元胞旋转前向等价测试 7 项、反向传播解析值/有限差分测试 2 项通过。
- 配置式 D=4 投影测量、来源记录、输出保护和环境文件续算测试 6 项通过。
- SU 薄入口实际完成两步更新并保存；FU/AD 保存、读取与续算测试 14 项通过。
- 独立生产 AD 入口完成两步优化和零步续算；方向导数与有限差分的相对差
  为 1.25e-10。加载 GFPEPS 紧凑双层后的同一检查为 5.08e-11。
  这些是小体系程序回归精度，不是物理能量的收敛精度。
- 本机使用迁移环境中的 TensorKit 快照实际加载和计算；70 个快照文件与
  原开发检出逐字节相同。
- 在新的带空格目录中解压迁移包，通过 Bash 入口完成输入检查；25 个输入
  文件与包旁校验清单一致。
- Bash 脚本语法、Python 辅助代码语法和 Git 补丁空白检查通过。

物理回归在独立项目环境下执行，GFPEPS 核心测试还从不同的当前工作目录运行。
迁移验证使用 **Windows + Git Bash**。Linux 服务器使用已有的 Julia 环境，
可用 `bash server/run.sh setup` 检查包是否能加载。

## 回归检查发现的修正

`compute_ob.jl` 原先末行将 `@__FILE__` 与 `&&` 连写，导致 Julia 解析失败；
入口改为显式 `if`，计算函数迁入 `src/fermionic/kagome/workflows/`。

Kagome 通用元胞旋转此前使用运行时整数索引 `getfield(a,d)` 读取共享
`Tset_struc`。前向结果正确，但 Zygote 0.7.10 的反向传播报
`UndefVarError: j not defined`。现仅在 Kagome 适配层中改用静态字段
`.T1` 到 `.T4`，保持相同旋转、收缩和费米符号；没有修改共享 CTMRG。

## 复现主要回归

在 `SU2_AD` 下：

```sh
APP=Kagome/fermionic/iPESS
julia "$APP/tests/test_server_config.jl"
julia "$APP/tests/test_server_workflow.jl"
julia "$APP/benchmarks/gfpeps_validation/tests/test_gutzwiller.jl"
julia "$APP/benchmarks/gfpeps_validation/tests/test_heisenberg.jl"
julia "$APP/tests/test_optimization_runner.jl"
```

测试第一次运行包括 Julia 编译时间。`test_server_workflow.jl` 使用 χ=4、两次
CTMRG 迭代检查数据流与续算；出现未收敛或磁化容差提示是预期的，不能将其
能量作为正式的精度结果。
