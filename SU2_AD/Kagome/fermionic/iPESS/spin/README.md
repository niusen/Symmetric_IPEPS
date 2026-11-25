# 从投影 GF-iPESS 优化 kagome 自旋态

在服务器的 `SU2_AD/` 目录下运行。SU、FU、AD 各自从 `D4.toml` 指定的同一原生 GF-iPESS 开始。代码位置与 Julia 调用示例见[源码说明](../../../../src/fermionic/kagome/README.md)。

```sh
APP=Kagome/fermionic/iPESS
bash "$APP/server/run.sh" setup
for method in su fu ad; do
    bash "$APP/server/run.sh" spin "$method" "$APP/spin/D4.toml" || break
done
```

模型取自旋 1/2，$H=J\sum_{\langle ij\rangle}\mathbf S_i\cdot\mathbf S_j-h\sum_i S_i^z$。
示例场为 $h/J=0.5$，可直接改 `[model]`。三角门各含三个交换键及每个格点一半的 Zeeman 项。
物理腿为两维纯自旋，保留虚腿 Z2 费米分级、上下 simplex 和三个 B 张量。
优化这五个张量，可以离开 Gaussian 变分族。Z2 不施加固定总 Sz 约束。

沿用三角晶格 Hofstadter–Hubbard 计算中先设模型和截断参数、分段演化、保存态和独立测量的方式。
SU 与 FU 共用 `[[schedule]]`，可追加不同的 `tau`、`dt`。每个时间步依次作用下三角半步、
上三角整步、下三角半步。FU 的 `n_sweep` 控制每个三角形在 CTM 内积下拟合三个 B
与 simplex 的最多 ALS 轮数，`use_gauge` 控制是否同时比较经环境规范变换的初值。
AD 经固定次数的 CTMRG 迭代求导，使用回溯步长；`memory=0` 为梯度下降，正数选择 L-BFGS。
Kagome 的元胞和张量连法由现有 Kagome 更新器负责。入口位于本目录，模型、可微能量
及运行函数位于 `src/fermionic/kagome/spin/`。

输入支持原生 iPESS 导出的 `tensors.mat` 和已投影的 `state.jld2`。MAT 导入时作一次局域单占据投影。
若要从 SU 接 FU，将 `job.input` 改为 SU 结果的 `state.jld2`，`job.output` 指向新的运行目录。
路径以 TOML 所在目录为基准。同一方法的输出目录已存在时会报错。D=8 时同时修改输入、输出和
`D_max`。AD 保持输入的虚空间，SU 和 FU 用 `D_max` 截断。

每种方法写入 `job.output/{su,fu,ad}/`。用相同 `[measurement]` 设置的 `initial.toml`
和 `final.toml` 比较每格点总能量、交换能、Zeeman 能以及磁化。逐键 $\langle S_i\cdot S_j\rangle$
和逐格点三分量磁化在 `*_observables.jld2`。每格点总能量为
`exchange_per_site - h * magnetization_per_site[3]`，`mz_over_saturation` 为该磁化的两倍。
`history.tsv` 分别记录 SU 截断权重、FU 拟合保真度/残差或 AD 有限迭代目标。
示例为短演化，正式能量需检查时间步、优化步数、环境维数及 CTMRG 收敛。
