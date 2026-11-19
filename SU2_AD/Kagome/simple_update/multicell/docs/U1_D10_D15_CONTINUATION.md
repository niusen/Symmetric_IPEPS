# Julia U(1)：读取已有 D10，继续到 D15

入口为 `multicell/run_u1_D10_D15_server.jl`，所有选项都在文件顶部。
使用原来的纯Julia TensorKit U(1) iPESS、sqrt3×sqrt3 cell；
不调用Python，不修改原来的D4→D10入口或算法函数。

## 读取哪个态

默认直接读取原结果中的
`multicell/results/u1_from_D4_D4_D10_h038_chi40/D10/final.jld2`。

SHA256为
`b16ca062dd869dfe2431e00d0a201e0571d7935fcc1e91fbf5707dffb6627607`。
这就是已下载、原χ=40复测交换能 −0.42235052773900206/site 的D10态。
D10只作为初态读取，不重新随机初始化、不重新优化，也不重算其观测量。
程序不会写入原来的D10文件或旧结果目录。

新任务计算链：
`已保存D10 → D11 → D12 → D13 → D14 → D15`。
每一级读取前一级的全部U1张量及lambda，不添加随机扰动、不重新投影SU2。
将Dmax升到新的D后，由原来的SU/SVD更新自动生长虚拟空间。
D是每根虚拟腿的总维数，不是charge sector的数量；结果仍记录实际保留的维数。

## 上传哪些新增文件

服务器已有上一轮完整Julia库和D10结果时，只需要新增3个运行文件：

1. `Kagome/simple_update/multicell/run_u1_D10_D15_server.jl`
2. `Kagome/simple_update/multicell/run_u1_D10_D15_server.sh`
3. `src/bosonic/kagome/native_measurement/u1_continuation_workflow.jl`

必须保持相对位置。原来的 `u1_server_workflow.jl`、U1算法、CTMRG等仍需保留。
本说明在`multicell/docs/`，小测试在`multicell/tests/test_u1_continuation.jl`，
运行不依赖这两个说明/测试文件。服务器若尚未保留原来的D10结果，需要把上面的
`final.jld2`也上传到入口指定的位置；无须重新上传D4–D9的结果。

## 启动、日志与续算

进入服务器原来的Julia `multicell` 目录后：

```sh
sh run_u1_D10_D15_server.sh
```

与原来的sh格式一致：自动后台运行，stdout/stderr一起写入
`logs/u1_D10_D15_时间戳_PID.out`，同名`.pid`保存任务PID。
提交前仅加载设置并核验D10文件SHA256，不进行演化或CTMRG。
一个Julia进程组，没有自动重启循环。可退出终端，无需在外面额外加`&`。
如需前台调试，执行 `sh run_u1_D10_D15_server.sh --foreground`。
可用 `JULIA_BIN` 指定Julia可执行文件，依赖环境沿用当前multicell项目。
停止整组进程使用启动时打印的 `kill -TERM -- -PID`。

中断后执行相同sh即可。SU从原子的`latest.jld2`和原Trotter阶段位置继续，
不会重复插入初始半步；已完成的SU和最终CTMRG测量会跳过。
原有的writer内核锁阻止同目录重复写入，遗留锁文件本身不会阻止重启。
每一级的父态SHA256和plan_id均会核验；改变父态或设置后拒绝混合旧检查点。
新任务输出目录与原来的D4→D10分开：

`multicell/results/u1_from_D10_D11_D15_h038_chi40/`

## 数值与输出格式

- h=0.38；沿用原Julia的 `H=Hex+h*sum(Sz)`、固定1/9扇区的约定。
- CPU，10线程；χ=40；CTMRG最多50轮，阈值1e-6。
- SU截断阈值1e-6；每250个完整步保存检查点；每100步打印进度。
- dt依次为0.03、0.01、0.003、0.001、0.0005。
- 请求τ依次为40、20、5、2、1。沿用`round(Int,tau/dt)`，即
  1333、2000、1667、2000、2000步，每个新D共9000个完整Trotter步。
- 这里是原来的固定演化日程，没有新增自适应SU收敛门槛，不能据此宣称SU严格收敛。
- 每D结束后只做一次最终CTMRG和完整测量；CTMRG未达标时如实记录标志，
  仍按原流程继续下一D。单个χ也不能验证χ收敛。

继续使用原来的汇总格式：summary.csv/json/toml/txt、site_magnetization.csv、
bond_energies.csv、triangle_observables.csv、virtual_spaces.csv。
各D保存initial/latest/final.jld2、stage检查点、CTMRG观测量及进度报告；
新增`lineage.toml`记录父D、原始D10和最终态指纹。
不读取历史dense结果进行比较；为保持表格格式，原来的参考值列留空。

测试只加载验证真实D10，不在本机演化D11–D15。小规模链式检查使用D4态、
每级两步SU与χ=2、CTMRG最多2轮，测试加载、继承、输出及中断恢复规则。
