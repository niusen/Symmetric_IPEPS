# 修改下面的 Julia 配置，再运行整个文件。只计算 sqrt3xsqrt3、h=0.38。
# 本文件做 SU2 simple update；不是梯度变分优化。CTMRG 测量另行读取输出态。
using Dates
su2_convergence_settings=(
    cases=[(D=2,initialization=:integer_singlet),
           (D=2,initialization=:integer_two_singlets),
           (D=3,initialization=:homogeneous),
           (D=3,initialization=:integer_triplet),
           (D=3,initialization=:uniform),
           (D=4,initialization=:homogeneous),
           (D=4,initialization=:integer_zero_triplet),
           (D=4,initialization=:uniform)],
    seeds=[1234],
    parity_pattern=Bool[0 0 1;0 1 1;1 0 1], # 行为独立 PESS，列为 B1/B2/B3；true 表示 up 腿半整数。
    h=0.38,
    threads=10,
    schedule=[(dt=.03,tau=40.),(dt=.01,tau=20.),(dt=.003,tau=5.),
              (dt=.001,tau=2.),(dt=.0005,tau=1.)],
    check_every=100,
    minimum_steps=300,
    stable_checks=3,
    lambda_tolerance=1e-8,
    wavefunction_tolerance=1e-8,
    truncation_tolerance=1e-6,
    hold_dt=.0005,
    hold_tau=.5,              # 额外完整演化 1000 步，保存前后态作 CTMRG 对照。
    continuation_input="",   # 留空为新计算；填 after_hold/state.jld2 则从它继续。
    continuation_schedule=[(dt=.03,tau=400.),(dt=.01,tau=40.),
                           (dt=.003,tau=10.),(dt=.001,tau=5.),(dt=.0005,tau=2.)],
    out=joinpath(@__DIR__,"..","results","su2_D234_"*Dates.format(now(),"yyyymmdd_HHMMSS")),
)
# :homogeneous 的初始表示：D3 为 0+1/2，D4 为 0+0+1/2。
# :uniform 使用相同表示，但将所有约化系数设为同一实数，避免只检查随机复数初态。
# 纯 1/2（以及纯两份 1/2）不能在当前 B(down,up,physical) 结构中产生非零 B。
# :integer_singlet 使用固定合法 parity_pattern，整数腿为 0，半整数腿为 1/2。
# :integer_two_singlets 把整数腿设成 0+0（维数2），半整数腿仍为单个1/2。
# 同时保存每个 B 沿虚腿展开后的矩阵秩，检查第二份0是否真正参与物理态。
# 另两种初始化使用同一分布，把整数腿换成 1（D3）或 0+1（D4）。
# 不从随机 D3 态独立截断各腿来选此分布，避免不同 SVD 的选择不相容而产生零 Tu。
# D4 均匀初始化可能自动退化为实际 D3，所以也测试整数/半整数分布不同的分支。
# SVD 允许改变表示；每条腿的最终实际维数和表示都会写入 report.toml。
# 只有 lambda 和九自旋波函数连续多个检查点都稳定才允许某个 dt 阶段提前结束。
# 这些局部检查仍需配合 before_hold/after_hold 的无限系统 CTMRG 能量比较。
if !isdefined(@__MODULE__, :run_su2_convergence)
    include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","workflows","charged_su2_convergence_driver.jl"))
end
if isempty(su2_convergence_settings.continuation_input)
    run_su2_convergence(su2_convergence_settings)
else
    continue_su2_convergence(su2_convergence_settings.continuation_input,
        su2_convergence_settings.out,su2_convergence_settings;
        schedule=su2_convergence_settings.continuation_schedule)
end
