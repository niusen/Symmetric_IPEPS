# 服务器任务：只修改这里的配置，然后 sh run_su2_D4_D12_server.sh。
# 演化、U(1)/dense 转换、CTMRG、观测量和数据保存全部使用 Julia/TensorKit。
const su2_server_settings = (
    dims = collect(4:12),
    h = 0.38,                         # H = H_exchange + h*sum(Sz)
    chi = 40,
    threads = 10,
    device = "cpu",                   # 可选 "cuda:1"；默认使用 CPU。
    ctm_iterations = 50,
    ctm_tolerance = 1e-6,
    schedule = [(dt=0.03, tau=40.0), (dt=0.01, tau=20.0),
                (dt=0.003, tau=5.0), (dt=0.001, tau=2.0), (dt=0.0005, tau=1.0)],
    truncation_tolerance = 1e-6,
    checkpoint_every = 250,           # 只存 SU 态，不运行 CTMRG/有限环面测量。
    progress_every = 100,
    initial_state = joinpath(@__DIR__, "su2_D4_seed.toml"),
    measurement_representation = :u1, # :u1（默认分块）或 :dense（无对称性 TensorMap）
    out = joinpath(@__DIR__, "results", "su2_D4_D12_h038_chi40_native"),
)
# 固定 sqrt3xsqrt3，辅助 j=1/2，投影最高权分量，m=-2< Sz >=1/9。
# D4 从已收敛候选态再演化；D5 从本轮 D4 出发，依次 D -> D+1。
# 每个 D 都完整做五阶段，round(tau/dt)=[1333,2000,1667,2000,2000]，共9000步。
# 没有提前停止、额外 hold、随机扰动或中间 CTMRG；每个 D 最后测量一次。
# D 是完整磁量子态数的上限，不是多重态数；实际表示和维数另存。
if !isdefined(@__MODULE__, :KAGOME_SETTINGS_ONLY)
    include(joinpath(@__DIR__, "su2_server_driver.jl"))
    if abspath(PROGRAM_FILE) == @__FILE__
        run_su2_server(su2_server_settings)
    end
end
