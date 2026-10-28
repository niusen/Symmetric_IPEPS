# One SU step and two CTMRG iterations; exercises native I/O and restart only.
using Test
include(joinpath(@__DIR__,"charged_su2_convergence_driver.jl"))
include(joinpath(@__DIR__,"su2_server_driver.jl"))
BLAS.set_num_threads(2)
@testset "pure Julia server: one-step I/O and resume" begin
    s=convergence_initial_state(2,:integer_singlet,1234)
    folder=mktempdir(;cleanup=false)
    seed=joinpath(folder,"seed.toml");server_export_seed(seed,s)
    cfg=(dims=[4],h=.38,chi=2,threads=2,device="cpu",ctm_iterations=2,ctm_tolerance=1e-6,
        schedule=[(dt=.01,tau=.01)],truncation_tolerance=1e-6,checkpoint_every=1,
        progress_every=1,initial_state=seed,measurement_representation=:u1,out=joinpath(folder,"job"))
    run_su2_server(cfg)
    report=joinpath(cfg.out,"D4","ctm","observables.toml")
    data=TOML.parsefile(report)
    @test data["ctm"]["iterations"]<=2
    @test isfinite(data["energy_per_spin"])
    @test data["settings"]["method"]=="pure Julia TensorKit factorized CTMRG"
    for name in ("summary.csv","summary.toml","summary.json","summary.txt","site_magnetization.csv",
                 "triangle_observables.csv","bond_energies.csv","virtual_spaces.csv")
        @test isfile(joinpath(cfg.out,name))
    end
    digest=bytes2hex(sha256(read(report)))
    run_su2_server(cfg)
    @test bytes2hex(sha256(read(report)))==digest
    @test_throws ErrorException run_su2_server(merge(cfg,(chi=3,)))
    println("Native server smoke output: ",cfg.out)
end
