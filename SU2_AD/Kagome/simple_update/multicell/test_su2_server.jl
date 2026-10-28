# Short workflow tests only: four Trotter steps per D, no large-D run/CTMRG.
using Test
include(joinpath(@__DIR__,"su2_server_driver.jl"))
BLAS.set_num_threads(2)
function compare_states(a,b)
    for (ca,cb) in zip(a.raw.cells,b.raw.cells),n in fieldnames(ServerBase.KagomeSUTensors)
        aa,bb=getfield(ca,n),getfield(cb,n)
        @test isnothing(aa)==isnothing(bb)
        isnothing(aa) && continue
        @test norm(convert(Array,aa)-convert(Array,bb))<1e-11
    end
end
@testset "server SU2 stages and exact mid-stage resume" begin
    out=mktempdir(;cleanup=false)
    seed=joinpath(@__DIR__,"su2_D4_seed.toml")
    cfg=(dims=[4,5],h=.38,chi=2,threads=2,device="cpu",ctm_iterations=2,
        ctm_tolerance=1e-6,schedule=[(dt=.01,tau=.02),(dt=.003,tau=.006)],
        truncation_tolerance=1e-6,checkpoint_every=1,progress_every=2,
        initial_state=seed,measurement_representation=:u1,out=out)
    @test sum(round(Int,s.tau/s.dt) for s in cfg.schedule)==4
    s=server_load_seed(seed);H=ServerBase.heisenberg_triangle()
    @test maximum(r.up_dimension for r in ServerBase.bond_report(s.raw))==4
    portable=joinpath(out,"seed.toml");server_export_seed(portable,s)
    compare_states(s,server_load_seed(portable))
    reference=deepcopy(s)
    for stage in cfg.schedule
        reference=ServerSU.itebd_charged(reference,H,stage.tau,stage.dt;Dmax=4)
    end
    full=joinpath(out,"full")
    final=server_evolve_D(cfg,4,seed,full,"test")
    compare_states(reference,server_load_seed(final))
    oldhash=bytes2hex(sha256(read(final)))
    server_evolve_D(cfg,4,seed,full,"test")
    @test bytes2hex(sha256(read(final)))==oldhash
    # Simulate an interruption after the down half-step + one complete body step.
    _,initialdata=ServerSU.load_charged_checkpoint(joinpath(full,"initial.jld2"))
    for step_done in (1,2)
        partial=joinpath(out,"partial$(step_done)");mkpath(partial)
        state=deepcopy(s);m=deepcopy(initialdata["metadata"])
        gate,half=ServerSU.Kernel.trotter_gate(H,cfg.schedule[1].dt)
        ServerBase.sweep!(state.raw,half,:dn;Dmax=4,trun_tol=1e-6)
        for _ in 1:step_done
            ServerSU._up_sweep!(state,gate;Dmax=4,trun_tol=1e-6)
            ServerBase.sweep!(state.raw,gate,:dn;Dmax=4,trun_tol=1e-6)
        end
        m["phase"]="body";m["stage_step"]=step_done;m["stage_seconds"]=0.
        ServerSU.save_charged_checkpoint(joinpath(partial,"latest.jld2"),state,H;metadata=m)
        resumed=server_evolve_D(cfg,4,seed,partial,"test")
        compare_states(reference,server_load_seed(resumed))
    end
    d5=server_evolve_D(cfg,5,final,joinpath(out,"D5"),"test")
    t=server_load_seed(d5)
    @test all(max(r.down_dimension,r.up_dimension)<=5 for r in ServerBase.bond_report(t.raw))
    @test isfile(joinpath(out,"D5","u1_state.toml"))
    @test_throws ErrorException server_evolve_D(cfg,5,final,joinpath(out,"D5"),"different settings")
    println("SMOKE outputs: ",out)
end
