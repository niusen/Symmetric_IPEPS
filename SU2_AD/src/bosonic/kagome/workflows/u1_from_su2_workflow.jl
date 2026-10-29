# Reuse the existing SU2 checkpoint reader, U1 restriction/SU and Julia CTMRG.
if !isdefined(@__MODULE__, :server_load_seed)
    include(joinpath(@__DIR__, "..", "native_measurement", "server_workflow.jl"))
end

function u1_trial_bond_rows(state)
    [Dict("representative"=>r,"bond"=>b,
          "down_dimension"=>dim(space(getfield(c,ServerBase.B_FIELDS[b]),1)),
          "up_dimension"=>dim(space(getfield(c,ServerBase.B_FIELDS[b]),2)),
          "down_space"=>string(space(getfield(c,ServerBase.B_FIELDS[b]),1)),
          "up_space"=>string(space(getfield(c,ServerBase.B_FIELDS[b]),2)))
     for (r,c) in enumerate(state.cells) for b in 1:3]
end

"""Project one charged SU2 checkpoint and continue with genuine U1 Simple Update.

The auxiliary component is selected ONCE, before evolution. Its 1D charge leg
is kept throughout U1 SU; SU2 multiplet degeneracies are no longer enforced.
The uniform field is a scalar within this fixed-total-Sz sector, so the SU gate
contains exchange only. The energy measurement includes h*sum(Sz) exactly once.
This comparison tests energy relaxation within a sector, not plateau stability.
"""
function run_u1_from_su2(cfg)
    cfg.Dmax>0 && cfg.threads>0 && cfg.chi>0 || error("invalid dimensions/threads")
    2<=cfg.ctm_iterations<=50 && cfg.ctm_tolerance>0 || error("invalid CTMRG settings")
    isfinite(cfg.h) && 0<=cfg.truncation_tolerance<1 || error("invalid h/truncation")
    !isempty(cfg.schedule) && all(s->isfinite(s.dt) && s.dt>0 && isfinite(s.tau) &&
        round(Int,s.tau/s.dt)>0,cfg.schedule) || error("each SU stage needs at least one step")
    ispath(cfg.out) && error("output already exists; choose a new out directory")
    BLAS.set_num_threads(cfg.threads)
    ServerNative.prepare_device!("cpu")
    source=server_load_seed(cfg.initial_state)
    ServerSU.check_charged(source;Dmax=cfg.Dmax)
    state=ServerU1.to_u1(source;component=cfg.component)
    ServerU1.check_u1(state;Dmax=cfg.Dmax)
    initial_charges=copy(state.auxiliary_charges)
    H=ServerU1.u1_heisenberg_triangle()
    metadata=Dict{String,Any}("method"=>"SU2 projection followed by U1 Simple Update",
        "source"=>abspath(cfg.initial_state),"source_sha256"=>bytes2hex(sha256(read(cfg.initial_state))),
        "Dmax"=>cfg.Dmax,"h"=>cfg.h,"component"=>cfg.component,"chi"=>cfg.chi,
        "device"=>"cpu","threads"=>cfg.threads,"TensorKit"=>string(pkgversion(TensorKit)),
        "ctm_iterations"=>cfg.ctm_iterations,"ctm_tolerance"=>cfg.ctm_tolerance,
        "truncation_tolerance"=>cfg.truncation_tolerance,"completed_stages"=>0,
        "schedule"=>[Dict("dt"=>s.dt,"tau"=>s.tau,"steps"=>round(Int,s.tau/s.dt)) for s in cfg.schedule])
    mkpath(cfg.out)
    server_atomic_toml(joinpath(cfg.out,"configuration.toml"),metadata)
    initial_file=joinpath(cfg.out,"initial_u1.jld2")
    ServerU1.save_u1_checkpoint(initial_file,state,H;metadata)
    ServerNative.write_csv(joinpath(cfg.out,"initial_virtual_spaces.csv"),u1_trial_bond_rows(state))
    measure(file,folder)=ServerNative.measure_checkpoint(file,joinpath(cfg.out,folder);
        representation=:u1,h=cfg.h,chi=cfg.chi,iterations=cfg.ctm_iterations,
        tolerance=cfg.ctm_tolerance,device="cpu",threads=cfg.threads,D=cfg.Dmax)
    println("Projected SU2 -> U1: D<=",cfg.Dmax,"; fixed auxiliary charges = ",initial_charges)
    flush(stdout)
    before=measure(initial_file,"initial_ctm")
    rows=Dict[Dict("stage"=>0,"energy_per_spin"=>before["energy_per_spin"],
        "m_normalized"=>before["m_normalized"],"ctm_converged"=>before["ctm"]["converged"],
        "ctm_error"=>before["ctm"]["error"])]
    for (k,stage) in enumerate(cfg.schedule)
        started=time()
        callback=(_,step,steps)->begin
            if step%100==0 || step==steps
                println("U1 D=$(cfg.Dmax) stage=$k/$(length(cfg.schedule)) dt=$(stage.dt) step=$step/$steps elapsed=$(round(time()-started;digits=1))s")
                flush(stdout)
            end
        end
        state=ServerU1.itebd_u1(state,H,stage.tau,stage.dt;Dmax=cfg.Dmax,
            trun_tol=cfg.truncation_tolerance,callback)
        state.auxiliary_charges==initial_charges || error("auxiliary charge changed")
        metadata["completed_stages"]=k
        ServerU1.save_u1_checkpoint(joinpath(cfg.out,@sprintf("stage_%02d.jld2",k)),state,H;metadata)
    end
    final_file=joinpath(cfg.out,"final_u1.jld2")
    ServerU1.save_u1_checkpoint(final_file,state,H;metadata)
    server_atomic_toml(joinpath(cfg.out,"u1_state.toml"),ServerU1.export_u1_portable(state))
    ServerNative.write_csv(joinpath(cfg.out,"final_virtual_spaces.csv"),u1_trial_bond_rows(state))
    after=measure(final_file,"final_ctm")
    push!(rows,Dict("stage"=>length(cfg.schedule),"energy_per_spin"=>after["energy_per_spin"],
        "m_normalized"=>after["m_normalized"],"ctm_converged"=>after["ctm"]["converged"],
        "ctm_error"=>after["ctm"]["error"]))
    report=Dict("initial"=>before,"final"=>after,
        "delta_energy_per_spin"=>after["energy_per_spin"]-before["energy_per_spin"],
        "both_ctm_converged"=>before["ctm"]["converged"] && after["ctm"]["converged"],
        "configuration"=>metadata)
    ServerNative.write_csv(joinpath(cfg.out,"comparison.csv"),rows)
    server_atomic_toml(joinpath(cfg.out,"comparison.toml"),report)
    println("U1 comparison: delta E = ",report["delta_energy_per_spin"],
        "; both CTMRG converged = ",report["both_ctm_converged"])
    println("Results: ",abspath(cfg.out));flush(stdout)
    return state,report
end
