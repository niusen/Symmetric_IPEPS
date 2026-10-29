# Server orchestration only. Reuse all existing U1 SU / CTMRG kernels.
if !isdefined(@__MODULE__,:server_load_seed)
    include(joinpath(@__DIR__,"server_workflow.jl"))
end
include("u1_seed_io.jl")

function u1_server_plan(cfg)
    !isempty(cfg.dims) && all(>=(4),cfg.dims) && issorted(cfg.dims) && allunique(cfg.dims) || error("invalid D sequence")
    cfg.threads>0 && cfg.chi>0 && 2<=cfg.ctm_iterations<=50 || error("invalid threads/chi/CTMRG cap")
    isfinite(cfg.h) && isfinite(cfg.ctm_tolerance) && cfg.ctm_tolerance>0 || error("invalid h/tolerance")
    cfg.checkpoint_every>0 && cfg.progress_every>0 && 0<=cfg.truncation_tolerance<1 || error("invalid SU settings")
    !isempty(cfg.schedule) && all(s->isfinite(s.dt) && s.dt>0 && isfinite(s.tau) &&
        round(Int,s.tau/s.dt)>0,cfg.schedule) || error("each SU stage needs at least one step")
    isfile(cfg.initial_state) || error("missing optimized U1 seed: $(cfg.initial_state)")
    seed=load_u1_seed(cfg.initial_state);ServerU1.check_u1(seed;Dmax=first(cfg.dims))
    seed.geometry.pattern==:sqrt3xsqrt3 && sum(values(seed.auxiliary_charges))==1 || error("expected sqrt3 cell in the 1/9 sector")
    Dict("format"=>"u1_server_plan_v1","dims"=>cfg.dims,"h"=>cfg.h,"chi"=>cfg.chi,
        "TensorKit"=>string(pkgversion(TensorKit)),"JLD2"=>string(pkgversion(JLD2)),
        "threads"=>cfg.threads,"device"=>cfg.device,"ctm_iterations"=>cfg.ctm_iterations,
        "ctm_tolerance"=>cfg.ctm_tolerance,"cell"=>"sqrt3xsqrt3","symmetry"=>"u1",
        "initial_sha256"=>bytes2hex(sha256(read(cfg.initial_state))),
        "truncation_tolerance"=>cfg.truncation_tolerance,
        "schedule"=>[Dict("dt"=>s.dt,"requested_tau"=>s.tau,"steps"=>round(Int,s.tau/s.dt)) for s in cfg.schedule])
end

function u1_server_evolve_D(cfg,D,input,folder,plan_id)
    mkpath(folder);latest=joinpath(folder,"latest.jld2");final=joinpath(folder,"final.jld2")
    H=ServerU1.u1_heisenberg_triangle()
    # Uniform h*sum(Sz) is a scalar in this fixed sector. Use exchange gates,
    # and add the field contribution exactly once in the final measurement.
    if isfile(latest)
        s,data=ServerU1.load_u1_checkpoint(latest);m=data["metadata"]
        m["plan_id"]==plan_id && m["Dmax"]==D || error("U1 checkpoint settings mismatch")
        println("RESUME U1 D=$D completed=$(m["completed_stages"]) phase=$(m["phase"]) step=$(m["stage_step"])")
        if m["completed_stages"]==length(cfg.schedule) &&
           all(isfile,[final,joinpath(folder,"virtual_spaces.toml"),joinpath(folder,"progress.toml")])
            return final
        end
    else
        s=load_u1_seed(input);ServerU1.check_u1(s;Dmax=D)
        m=Dict{String,Any}("plan_id"=>plan_id,"Dmax"=>D,"completed_stages"=>0,"stage_step"=>0,
            "phase"=>"stage_complete","h"=>cfg.h,"cell"=>"sqrt3xsqrt3","symmetry"=>"u1",
            "initial_source_sha256"=>bytes2hex(sha256(read(input))),"initial_source"=>abspath(input),
            "TensorKit"=>string(pkgversion(TensorKit)),"history"=>Dict[])
        ServerU1.save_u1_checkpoint(joinpath(folder,"initial.jld2"),s,H;metadata=m)
    end
    charges=copy(s.auxiliary_charges)
    for k in (m["completed_stages"]+1):length(cfg.schedule)
        stage=cfg.schedule[k];steps=round(Int,stage.tau/stage.dt)
        gate,half=ServerU1.Kernels.trotter_gate(H,stage.dt);stamp=time()
        continuing=m["phase"]=="body";start=continuing ? m["stage_step"] : 0
        if !continuing
            ServerU1._sweep!(s,half,:dn;Dmax=D,trun_tol=cfg.truncation_tolerance)
            m["phase"]="body";m["stage_step"]=0;m["stage_seconds"]=0.0
            ServerU1.save_u1_checkpoint(latest,s,H;metadata=m)
        end
        accumulated=m["stage_seconds"]
        for step in (start+1):steps
            ServerU1._sweep!(s,gate,:up;Dmax=D,trun_tol=cfg.truncation_tolerance)
            ServerU1._sweep!(s,gate,:dn;Dmax=D,trun_tol=cfg.truncation_tolerance)
            m["stage_step"]=step
            if step%cfg.progress_every==0 || step==steps
                println("U1 D=$D stage=$k/$(length(cfg.schedule)) dt=$(stage.dt) step=$step/$steps elapsed=$(round(time()-stamp;digits=1))s")
                flush(stdout)
            end
            if step%cfg.checkpoint_every==0
                m["stage_seconds"]=accumulated+time()-stamp
                ServerU1.save_u1_checkpoint(latest,s,H;metadata=m)
            end
        end
        # Same Trotter sequence as itebd_u1. A resumed body does NOT receive
        # another initial down half-step; the final up half-step runs once.
        ServerU1._sweep!(s,half,:up;Dmax=D,trun_tol=cfg.truncation_tolerance)
        ServerU1.check_u1(s;Dmax=D)
        s.auxiliary_charges==charges || error("auxiliary charges changed")
        push!(m["history"],Dict("stage"=>k,"dt"=>stage.dt,"requested_tau"=>stage.tau,
            "actual_tau"=>steps*stage.dt,"steps"=>steps,"seconds"=>accumulated+time()-stamp))
        m["completed_stages"]=k;m["phase"]="stage_complete";m["stage_step"]=0
        ServerU1.save_u1_checkpoint(latest,s,H;metadata=m)
        ServerU1.save_u1_checkpoint(joinpath(folder,@sprintf("stage_%02d.jld2",k)),s,H;metadata=m)
    end
    ServerU1.save_u1_checkpoint(final,s,H;metadata=m)
    # KagomeSUCell is only the existing geometry/tensor storage adapter here;
    # bond_report does not impose symmetry or modify the U1 tensors.
    raw=ServerBase.KagomeSUCell(s.geometry,s.cells)
    server_bond_report(folder,(raw=raw,))
    server_atomic_toml(joinpath(folder,"progress.toml"),m)
    server_atomic_toml(joinpath(folder,"u1_state.toml"),ServerU1.export_u1_portable(s))
    return final
end

function u1_server_run_locked(cfg,plan)
    planpath=joinpath(cfg.out,"configuration.toml")
    if isfile(planpath)
        TOML.parsefile(planpath)==plan || error("settings/seed/version changed; choose a new out directory")
    else
        server_atomic_toml(planpath,plan)
    end
    plan_id=bytes2hex(sha256(read(planpath)));BLAS.set_num_threads(cfg.threads)
    println("U1 server: h=$(cfg.h), sqrt3xsqrt3, D=$(cfg.dims), chi=$(cfg.chi), CTM <=$(cfg.ctm_iterations), $(cfg.device)")
    println("Full steps per D: ",sum(round(Int,s.tau/s.dt) for s in cfg.schedule),"; ONE final CTMRG + observables per D")
    println("Chain: optimized U1 D4 seed -> D4 -> D5 -> ...; no noise, no imposed SU2 multiplets")
    flush(stdout);ServerNative.prepare_device!(cfg.device)
    input=cfg.initial_state
    for D in cfg.dims
        folder=joinpath(cfg.out,"D$(D)")
        input=u1_server_evolve_D(cfg,D,input,folder,plan_id)
        ServerNative.measure_checkpoint(input,joinpath(folder,"ctm");representation=:u1,h=cfg.h,
            chi=cfg.chi,iterations=cfg.ctm_iterations,tolerance=cfg.ctm_tolerance,
            device=cfg.device,threads=cfg.threads,D=D)
        server_collect_outputs(cfg.out;reference_file=cfg.dense_reference,evolution_symmetry=:u1)
        ServerNative.CTM.square_J1_reclaim_device_memory!()
    end
    println("DONE: ",joinpath(cfg.out,"summary.csv"));flush(stdout)
    cfg.out
end

function run_u1_server(cfg)
    plan=u1_server_plan(cfg);mkpath(cfg.out)
    if Sys.isunix()
        open(joinpath(cfg.out,"writer.lock"),"a+") do io
            # Kernel lock dies with the process. A remaining filename is harmless.
            if ccall(:flock,Cint,(Cint,Cint),fd(io),6)!=0
                seekstart(io);owner=strip(read(io,String))
                error("another live U1 job owns $(cfg.out): $owner; stop that job first, do not delete its lock")
            end
            try
                seekstart(io);truncate(io,0);println(io,"PID=",getpid());flush(io)
                u1_server_run_locked(cfg,plan)
            finally
                ccall(:flock,Cint,(Cint,Cint),fd(io),8)
            end
        end
    else
        FileWatching.Pidfile.mkpidlock(joinpath(cfg.out,"writer.pid");wait=false) do
            u1_server_run_locked(cfg,plan)
        end
    end
end
