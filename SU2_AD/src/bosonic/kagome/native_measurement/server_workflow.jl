# Server orchestration only. Existing SU, restriction and CTMRG kernels are unchanged.
using TensorKit, LinearAlgebra, TOML, SHA, Dates, Printf, FileWatching, JLD2
if !isdefined(@__MODULE__, :KagomeChargedSU2)
    include(joinpath(@__DIR__, "..", "simple_update_charged_cell.jl"))
end
if !isdefined(@__MODULE__, :KagomeU1Cells)
    include(joinpath(@__DIR__, "..", "simple_update_u1_cells.jl"))
end
const ServerSU = KagomeChargedSU2
const ServerBase = ServerSU.BaseSU
const ServerU1 = KagomeU1Cells
if !isdefined(@__MODULE__,:KagomeNativeMeasurement)
    include(joinpath(@__DIR__,"KagomeNativeMeasurement.jl"))
end
const ServerNative=KagomeNativeMeasurement
const SERVER_RUN_DIR=normpath(joinpath(@__DIR__,"..","..","..","..","Kagome","simple_update","multicell"))

function server_atomic_toml(path, data)
    mkpath(dirname(path)); tmp=tempname(dirname(path))
    try
        open(io->TOML.print(io,data;sorted=true),tmp,"w")
        mv(tmp,path;force=true)
    finally
        isfile(tmp) && rm(tmp)
    end
end

# Neutral small-D seed format: no serialized TensorKit internal types. Retain
# dual arrows and domain partition, so both old/new TensorKit can reconstruct it.
function server_space_data(V)
    Dict("dual"=>isdual(V), "sectors"=>[
        Dict("twice_spin"=>Int(dim(q))-1,"multiplicity"=>Int(dim(V,q))) for q in sectors(V)])
end
function server_tensor_data(t)
    A=convert(Array,t)
    Dict("dims"=>collect(size(A)),"real"=>real.(vec(A)),"imag"=>imag.(vec(A)),
         "codomain"=>[server_space_data(V) for V in codomain(t)],
         "domain"=>[server_space_data(V) for V in domain(t)])
end
function server_restore_space(d)
    V=SU2Space((q["twice_spin"]//2=>q["multiplicity"] for q in d["sectors"])...)
    d["dual"] ? V' : V
end
function server_restore_tensor(d)
    cod=prod(server_restore_space.(d["codomain"]);init=one(SU2Space(0=>1)))
    dom=prod(server_restore_space.(d["domain"]);init=one(SU2Space(0=>1)))
    A=reshape(ComplexF64.(d["real"],d["imag"]),Tuple(d["dims"]))
    t=TensorMap(A,cod,dom;tol=1e-11)
    norm(convert(Array,t)-A)<1e-10*max(norm(A),1) || error("portable SU2 seed reconstruction failed")
    t
end
function server_export_seed(path,s;metadata=Dict{String,Any}())
    fields=fieldnames(ServerBase.KagomeSUTensors)
    data=Dict("format"=>"kagome_su2_seed_v1","anchor"=>s.anchor,
        "twice_auxiliary_spin"=>Int(2s.auxiliary_spin),"metadata"=>metadata,
        "cells"=>[Dict(string(n)=>server_tensor_data(getfield(c,n)) for n in fields
                      if !isnothing(getfield(c,n))) for c in s.raw.cells])
    server_atomic_toml(path,data)
end
function server_load_seed(path)
    if endswith(lowercase(path),".jld2")
        return first(ServerSU.load_charged_checkpoint(path))
    end
    data=TOML.parsefile(path)
    data["format"]=="kagome_su2_seed_v1" || error("unknown SU2 seed format")
    fields=fieldnames(ServerBase.KagomeSUTensors)
    cells=[ServerBase.KagomeSUTensors((haskey(c,string(n)) ? server_restore_tensor(c[string(n)]) : nothing
                                     for n in fields)...) for c in data["cells"]]
    g=ServerBase.KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3)
    ServerSU.check_charged(ServerSU.ChargedKagomeState(ServerBase.KagomeSUCell(g,cells),
        data["anchor"],data["twice_auxiliary_spin"]//2))
end
function server_bond_report(folder,s)
    rows=[Dict(string(k)=>v for (k,v) in pairs(row)) for row in ServerBase.bond_report(s.raw)]
    server_atomic_toml(joinpath(folder,"virtual_spaces.toml"),Dict("bonds"=>rows))
    open(joinpath(folder,"virtual_spaces.csv"),"w") do io
        println(io,"representative,bond,down_dimension,up_dimension,down_space,up_space")
        csvquote(v)="\""*replace(string(v),"\""=>"\"\"")*"\""
        for r in rows
            println(io,join(csvquote.([r["cell"],r["bond"],r["down_dimension"],r["up_dimension"],r["down_space"],r["up_space"]]),","))
        end
    end
end
function server_plan(cfg)
    !isempty(cfg.dims) && all(>=(4),cfg.dims) && issorted(cfg.dims) && allunique(cfg.dims) || error("invalid D sequence")
    cfg.threads>0 && cfg.chi>0 && 2<=cfg.ctm_iterations<=50 || error("invalid threads/chi/CTMRG cap")
    isfinite(cfg.h) && isfinite(cfg.ctm_tolerance) && cfg.ctm_tolerance>0 || error("invalid h or CTM tolerance")
    !isempty(cfg.schedule) && all(s->isfinite(s.dt) && s.dt>0 && isfinite(s.tau) && s.tau>=0,cfg.schedule) || error("invalid schedule")
    cfg.checkpoint_every>0 && cfg.progress_every>0 && 0<=cfg.truncation_tolerance<1 || error("invalid SU settings")
    cfg.measurement_representation in (:u1,:dense) || error("measurement_representation must be :u1 or :dense")
    isfile(cfg.initial_state) || error("missing SU2 seed: $(cfg.initial_state)")
    Dict("format"=>"su2_server_plan_v2_native","dims"=>cfg.dims,"h"=>cfg.h,"chi"=>cfg.chi,
         "TensorKit"=>string(pkgversion(TensorKit)),"JLD2"=>string(pkgversion(JLD2)),
         "threads"=>cfg.threads,"device"=>cfg.device,"representation"=>string(cfg.measurement_representation),"ctm_iterations"=>cfg.ctm_iterations,
         "ctm_tolerance"=>cfg.ctm_tolerance,"cell"=>"sqrt3xsqrt3","component"=>1,
         "initial_sha256"=>bytes2hex(sha256(read(cfg.initial_state))),
         "truncation_tolerance"=>cfg.truncation_tolerance,
         "schedule"=>[Dict("dt"=>s.dt,"tau"=>s.tau,"steps"=>round(Int,s.tau/s.dt)) for s in cfg.schedule])
end
function server_measure_final(cfg,folder,D)
    ServerNative.measure_checkpoint(joinpath(folder,"final.jld2"),joinpath(folder,"ctm");
        representation=cfg.measurement_representation,component=1,h=cfg.h,chi=cfg.chi,
        iterations=cfg.ctm_iterations,tolerance=cfg.ctm_tolerance,device=cfg.device,
        threads=cfg.threads,D=D)
end
include(joinpath(@__DIR__,"server_outputs.jl"))

# Stage restarts retain whether the initial down half-step was already applied.
# Resuming a body checkpoint therefore NEVER adds an extra boundary half-step.
function server_evolve_D(cfg,D,input,folder,plan_id)
    mkpath(folder);latest=joinpath(folder,"latest.jld2");final=joinpath(folder,"final.jld2")
    H=ServerBase.heisenberg_triangle()
    if isfile(latest)
        s,data=ServerSU.load_charged_checkpoint(latest);m=data["metadata"]
        m["plan_id"]==plan_id && m["Dmax"]==D || error("checkpoint configuration mismatch")
        println("RESUME D=$D completed=$(m["completed_stages"]) phase=$(m["phase"]) step=$(m["stage_step"])")
        if m["completed_stages"]==length(cfg.schedule) &&
           all(isfile, [final,joinpath(folder,"u1_state.toml"),joinpath(folder,"virtual_spaces.toml")])
            # Preserve hashes and let the measurement helper skip completed CTM.
            return final
        end
    else
        s=server_load_seed(input)
        m=Dict{String,Any}("plan_id"=>plan_id,"Dmax"=>D,"completed_stages"=>0,"stage_step"=>0,
            "phase"=>"stage_complete","h"=>cfg.h,"cell"=>"sqrt3xsqrt3","symmetry"=>"su2",
            "initial_source_sha256"=>bytes2hex(sha256(read(input))),"initial_source"=>abspath(input),
            "TensorKit"=>string(pkgversion(TensorKit)),"history"=>Dict[],"component"=>1)
        ServerSU.check_charged(s;Dmax=D)
        ServerSU.save_charged_checkpoint(joinpath(folder,"initial.jld2"),s,H;metadata=m)
    end
    for k in (m["completed_stages"]+1):length(cfg.schedule)
        stage=cfg.schedule[k];steps=round(Int,stage.tau/stage.dt)
        gate,half=ServerSU.Kernel.trotter_gate(H,stage.dt);stamp=time()
        continuing=m["phase"]=="body"
        start=continuing ? m["stage_step"] : 0
        if !continuing
            ServerBase.sweep!(s.raw,half,:dn;Dmax=D,trun_tol=cfg.truncation_tolerance)
            m["phase"]="body";m["stage_step"]=0;m["stage_seconds"]=0.0
            ServerSU.save_charged_checkpoint(latest,s,H;metadata=m)
        end
        accumulated=m["stage_seconds"]
        for step in (start+1):steps
            ServerSU._up_sweep!(s,gate;Dmax=D,trun_tol=cfg.truncation_tolerance)
            ServerBase.sweep!(s.raw,gate,:dn;Dmax=D,trun_tol=cfg.truncation_tolerance)
            m["stage_step"]=step
            if step%cfg.progress_every==0 || step==steps
                println("SU2 D=$D stage=$k/$(length(cfg.schedule)) dt=$(stage.dt) step=$step/$steps elapsed=$(round(time()-stamp;digits=1))s")
                flush(stdout)
            end
            if step%cfg.checkpoint_every==0
                m["stage_seconds"]=accumulated+time()-stamp
                ServerSU.save_charged_checkpoint(latest,s,H;metadata=m)
            end
        end
        ServerSU._up_sweep!(s,half;Dmax=D,trun_tol=cfg.truncation_tolerance)
        ServerSU.check_charged(s;Dmax=D)
        push!(m["history"],Dict("stage"=>k,"dt"=>stage.dt,"requested_tau"=>stage.tau,
            "actual_tau"=>steps*stage.dt,"steps"=>steps,"seconds"=>accumulated+time()-stamp))
        m["completed_stages"]=k;m["phase"]="stage_complete";m["stage_step"]=0
        ServerSU.save_charged_checkpoint(latest,s,H;metadata=m)
        ServerSU.save_charged_checkpoint(joinpath(folder,@sprintf("stage_%02d.jld2",k)),s,H;metadata=m)
    end
    ServerSU.save_charged_checkpoint(final,s,H;metadata=m)
    server_bond_report(folder,s)
    server_atomic_toml(joinpath(folder,"progress.toml"),m)
    # No nine-site torus contraction, no dense double layer, no D<=6 guard.
    u1=ServerU1.to_u1(s;component=1)
    data=ServerU1.export_u1_portable(u1)
    data["conversion"]=Dict("source_sha256"=>bytes2hex(sha256(read(final))),"component"=>1)
    server_atomic_toml(joinpath(folder,"u1_state.toml"),data)
    return final
end
function server_run_locked(cfg,plan)
    planpath=joinpath(cfg.out,"configuration.toml")
    if isfile(planpath)
        TOML.parsefile(planpath)==plan || error("output settings differ; choose a new out directory")
    else
        server_atomic_toml(planpath,plan)
    end
    plan_id=bytes2hex(sha256(read(planpath)))
    BLAS.set_num_threads(cfg.threads)
    println("SU2 server: h=$(cfg.h), sqrt3xsqrt3, D=$(cfg.dims), chi=$(cfg.chi), CTM <=$(cfg.ctm_iterations), $(cfg.device)")
    println("Full steps per D: ",sum(round(Int,s.tau/s.dt) for s in cfg.schedule),"; one final pure Julia CTMRG + observables")
    flush(stdout)
    # Reject wrong paths / missing GPU before a long SU evolution starts.
    ServerNative.prepare_device!(cfg.device)
    input=cfg.initial_state
    for D in cfg.dims
        folder=joinpath(cfg.out,"D$(D)")
        input=server_evolve_D(cfg,D,input,folder,plan_id)
        server_measure_final(cfg,folder,D)
        server_collect_outputs(cfg.out;reference_file=get(cfg,:dense_reference,joinpath(SERVER_RUN_DIR,"data","dense_reference_h038.csv")))
        ServerNative.CTM.square_J1_reclaim_device_memory!()
    end
    println("DONE: ",joinpath(cfg.out,"summary.csv"));flush(stdout)
    return cfg.out
end
function run_su2_server(cfg)
    plan=server_plan(cfg);mkpath(cfg.out)
    if Sys.isunix()
        # Kernel flock: released on SIGTERM/SIGKILL; the .lock filename may remain.
        # Never delete a lock file to try to stop a live job.
        open(joinpath(cfg.out,"writer.lock"),"a+") do io
            ccall(:flock,Cint,(Cint,Cint),fd(io),6)==0 || error("another process owns $(cfg.out)")
            try
                seekstart(io);truncate(io,0);println(io,"PID=",getpid());flush(io)
                server_run_locked(cfg,plan)
            finally
                ccall(:flock,Cint,(Cint,Cint),fd(io),8)
            end
        end
    else
        FileWatching.Pidfile.mkpidlock(joinpath(cfg.out,"writer.pid");wait=false) do
            server_run_locked(cfg,plan)
        end
    end
end
