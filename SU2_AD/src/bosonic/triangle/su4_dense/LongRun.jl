# Server orchestration only. All tensor-network algorithms are in src/.
if !isdefined(@__MODULE__,:TensorMapSU4)
    include(joinpath(@__DIR__,"TensorMapSU4.jl"))
end
module DenseSU4LongRun
using ..TensorMapSU4
using LinearAlgebra, Dates, Printf, MAT
const TM=TensorMapSU4
export LongRunConfig, run_batch, run_one, evaluate_checkpoint

const DEFAULT_OUTPUT_DIR=normpath(joinpath(@__DIR__,"../../../../SU4/simple_update_dense/long_runs_tensormap"))

Base.@kwdef struct LongRunConfig
    Ds::Vector{Int}=collect(4:16)
    seeds::Vector{Int}=[1234,5678,9012]
    Lx::Int=2
    Ly::Int=2
    J::Float64=1.0
    convention::Symbol=:traceless
    schedule::Vector{Tuple{Float64,Int}}=[(0.1,1000),(0.05,1000),(0.02,1500),
        (0.01,1500),(0.005,2000),(0.002,2000),(0.001,2500),(0.0002,3000),(0.0001,3000)]
    initialization::Symbol=:previous_D # :random retains independent initialization
    continuation_schedule::Vector{Tuple{Float64,Int}}=[(0.005,500),(0.002,800),
        (0.001,1000),(0.0002,1200),(0.0001,1500)]
    expansion_noise::Float64=1e-3
    min_sweeps::Int=100
    measure_every::Int=20 # check SU lambda spectra, not a proxy energy
    checkpoint_every::Int=10
    stable_checks::Int=4
    trun_tol::Float64=1e-12
    lambda_tol_floor::Float64=1e-8
    lambda_tol_scale::Float64=1e-3 # original itebd_iPESS stopping tolerance: dt*1e-3
    blas_threads::Int=min(8,Sys.CPU_THREADS)
    ctm_chis::Vector{Int}=[16,32,48]
    ctm_max_iterations::Int=50
    ctm_corner_tol::Float64=1e-7
    ctm_chi_tol::Float64=1e-5
    export_tensors::Bool=true
    resume::Bool=true
    output_dir::String=DEFAULT_OUTPUT_DIR
end

function validate(cfg)
    (cfg.Lx,cfg.Ly) in ((2,2),(4,4)) || throw(ArgumentError("use a 2x2 or 4x4 cell supported by the original CTMRG"))
    !isempty(cfg.Ds) && all(>(0),cfg.Ds) && allunique(cfg.Ds) || throw(ArgumentError("invalid D list"))
    cfg.initialization in (:previous_D,:random) || throw(ArgumentError("invalid initialization"))
    cfg.initialization!=:previous_D || issorted(cfg.Ds) || throw(ArgumentError("continuation requires increasing D"))
    isfinite(cfg.expansion_noise) && cfg.expansion_noise>0 || throw(ArgumentError("expansion noise must be positive"))
    !isempty(cfg.seeds) && allunique(cfg.seeds) || throw(ArgumentError("invalid seeds"))
    !isempty(cfg.schedule) && all(t->isfinite(t[1]) && t[1]>0 && t[2]>0,cfg.schedule) || throw(ArgumentError("invalid schedule"))
    issorted(first.(cfg.schedule);rev=true) || throw(ArgumentError("time steps must decrease"))
    !isempty(cfg.continuation_schedule) && all(t->isfinite(t[1]) && t[1]>0 && t[2]>0,cfg.continuation_schedule) || throw(ArgumentError("invalid continuation schedule"))
    issorted(first.(cfg.continuation_schedule);rev=true) || throw(ArgumentError("continuation time steps must decrease"))
    all(>(0),(cfg.min_sweeps,cfg.measure_every,cfg.checkpoint_every,cfg.stable_checks,cfg.blas_threads)) || throw(ArgumentError("invalid sweep/thread settings"))
    0<=cfg.trun_tol<1 && isfinite(cfg.J) || throw(ArgumentError("invalid J or truncation tolerance"))
    all(x->isfinite(x) && x>=0,(cfg.lambda_tol_floor,cfg.lambda_tol_scale)) || throw(ArgumentError("invalid SU tolerance"))
    cfg.convention in (:traceless,:permutation) || throw(ArgumentError("invalid convention"))
    !isempty(cfg.ctm_chis) && all(>(0),cfg.ctm_chis) && issorted(cfg.ctm_chis) && allunique(cfg.ctm_chis) || throw(ArgumentError("invalid chi list"))
    1<=cfg.ctm_max_iterations<=50 && cfg.ctm_corner_tol>0 && cfg.ctm_chi_tol>0 || throw(ArgumentError("CTMRG permits at most 50 iterations per chi"))
end
is_continuation(cfg,D)=cfg.initialization==:previous_D && D!=first(cfg.Ds)
run_schedule(cfg,D)=is_continuation(cfg,D) ? cfg.continuation_schedule : cfg.schedule
function signature(cfg,D=first(cfg.Ds))
    base=(algorithm=:src_TensorMap_iPESS_v1,Lx=cfg.Lx,Ly=cfg.Ly,J=cfg.J,convention=cfg.convention,
    schedule=run_schedule(cfg,D),min_sweeps=cfg.min_sweeps,measure_every=cfg.measure_every,stable_checks=cfg.stable_checks,
    trun_tol=cfg.trun_tol,lambda_tol_floor=cfg.lambda_tol_floor,lambda_tol_scale=cfg.lambda_tol_scale)
    return is_continuation(cfg,D) ? merge(base,(initialization=:previous_D,expansion_noise=cfg.expansion_noise,
        source_D=cfg.Ds[findfirst(==(D),cfg.Ds)-1])) : base
end
batch_directory(cfg)=cfg.initialization==:previous_D ? joinpath(abspath(cfg.output_dir),"continuation") : abspath(cfg.output_dir)
run_directory(cfg,D,seed;chain=cfg.initialization==:previous_D)=joinpath(
    chain ? joinpath(abspath(cfg.output_dir),"continuation") : abspath(cfg.output_dir),
    @sprintf("L%dx%d_J%.8g_%s_D%02d_seed%d",cfg.Lx,cfg.Ly,cfg.J,string(cfg.convention),D,seed))
random_initialization(D)=(method=:random,source_D=0,source_checkpoint="",noise=0.0,initial_D=D)
function atomic_write(writer,path)
    tmp=path*".tmp"
    writer(tmp)
    Base.Filesystem.rename(tmp,path)
end
function write_checkpoint(path,state,cfg,D,seed,progress,history,stages;initialization=random_initialization(D))
    metadata=(J=cfg.J,convention=cfg.convention,Dmax=D,seed=seed,initialization=initialization,
        longrun=(signature=signature(cfg,D),progress=progress,history=history,stages=stages))
    atomic_write(p->save_state(p,state;metadata),path)
end
function row_matrix(rows)
    isempty(rows) && return zeros(0,0),String[]
    return reduce(vcat,[permutedims(Float64.(collect(values(r)))) for r in rows]),string.(collect(keys(first(rows))))
end

function run_one(cfg::LongRunConfig,D::Int,seed::Int)
    validate(cfg)
    D>0 || throw(ArgumentError("D must be positive"))
    D in cfg.Ds || throw(ArgumentError("D must be present in cfg.Ds"))
    BLAS.set_num_threads(cfg.blas_threads)
    dir=run_directory(cfg,D,seed); mkpath(dir)
    checkpoint=joinpath(dir,"checkpoint.jls")
    schedule=run_schedule(cfg,D)
    setting=model_settings(cfg.Lx,cfg.Ly)
    parameters=Dict("J"=>cfg.J,"K"=>0.0,"Φ"=>0.0)
    # Reuse an existing independent run for the first D; later D values have
    # their own chain checkpoints and never overwrite independent results.
    independent_checkpoint=joinpath(run_directory(cfg,D,seed;chain=false),"checkpoint.jls")
    import_existing=!isfile(checkpoint) && cfg.initialization==:previous_D &&
        D==first(cfg.Ds) && isfile(independent_checkpoint)
    if isfile(checkpoint) || import_existing
        cfg.resume || error("checkpoint exists; use resume=true or a new output directory")
        source_checkpoint=import_existing ? independent_checkpoint : checkpoint
        saved=load_state(source_checkpoint); meta=saved.metadata
        meta.Dmax==D && meta.seed==seed || error("checkpoint parameters differ; use a new output directory")
        current_signature=signature(cfg,D)
        previous_signature=meta.longrun.signature
        # Accept only the specific correction from our earlier, overly strict
        # default to the original src tolerance. Physical parameters and the
        # evolution schedule must still match exactly.
        restored_tolerance=cfg.lambda_tol_scale==1e-3 &&
            previous_signature==merge(current_signature,(lambda_tol_scale=1e-4,))
        previous_signature==current_signature || restored_tolerance ||
            error("checkpoint parameters differ; use a new output directory")
        state=saved.state; progress=meta.longrun.progress; history=meta.longrun.history; stages=meta.longrun.stages
        initialization=import_existing ? (method=:existing_checkpoint,source_D=D,
            source_checkpoint=abspath(source_checkpoint),noise=0.0,initial_D=D) :
            get(meta,:initialization,random_initialization(D))
        if restored_tolerance
            for i in eachindex(stages)
                stage=stages[i]
                checks=filter(r->r.stage==stage.stage && cfg.min_sweeps<=r.sweep<=stage.sweeps,history)
                tol=max(cfg.lambda_tol_floor,cfg.lambda_tol_scale*stage.dt)
                converged=length(checks)>=cfg.stable_checks &&
                    all(r->r.lambda_change<=tol,last(checks,cfg.stable_checks))
                stages[i]=merge(stage,(converged=converged,))
            end
            progress=merge(progress,(stable=0,))
            println("Restored original SU tolerance dt*1e-3; retained state and sweep history.")
        end
        if restored_tolerance || import_existing
            write_checkpoint(checkpoint,state,cfg,D,seed,progress,history,stages;initialization)
        end
        println("Resuming D=",D," seed=",seed," total_sweeps=",progress.total_sweeps)
    else
        if is_continuation(cfg,D)
            source_D=cfg.Ds[findfirst(==(D),cfg.Ds)-1]
            source_checkpoint=joinpath(run_directory(cfg,source_D,seed),"checkpoint.jls")
            isfile(source_checkpoint) || error("missing previous-D checkpoint: $source_checkpoint")
            source=load_state(source_checkpoint); meta=source.metadata
            meta.Dmax==source_D && meta.seed==seed && meta.J==cfg.J && meta.convention==cfg.convention &&
                size(source.state.B)==(cfg.Lx,cfg.Ly) || error("previous-D checkpoint model parameters differ")
            meta.longrun.progress.complete || error("previous-D SU has not completed its schedule")
            state=expand_state(source.state,D;seed,noise=cfg.expansion_noise)
            initialization=(method=:previous_D,source_D=source_D,source_checkpoint=abspath(source_checkpoint),
                noise=cfg.expansion_noise,initial_D=source_D)
            println("D=",D," seed=",seed," initialized from D=",source_D," noise=",cfg.expansion_noise)
        else
            state=init_state(cfg.Lx,cfg.Ly;D,seed)
            initialization=random_initialization(D)
        end
        progress=(stage=1,sweep=0,total_sweeps=0,stable=0,block_lambda=0.0,elapsed_seconds=0.0,complete=false)
        history=NamedTuple[]; stages=NamedTuple[]
        write_checkpoint(checkpoint,state,cfg,D,seed,progress,history,stages;initialization)
    end
    open(joinpath(dir,"run.log"),"a") do log
        println(log,"\nStarted/resumed ",Dates.now()," Julia ",VERSION,"; TensorMap / src iPESS")
        while !progress.complete
            stage=progress.stage; dt,max_sweeps=schedule[stage]
            tol=max(cfg.lambda_tol_floor,cfg.lambda_tol_scale*dt)
            gates=TM.gate_RU_LD_RD(parameters,setting,dt,TM.ComplexSpace)
            for step=progress.sweep+1:max_sweeps
                started=time()
                old=deepcopy((state.lambda1,state.lambda2,state.lambda3))
                TM.triangle_update_iPESS(D,step,state.B,state.T,state.lambda1,state.lambda2,state.lambda3,gates,cfg.trun_tol)
                change=maximum(maximum(TM.check_convergence(a,b)) for (a,b) in zip((state.lambda1,state.lambda2,state.lambda3),old))
                isfinite(change) || error("non-finite SU convergence measure")
                progress=merge(progress,(sweep=step,total_sweeps=progress.total_sweeps+1,
                    block_lambda=max(progress.block_lambda,change),elapsed_seconds=progress.elapsed_seconds+time()-started))
                check=step%cfg.measure_every==0 || step==max_sweeps
                converged=false
                if check
                    stable=step>=cfg.min_sweeps && progress.block_lambda<=tol ? progress.stable+1 : 0
                    converged=stable>=cfg.stable_checks
                    push!(history,(stage=stage,dt=dt,sweep=step,total_sweeps=progress.total_sweeps,
                        lambda_change=progress.block_lambda,elapsed_seconds=progress.elapsed_seconds))
                    line=@sprintf("D=%d seed=%d dt=%.4g sweep=%d/%d dLambda=%.4e stable=%d/%d",D,seed,dt,step,max_sweeps,progress.block_lambda,stable,cfg.stable_checks)
                    println(line); flush(stdout); println(log,line); flush(log)
                    progress=merge(progress,(stable=stable,block_lambda=0.0))
                end
                if converged || step==max_sweeps
                    push!(stages,(stage=stage,dt=dt,sweeps=step,converged=converged))
                    final=stage==length(schedule)
                    progress=merge(progress,(stage=final ? stage : stage+1,sweep=final ? step : 0,
                        stable=0,block_lambda=0.0,complete=final))
                    write_checkpoint(checkpoint,state,cfg,D,seed,progress,history,stages;initialization)
                    break
                elseif check || step%cfg.checkpoint_every==0
                    write_checkpoint(checkpoint,state,cfg,D,seed,progress,history,stages;initialization)
                end
            end
        end
    end
    # The SU checkpoint is already durable if measurement is interrupted.
    started=time()
    obs=measure_state(state;J=cfg.J,convention=cfg.convention,chis=cfg.ctm_chis,
        max_iterations=cfg.ctm_max_iterations,corner_tol=cfg.ctm_corner_tol,chi_tol=cfg.ctm_chi_tol)
    data=observable_mat_data(obs)
    data["history"],data["history_columns"]=row_matrix(history)
    data["stage_summary"],data["stage_summary_columns"]=row_matrix(stages)
    converged=stages[end].converged && obs.ctm.chi_converged
    merge!(data,Dict("Dmax"=>D,"initial_D"=>initialization.initial_D,"seed"=>seed,"complete"=>true,
        "initialization"=>string(initialization.method),"initialization_mode"=>string(cfg.initialization),
        "source_D"=>initialization.source_D,"source_checkpoint"=>initialization.source_checkpoint,
        "expansion_noise"=>initialization.noise,
        "su_converged"=>stages[end].converged,"converged"=>converged,
        "all_stages_converged"=>all(r->r.converged,stages),"total_sweeps"=>progress.total_sweeps,
        "history_quantity"=>"SU lambda convergence (no proxy energy)",
        "ctm_max_iterations"=>cfg.ctm_max_iterations,"ctm_elapsed_seconds"=>time()-started,
        "schedule"=>hcat(first.(schedule),last.(schedule))))
    if cfg.export_tensors
        for (name,A) in (("B_set",state.B),("T_set",state.T),("lambda_set1",state.lambda1),("lambda_set2",state.lambda2),("lambda_set3",state.lambda3))
            data[name]=Matrix{Any}(map(t->convert(Array,t),A))
        end
    end
    matpath=joinpath(dir,"result.mat")
    atomic_write(p->matwrite(p,data;compress=true),matpath)
    save_observables(joinpath(dir,"observables"),obs)
    print_observables(obs)
    return (D=D,seed=seed,energy=obs.energy_per_site,converged=converged,complete=true,
        status=converged ? 1 : 2,matfile=matpath,total_sweeps=progress.total_sweeps,elapsed_seconds=progress.elapsed_seconds)
end

function write_batch_summary(cfg,rows)
    energy=fill(NaN,length(cfg.Ds),length(cfg.seeds)); status=zeros(Int,size(energy))
    for r in rows
        i=findfirst(==(r.D),cfg.Ds); j=findfirst(==(r.seed),cfg.seeds)
        energy[i,j]=r.energy; status[i,j]=r.status
    end
    data=Dict("D_values"=>cfg.Ds,"seeds"=>cfg.seeds,"environment"=>"ctmrg",
        "tensor_format"=>"TensorMap_ComplexSpace","initialization_mode"=>string(cfg.initialization),
        "energy_per_site"=>energy,"status"=>status,
        "status_description"=>"0=pending, 1=SU-and-CTMRG-converged, 2=complete/not-converged, -1=failed")
    atomic_write(p->matwrite(p,data;compress=true),joinpath(batch_directory(cfg),"scan_summary.mat"))
    atomic_write(joinpath(batch_directory(cfg),"scan_summary.csv")) do path
        open(path,"w") do io
            println(io,"D,seed,energy_per_site,status,total_sweeps,elapsed_seconds")
            for r in rows
                println(io,join((r.D,r.seed,r.energy,r.status,r.total_sweeps,r.elapsed_seconds),','))
            end
        end
    end
end
function run_batch(cfg::LongRunConfig=LongRunConfig())
    validate(cfg); mkpath(batch_directory(cfg)); rows=NamedTuple[]
    write_batch_summary(cfg,rows)
    for D in cfg.Ds,seed in cfg.seeds
        try
            push!(rows,run_one(cfg,D,seed))
        catch err
            err isa InterruptException && rethrow()
            message=sprint(showerror,err,catch_backtrace())
            @warn "D=$D seed=$seed failed; checkpoint retained" exception=(err,catch_backtrace())
            open(joinpath(batch_directory(cfg),"failures.log"),"a") do io
                println(io,Dates.now()," D=",D," seed=",seed,"\n",message)
            end
            push!(rows,(D=D,seed=seed,energy=NaN,converged=false,complete=false,status=-1,
                matfile="",total_sweeps=0,elapsed_seconds=NaN))
        end
        write_batch_summary(cfg,rows); GC.gc()
    end
    return rows
end
"""Re-evaluate a saved TensorMap checkpoint with the original CTMRG/spin routines."""
function evaluate_checkpoint(path::AbstractString;J=nothing,convention=nothing,
        output_prefix=splitext(path)[1]*"_ctmrg_observables",kwargs...)
    data=load_state(path)
    J=isnothing(J) ? get(data.metadata,:J,1.0) : J
    convention=isnothing(convention) ? get(data.metadata,:convention,:traceless) : convention
    obs=measure_state(data.state;J,convention,kwargs...)
    mkpath(dirname(abspath(output_prefix)))
    save_observables(output_prefix,obs)
    matwrite(output_prefix*".mat",observable_mat_data(obs);compress=true)
    print_observables(obs)
    return obs
end
end # module DenseSU4LongRun
