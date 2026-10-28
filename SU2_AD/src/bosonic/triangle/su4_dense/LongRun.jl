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
    min_sweeps::Int=100
    measure_every::Int=20 # check SU lambda spectra, not a proxy energy
    checkpoint_every::Int=10
    stable_checks::Int=4
    trun_tol::Float64=1e-12
    lambda_tol_floor::Float64=1e-8
    lambda_tol_scale::Float64=1e-4
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
    !isempty(cfg.seeds) && allunique(cfg.seeds) || throw(ArgumentError("invalid seeds"))
    !isempty(cfg.schedule) && all(t->isfinite(t[1]) && t[1]>0 && t[2]>0,cfg.schedule) || throw(ArgumentError("invalid schedule"))
    issorted(first.(cfg.schedule);rev=true) || throw(ArgumentError("time steps must decrease"))
    all(>(0),(cfg.min_sweeps,cfg.measure_every,cfg.checkpoint_every,cfg.stable_checks,cfg.blas_threads)) || throw(ArgumentError("invalid sweep/thread settings"))
    0<=cfg.trun_tol<1 && isfinite(cfg.J) || throw(ArgumentError("invalid J or truncation tolerance"))
    all(x->isfinite(x) && x>=0,(cfg.lambda_tol_floor,cfg.lambda_tol_scale)) || throw(ArgumentError("invalid SU tolerance"))
    cfg.convention in (:traceless,:permutation) || throw(ArgumentError("invalid convention"))
    !isempty(cfg.ctm_chis) && all(>(0),cfg.ctm_chis) && issorted(cfg.ctm_chis) && allunique(cfg.ctm_chis) || throw(ArgumentError("invalid chi list"))
    1<=cfg.ctm_max_iterations<=50 && cfg.ctm_corner_tol>0 && cfg.ctm_chi_tol>0 || throw(ArgumentError("CTMRG permits at most 50 iterations per chi"))
end
signature(cfg)=(algorithm=:src_TensorMap_iPESS_v1,Lx=cfg.Lx,Ly=cfg.Ly,J=cfg.J,convention=cfg.convention,
    schedule=cfg.schedule,min_sweeps=cfg.min_sweeps,measure_every=cfg.measure_every,stable_checks=cfg.stable_checks,
    trun_tol=cfg.trun_tol,lambda_tol_floor=cfg.lambda_tol_floor,lambda_tol_scale=cfg.lambda_tol_scale)
run_directory(cfg,D,seed)=joinpath(abspath(cfg.output_dir),@sprintf("L%dx%d_J%.8g_%s_D%02d_seed%d",cfg.Lx,cfg.Ly,cfg.J,string(cfg.convention),D,seed))
function atomic_write(writer,path)
    tmp=path*".tmp"
    writer(tmp)
    Base.Filesystem.rename(tmp,path)
end
function write_checkpoint(path,state,cfg,D,seed,progress,history,stages)
    metadata=(J=cfg.J,convention=cfg.convention,Dmax=D,seed=seed,
        longrun=(signature=signature(cfg),progress=progress,history=history,stages=stages))
    atomic_write(p->save_state(p,state;metadata),path)
end
function row_matrix(rows)
    isempty(rows) && return zeros(0,0),String[]
    return reduce(vcat,[permutedims(Float64.(collect(values(r)))) for r in rows]),string.(collect(keys(first(rows))))
end

function run_one(cfg::LongRunConfig,D::Int,seed::Int)
    validate(cfg)
    D>0 || throw(ArgumentError("D must be positive"))
    BLAS.set_num_threads(cfg.blas_threads)
    dir=run_directory(cfg,D,seed); mkpath(dir)
    checkpoint=joinpath(dir,"checkpoint.jls")
    setting=model_settings(cfg.Lx,cfg.Ly)
    parameters=Dict("J"=>cfg.J,"K"=>0.0,"Φ"=>0.0)
    if isfile(checkpoint)
        cfg.resume || error("checkpoint exists; use resume=true or a new output directory")
        saved=load_state(checkpoint); meta=saved.metadata
        meta.longrun.signature==signature(cfg) && meta.Dmax==D && meta.seed==seed || error("checkpoint parameters differ; use a new output directory")
        state=saved.state; progress=meta.longrun.progress; history=meta.longrun.history; stages=meta.longrun.stages
        println("Resuming D=",D," seed=",seed," total_sweeps=",progress.total_sweeps)
    else
        state=init_state(cfg.Lx,cfg.Ly;D,seed)
        progress=(stage=1,sweep=0,total_sweeps=0,stable=0,block_lambda=0.0,elapsed_seconds=0.0,complete=false)
        history=NamedTuple[]; stages=NamedTuple[]
        write_checkpoint(checkpoint,state,cfg,D,seed,progress,history,stages)
    end
    open(joinpath(dir,"run.log"),"a") do log
        println(log,"\nStarted/resumed ",Dates.now()," Julia ",VERSION,"; TensorMap / src iPESS")
        while !progress.complete
            stage=progress.stage; dt,max_sweeps=cfg.schedule[stage]
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
                    final=stage==length(cfg.schedule)
                    progress=merge(progress,(stage=final ? stage : stage+1,sweep=final ? step : 0,
                        stable=0,block_lambda=0.0,complete=final))
                    write_checkpoint(checkpoint,state,cfg,D,seed,progress,history,stages)
                    break
                elseif check || step%cfg.checkpoint_every==0
                    write_checkpoint(checkpoint,state,cfg,D,seed,progress,history,stages)
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
    merge!(data,Dict("Dmax"=>D,"initial_D"=>D,"seed"=>seed,"complete"=>true,
        "su_converged"=>stages[end].converged,"converged"=>converged,
        "all_stages_converged"=>all(r->r.converged,stages),"total_sweeps"=>progress.total_sweeps,
        "history_quantity"=>"SU lambda convergence (no proxy energy)",
        "ctm_max_iterations"=>cfg.ctm_max_iterations,"ctm_elapsed_seconds"=>time()-started,
        "schedule"=>hcat(first.(cfg.schedule),last.(cfg.schedule))))
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
        "tensor_format"=>"TensorMap_ComplexSpace","energy_per_site"=>energy,"status"=>status,
        "status_description"=>"0=pending, 1=SU-and-CTMRG-converged, 2=complete/not-converged, -1=failed")
    atomic_write(p->matwrite(p,data;compress=true),joinpath(cfg.output_dir,"scan_summary.mat"))
    atomic_write(joinpath(cfg.output_dir,"scan_summary.csv")) do path
        open(path,"w") do io
            println(io,"D,seed,energy_per_site,status,total_sweeps,elapsed_seconds")
            for r in rows
                println(io,join((r.D,r.seed,r.energy,r.status,r.total_sweeps,r.elapsed_seconds),','))
            end
        end
    end
end
function run_batch(cfg::LongRunConfig=LongRunConfig())
    validate(cfg); mkpath(cfg.output_dir); rows=NamedTuple[]
    write_batch_summary(cfg,rows)
    for D in cfg.Ds,seed in cfg.seeds
        try
            push!(rows,run_one(cfg,D,seed))
        catch err
            err isa InterruptException && rethrow()
            message=sprint(showerror,err,catch_backtrace())
            @warn "D=$D seed=$seed failed; checkpoint retained" exception=(err,catch_backtrace())
            open(joinpath(cfg.output_dir,"failures.log"),"a") do io
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
