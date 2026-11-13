Base.@kwdef struct SixRunConfig
    backend::Symbol=:dense
    Dmax::Int=backend==:su4 ? 7 : 4
    cell::Tuple{Int,Int}=(2,2)
    J::Float64=1.0
    seed::Int=1234
    virtual_space::Any=nothing
    schedule::Vector{Tuple{Float64,Int}}=[(0.1,1000),(0.05,1000),(0.02,1500),
        (0.01,1500),(0.005,2000),(0.002,2000),(0.001,2500)]
    min_sweeps::Int=100
    check_every::Int=20
    checkpoint_every::Int=10
    stable_checks::Int=4
    trun_tol::Float64=1e-12
    blas_threads::Int=8
    chis::Vector{Int}=[16,32,48]
    ctm_max_iterations::Int=50
    corner_tol::Float64=1e-7
    chi_tol::Float64=1e-5
    onsite::Bool=true
    output_dir::String=normpath(joinpath(@__DIR__,"../../../../SU4/triangle_six/results"))
end

function validate_config(cfg)
    cfg.backend in (:dense,:su4) && cfg.Dmax>0 && isfinite(cfg.J) || error("Invalid backend/Dmax/J")
    setup_cell(cfg.cell...)
    !isempty(cfg.schedule) && all(t->isfinite(t[1]) && t[1]>0 && t[2]>0,cfg.schedule) &&
        issorted(first.(cfg.schedule);rev=true) || error("Invalid SU schedule")
    all(>(0),(cfg.min_sweeps,cfg.check_every,cfg.checkpoint_every,cfg.stable_checks,cfg.blas_threads)) || error("Invalid sweep/thread settings")
    0<=cfg.trun_tol<1 && cfg.corner_tol>0 && cfg.chi_tol>0 || error("Invalid tolerances")
    1<=cfg.ctm_max_iterations<=50 || error("CTMRG maximum must be <=50")
    !isempty(cfg.chis) && all(>(0),cfg.chis) && issorted(cfg.chis) && allunique(cfg.chis) || error("Invalid chi sequence")
    if cfg.backend==:su4
        V=isnothing(cfg.virtual_space) ? default_virtual_space() : cfg.virtual_space
        sectortype(V)==SU4Irrep && dim(V)<=cfg.Dmax || error("SU(4) initial virtual space exceeds Dmax or has the wrong sector type")
    end
end

function run_signature(cfg)
    return (;model=:SU4_antisymmetric_two,backend=cfg.backend,Dmax=cfg.Dmax,cell=cfg.cell,J=cfg.J,
        seed=cfg.seed,virtual_space=cfg.backend==:su4 ? repr(isnothing(cfg.virtual_space) ? default_virtual_space() : cfg.virtual_space) : "dense",
        schedule=cfg.schedule,min_sweeps=cfg.min_sweeps,check_every=cfg.check_every,
        stable_checks=cfg.stable_checks,trun_tol=cfg.trun_tol,lambda_tol_scale=1e-3)
end
function atomic_write(writer,path)
    tmp=path*".tmp"
    writer(tmp)
    Base.Filesystem.rename(tmp,path)
end
function run_directory(cfg)
    return joinpath(abspath(cfg.output_dir),@sprintf("L%dx%d_J%.8g_%s_6_D%02d_seed%d",
        cfg.cell...,cfg.J,string(cfg.backend),cfg.Dmax,cfg.seed))
end
function save_checkpoint(path,state,signature,progress,history,stages)
    atomic_write(p->serialize(p,(format=:triangle_su4_six_v1,state=state,signature=signature,
        progress=progress,history=history,stages=stages)),path)
end
function load_checkpoint(path)
    saved=deserialize(path)
    hasproperty(saved,:format) && saved.format==:triangle_su4_six_v1 || error("Expected a six-irrep checkpoint; four-irrep checkpoints cannot be resumed")
    return saved
end
row_matrix(rows)=isempty(rows) ? zeros(0,0) : reduce(vcat,[permutedims(Float64.(collect(values(r)))) for r in rows])

"""Restartable single-D simulation using the old iPESS update and CTMRG kernels."""
function run_simulation(cfg::SixRunConfig=SixRunConfig())
    validate_config(cfg); BLAS.set_num_threads(cfg.blas_threads)
    model=six_model(backend=cfg.backend,J=cfg.J)
    dir=run_directory(cfg); mkpath(dir); checkpoint=joinpath(dir,"checkpoint.jls")
    signature=run_signature(cfg)
    if isfile(checkpoint)
        saved=load_checkpoint(checkpoint)
        saved.signature==signature || error("SU parameters differ from checkpoint: $checkpoint")
        state=saved.state; progress=saved.progress; history=saved.history; stages=saved.stages
        println("Resuming six-irrep ",cfg.backend," total_sweeps=",progress.total_sweeps)
    else
        state=initial_state(backend=cfg.backend,D=cfg.Dmax,cell=cfg.cell,seed=cfg.seed,virtual_space=cfg.virtual_space)
        progress=(stage=1,sweep=0,total_sweeps=0,stable=0,block_lambda=0.0,elapsed_seconds=0.0,complete=false)
        history=NamedTuple[]; stages=NamedTuple[]
        save_checkpoint(checkpoint,state,signature,progress,history,stages)
    end
    open(joinpath(dir,"run.log"),"a") do log
        println(log,Dates.now()," Started/resumed SU4 six-irrep; backend=",cfg.backend); flush(log)
        while !progress.complete
            stage=progress.stage; dt,max_sweeps=cfg.schedule[stage]
            gates=triangle_gates(model,dt;cell=cfg.cell)
            for step=progress.sweep+1:max_sweeps
                started=time(); old=deepcopy((state.lambda1,state.lambda2,state.lambda3))
                triangle_update_iPESS(cfg.Dmax,step,state.B,state.T,state.lambda1,state.lambda2,state.lambda3,gates,cfg.trun_tol)
                change=maximum(maximum(check_convergence(a,b)) for (a,b) in zip((state.lambda1,state.lambda2,state.lambda3),old))
                isfinite(change) && all(t->isfinite(norm(t)) && norm(t)>0,(state.B...,state.T...)) || error("Non-finite or zero state after SU")
                actual_D=maximum(dim(space(b,k)) for b in state.B for k=1:3)
                actual_D<=cfg.Dmax || error("Truncation exceeded total dimension Dmax: actual=$actual_D")
                progress=merge(progress,(sweep=step,total_sweeps=progress.total_sweeps+1,
                    block_lambda=max(progress.block_lambda,change),elapsed_seconds=progress.elapsed_seconds+time()-started))
                check=step%cfg.check_every==0 || step==max_sweeps
                converged=false
                if check
                    stable=step>=cfg.min_sweeps && progress.block_lambda<=max(1e-8,dt*1e-3) ? progress.stable+1 : 0
                    converged=stable>=cfg.stable_checks
                    push!(history,(stage=stage,dt=dt,sweep=step,total_sweeps=progress.total_sweeps,
                        lambda_change=progress.block_lambda,elapsed_seconds=progress.elapsed_seconds,actual_D=actual_D))
                    line=@sprintf("six-irrep %s Dmax=%d dt=%.4g sweep=%d/%d dLambda=%.5e stable=%d/%d actual_D=%d",
                        string(cfg.backend),cfg.Dmax,dt,step,max_sweeps,progress.block_lambda,stable,cfg.stable_checks,actual_D)
                    println(line); flush(stdout); println(log,line); flush(log)
                    progress=merge(progress,(stable=stable,block_lambda=0.0))
                end
                if converged || step==max_sweeps
                    push!(stages,(stage=stage,dt=dt,sweeps=step,converged=converged))
                    final=stage==length(cfg.schedule)
                    progress=merge(progress,(stage=final ? stage : stage+1,sweep=final ? step : 0,
                        stable=0,block_lambda=0.0,complete=final))
                    save_checkpoint(checkpoint,state,signature,progress,history,stages)
                    break
                elseif check || step%cfg.checkpoint_every==0
                    save_checkpoint(checkpoint,state,signature,progress,history,stages)
                end
            end
        end
        println(log,Dates.now()," SU complete; CTMRG/observables start (see main stdout log)"); flush(log)
    end
    obs=measure_state(state;J=cfg.J,chis=cfg.chis,max_iterations=cfg.ctm_max_iterations,
        corner_tol=cfg.corner_tol,chi_tol=cfg.chi_tol,onsite=cfg.onsite)
    data=observable_mat_data(obs)
    merge!(data,Dict("Dmax"=>cfg.Dmax,"seed"=>cfg.seed,"complete"=>true,
        "su_converged"=>last(stages).converged,"converged"=>last(stages).converged && obs.ctm.chi_converged,
        "total_sweeps"=>progress.total_sweeps,"su_elapsed_seconds"=>progress.elapsed_seconds,
        "schedule"=>hcat(first.(cfg.schedule),last.(cfg.schedule)),
        "history"=>row_matrix(history),"history_columns"=>string.(collect(keys(first(history)))),
        "stage_summary"=>row_matrix(stages),"stage_summary_columns"=>string.(collect(keys(first(stages)))),
        "virtual_sector_structure"=>[repr(space(state.B[x,y],k)) for x=1:cfg.cell[1],y=1:cfg.cell[2],k=1:3]))
    atomic_write(p->matwrite(p,data;compress=true),joinpath(dir,"result.mat"))
    save_observables(joinpath(dir,"observables"),obs); print_observables(obs)
    return (;state,obs,checkpoint,matfile=joinpath(dir,"result.mat"),total_sweeps=progress.total_sweeps)
end

function evaluate_checkpoint(path;output_prefix=splitext(path)[1]*"_ctmrg_observables",kwargs...)
    saved=load_checkpoint(path)
    obs=measure_state(saved.state;J=saved.signature.J,kwargs...)
    mkpath(dirname(abspath(output_prefix)))
    atomic_write(p->matwrite(p,observable_mat_data(obs);compress=true),output_prefix*".mat")
    save_observables(output_prefix,obs); print_observables(obs)
    return obs
end
