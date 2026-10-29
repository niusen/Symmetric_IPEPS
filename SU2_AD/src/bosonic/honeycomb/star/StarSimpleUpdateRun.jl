module HoneycombStarRun
using TOML, Dates, LinearAlgebra, JLD2
include("StarRunConfig.jl")
include("../Honeycomb_SU4.jl")
using .HoneycombSU4
using TensorKit, SUNRepresentations

export read_star_config, describe_star_run, run_star_server, measure_star_server
const SCHEMA=1

function _atomic_write(writer,path)
    tmp=tempname(dirname(path);cleanup=false);
    try
        writer(tmp);
        Base.Filesystem.rename(tmp,path); # Same directory: atomic replacement on the server.
    finally
        isfile(tmp) && rm(tmp);
    end
end

function _csv(path,rows)
    isempty(rows) && return;
    _atomic_write(path) do tmp
        open(tmp,"w") do io
            println(io,join(keys(first(rows)),","));
            for r in rows;println(io,join(values(r),","));end
        end
    end
end

function _locked(f,folder)
    mkpath(folder);lock=joinpath(folder,".star_su_lock");
    try
        mkdir(lock);
    catch
        isdir(lock) || rethrow();
        error("Output is locked: $lock. After confirming that the old job has stopped, remove this lock directory and resume.");
    end
    owner=joinpath(lock,"owner.txt");
    try
        write(owner,"pid=$(getpid()) host=$(get(ENV,"HOSTNAME",get(ENV,"COMPUTERNAME","unknown"))) time=$(now())\n");
        return f();
    finally
        isfile(owner) && rm(owner);
        isdir(lock) && rm(lock); # Empty directory only; never recursively remove user data.
    end
end

function _spectra(lx,ly)
    [b.axis==:horizontal ? lx[b.site2...] : ly[b.site1...] for b in honeycomb_bonds(size(lx)...)];
end

function _signature(lx,ly)
    [Dict(Tuple(Int.(SUNRepresentations.dynkin_label(c)))=>abs2.(diag(b))
        for (c,b) in blocks(s)) for s in _spectra(lx,ly)];
end

function _change(a,b)
    error=0.;
    for (x,y) in zip(a,b)
        keys(x)==keys(y) || return Inf;
        for k in keys(x)
            length(x[k])==length(y[k]) || return Inf;
            error=max(error,norm(x[k]-y[k]));
        end
    end
    return error
end

function _validate(T,lx,ly,cap)
    for t in T
        n=norm(t);
        isfinite(n) && n>0 || error("nonfinite or zero site tensor");
    end
    for s in _spectra(lx,ly)
        0<dim(space(s,1))<=cap || error("invalid real-bond dimension");
        all(v->isfinite(v) && v>=0,(v for (_,b) in blocks(s) for v in diag(b))) || error("invalid bond weights");
        isfinite(norm(s)) && norm(s)>0 || error("zero bond weights");
    end
end

function _sectors(lx,ly)
    rows=NamedTuple[];
    for (bond,s) in zip(honeycomb_bonds(size(lx)...),_spectra(lx,ly))
        total=sum(dim(c)*sum(abs2,abs2.(diag(b))) for (c,b) in blocks(s));
        for (c,b) in blocks(s)
            push!(rows,(direction=String(bond.direction),x=bond.site1[1],y=bond.site1[2],
                D=dim(space(s,1)),dynkin=join(Int.(SUNRepresentations.dynkin_label(c)),":"),
                irrep_dim=Int(dim(c)),multiplicity=size(b,1),
                schmidt_weight=dim(c)*sum(abs2,abs2.(diag(b)))/total));
        end
    end
    return rows
end

function _checkpoint(c,T,lx,ly,p,history;archive=false)
    latest=joinpath(c.output_dir,"state_latest.jld2");
    sectors=_sectors(lx,ly);
    _atomic_write(latest) do tmp
        jldsave(tmp;schema_version=SCHEMA,physics=star_physics(c),T_set=T,
            lambdax_set=lx,lambday_set=ly,progress=p,history,sectors,
            julia_version=string(VERSION),tensorkit_version=string(Base.pkgversion(TensorKit)));
    end
    if archive
        path=joinpath(c.output_dir,"state_step$(lpad(string(p["total_steps"]),8,'0')).jld2");
        _atomic_write(tmp->cp(latest,tmp;force=true),path);
    end
    _csv(joinpath(c.output_dir,"evolution.csv"),history);
    _csv(joinpath(c.output_dir,"bond_sectors_latest.csv"),sectors);
    _atomic_write(joinpath(c.output_dir,"status.toml")) do tmp
        open(io->TOML.print(io,Dict("progress"=>p,"julia_version"=>string(VERSION),
            "tensorkit_version"=>string(Base.pkgversion(TensorKit)),"saved_at"=>string(now()))),tmp,"w");
    end
    return latest
end

function _load(c)
    path=joinpath(c.output_dir,"state_latest.jld2");
    isfile(path) || error("No checkpoint to resume: $path");
    s=load(path);
    s["schema_version"]==SCHEMA || error("unsupported checkpoint schema");
    s["physics"]==star_physics(c) || error("Resume requires the same cell, J, identity sweeps and schedule. Use a separate output directory for a different run.");
    return s
end

function _record!(history,T,lx,ly,p,phase,dt,cap,ds;energy=false)
    e=NaN;px=NaN;py=NaN;pz=NaN;
    if energy
        ob=honeycomb_energy_simple(Dict("J"=>1.),T,lx,ly;convention=:permutation);
        all(isfinite,ob.energies) || error("nonfinite simple-lambda energy");
        e=ob.energy_per_site;px=ob.bond_energy.x;py=ob.bond_energy.y;pz=ob.bond_energy.z;
    end
    dims=[dim(space(s,1)) for s in _spectra(lx,ly)];
    row=(record=length(history)+1,phase=phase,stage=Int(p["stage"]),stage_step=Int(p["stage_step"]),
        total_steps=Int(p["total_steps"]),identity_done=Int(p["identity_done"]),tau=Float64(p["tau"]),
        dt=dt,Dmax=cap,min_D=minimum(dims),max_D=maximum(dims),delta_spectrum=ds,
        E_P_simple=e,P_x=px,P_y=py,P_z=pz);
    push!(history,row);println("STAR SERVER ",row);flush(stdout);
end

"""Checkpoint after each chunk. Exceptions leave the last valid disk state intact.
stop_after_chunks is a lightweight restart-test hook, not a convergence criterion.
"""
function run_star_server(c::StarRunConfig;resume=false,initial_checkpoint=nothing,stop_after_chunks=typemax(Int))
    resume && !isnothing(initial_checkpoint) && error("Use either resume or initial_checkpoint, not both");
    describe_star_run(c);BLAS.set_num_threads(c.blas_threads);
    return _locked(c.output_dir) do
        latest=joinpath(c.output_dir,"state_latest.jld2");
        if resume
            s=_load(c);T=s["T_set"];lx=s["lambdax_set"];ly=s["lambday_set"];
            p=s["progress"];history=s["history"];
        else
            (isfile(latest) || isfile(joinpath(c.output_dir,"config_used.toml"))) &&
                error("Output already contains a run. Use --resume or a different --output.");
            if isnothing(initial_checkpoint)
                T,lx,ly,_=initial_honeycomb_star_iPEPS(Lx=c.Lx,Ly=c.Ly);
                p=Dict{String,Any}("stage"=>1,"stage_step"=>0,"total_steps"=>0,
                    "identity_done"=>0,"tau"=>0.,"finished"=>false);
            else
                source=load(initial_checkpoint);
                source["schema_version"]==SCHEMA || error("unsupported source checkpoint");
                source["physics"].Lx==c.Lx && source["physics"].Ly==c.Ly && source["physics"].J==c.J ||
                    error("source checkpoint cell and J must match");
                T=source["T_set"];lx=source["lambdax_set"];ly=source["lambday_set"];
                _validate(T,lx,ly,first(c.stages).Dmax);
                p=Dict{String,Any}("stage"=>1,"stage_step"=>0,
                    "total_steps"=>source["progress"]["total_steps"],
                    "identity_done"=>c.identity_sweeps,"tau"=>source["progress"]["tau"],
                    "finished"=>false,"initial_checkpoint"=>abspath(initial_checkpoint));
                println("Continue saved tensors under a new schedule: ",initial_checkpoint);
            end
            history=NamedTuple[];
            _atomic_write(joinpath(c.output_dir,"config_used.toml")) do tmp
                open(io->TOML.print(io,c.raw),tmp,"w");
            end
            initial_cap=maximum(dim(space(s,1)) for s in _spectra(lx,ly));
            _record!(history,T,lx,ly,p,"initial",0.,initial_cap,0.);
            _checkpoint(c,T,lx,ly,p,history);
        end
        _validate(T,lx,ly,max(9,maximum(s.Dmax for s in c.stages)));
        # Rebuild text snapshots from the checkpoint after any interrupted write.
        _csv(joinpath(c.output_dir,"evolution.csv"),history);
        _csv(joinpath(c.output_dir,"bond_sectors_latest.csv"),_sectors(lx,ly));
        started=time();chunks=0;
        try
            while !p["finished"]
                if chunks>=stop_after_chunks || (c.max_seconds>0 && time()-started>=c.max_seconds)
                    println("Clean stop at checkpoint: ",latest);return :checkpointed;
                end
                old_signature=_signature(lx,ly);
                gauge=p["identity_done"]<c.identity_sweeps;
                if gauge
                    n=min(c.checkpoint_every,c.identity_sweeps-p["identity_done"]);
                    dt=.01;cap=9;phase="identity";
                    itebd_honeycomb(Dict("J"=>0.),T,lx,ly,n*dt,dt,cap;
                        convention=:permutation,step_offset=p["identity_done"],verbose=true);
                    p["identity_done"]+=n;
                else
                    k=p["stage"];
                    k>length(c.stages) && error("invalid checkpoint stage");
                    s=c.stages[k];n=min(c.checkpoint_every,s.steps-p["stage_step"]);
                    k==1 && p["stage_step"]==0 && (n=1); # Secure the first expanded state promptly.
                    dt=s.dt;cap=s.Dmax;phase="evolve";
                    itebd_honeycomb(Dict("J"=>c.J),T,lx,ly,n*dt,dt,cap;
                        convention=:permutation,step_offset=p["total_steps"],verbose=true);
                    p["stage_step"]+=n;p["total_steps"]+=n;p["tau"]+=n*dt;
                end
                _validate(T,lx,ly,cap);
                endstage=!gauge && p["stage_step"]==c.stages[p["stage"]].steps;
                ds=_change(old_signature,_signature(lx,ly));
                record_progress=copy(p);
                _record!(history,T,lx,ly,p,phase,dt,cap,ds);
                if endstage
                    p["stage"]+=1;p["stage_step"]=0;
                    p["finished"]=p["stage"]>length(c.stages);
                end
                archive=!gauge && (endstage || (c.keep_every>0 &&
                    div(p["total_steps"],c.keep_every)!=div(p["total_steps"]-n,c.keep_every)));
                _checkpoint(c,T,lx,ly,p,history;archive=archive); # Before optional expensive observables.
                chunks+=1;
                measure=c.local_energy_every>0 && !gauge &&
                    (endstage || div(p["total_steps"],c.local_energy_every)!=div(p["total_steps"]-n,c.local_energy_every));
                if measure
                    try
                        pop!(history);
                        # Log the completed stage index, even after advancing progress.
                        _record!(history,T,lx,ly,record_progress,phase,dt,cap,ds;energy=true);
                        _checkpoint(c,T,lx,ly,p,history;archive=archive);
                    catch err
                        # Recover the already saved history if only the observable failed.
                        history=load(latest,"history");
                        @warn "Simple-lambda measurement failed; SU checkpoint remains valid" exception=(err,catch_backtrace());
                    end
                end
            end
        catch err
            open(joinpath(c.output_dir,"failure.log"),"a") do io
                println(io,now()," latest valid checkpoint: ",latest,"\n",sprint(showerror,err,catch_backtrace()));
            end
            rethrow();
        end
        println("Schedule complete (not an automatic convergence claim): ",latest);
        return :finished
    end
end

function measure_star_server(c::StarRunConfig)
    (c.Lx,c.Ly) in ((2,2),(4,4)) || error("CTMRG supports 2x2 or 4x4 cells");
    BLAS.set_num_threads(c.ctm_threads);
    return _locked(c.output_dir) do
        s=_load(c);T=s["T_set"];p=s["progress"];
        source=joinpath(c.output_dir,"ctm_source_step$(p["total_steps"]).jld2");
        _atomic_write(tmp->cp(joinpath(c.output_dir,"state_latest.jld2"),tmp;force=true),source);
        setting=LS_CTMRG_settings(CTM_ite_nums=c.ctm_iterations,CTM_conv_tol=c.ctm_tol,
            CTM_ite_info=true,CTM_conv_info=true,grad_checkpoint=false);
        rows=NamedTuple[];env=nothing;previous=NaN;
        for ch in c.ctm_chis
            try
                # Only reuse a CTM built for this exact checkpoint tensor and gauge.
                ob=honeycomb_energy_CTM(Dict("J"=>1.),T,ch,setting;convention=:permutation,CTM0=env);
                physical=all(v->isfinite(v) && -1-1e-8<=v<=1+1e-8,ob.energies);
                valid=physical && ob.corner_converged;
                stable=valid && isfinite(previous) && abs(ob.energy_per_site-previous)<c.ctm_energy_tol;
                row=(step=p["total_steps"],tau=p["tau"],schedule_complete=p["finished"],chi=ch,
                    E_P_CTM=ob.energy_per_site,E_H=c.J*ob.energy_per_site,
                    P_x=ob.bond_energy.x,P_y=ob.bond_energy.y,P_z=ob.bond_energy.z,
                    min_P=minimum(ob.energies),max_P=maximum(ob.energies),
                    corner_error=ob.corner_error,iterations=ob.iterations,physical=physical,
                    corner_converged=ob.corner_converged,
                    delta_E_chi=valid && isfinite(previous) ? abs(ob.energy_per_site-previous) : NaN,
                    chi_stable=stable);
                push!(rows,row);println("STAR SERVER CTM ",row);flush(stdout);
                _atomic_write(joinpath(c.output_dir,"ctm_step$(p["total_steps"])_chi$(ch).jld2")) do tmp
                    jldsave(tmp;row,CTM=ob.CTM,source_step=p["total_steps"],
                        source_checkpoint=basename(source),bonds=ob.bonds,bond_P=ob.energies);
                end
                _csv(joinpath(c.output_dir,"ctm_step$(p["total_steps"]).csv"),rows);
                bondrows=[(direction=String(b.direction),x1=b.site1[1],y1=b.site1[2],
                    x2=b.site2[1],y2=b.site2[2],P=value) for (b,value) in zip(ob.bonds,ob.energies)];
                _csv(joinpath(c.output_dir,"ctm_bonds_step$(p["total_steps"])_chi$(ch).csv"),bondrows);
                env=valid ? ob.CTM : nothing;previous=valid ? ob.energy_per_site : NaN;
            catch err
                env=nothing;previous=NaN;
                open(joinpath(c.output_dir,"ctm_failure.log"),"a") do io
                    println(io,"chi=",ch," ",now(),"\n",sprint(showerror,err,catch_backtrace()));
                end
                @warn "CTMRG failed at chi=$ch; trying the next configured chi" exception=(err,catch_backtrace());
            end
        end
        return rows
    end
end
end
