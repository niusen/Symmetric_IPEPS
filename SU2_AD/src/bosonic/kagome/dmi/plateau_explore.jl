module KagomeDMIPlateauExplore
using TensorKit, LinearAlgebra, JLD2, TOML
if !isdefined(parentmodule(@__MODULE__),:KagomeDMI)
    Base.include(parentmodule(@__MODULE__),joinpath(@__DIR__,"KagomeDMI.jl"))
end
import ..KagomeDMI
const K=KagomeDMI
const N=K.N
function point(cfg,D,Dz,h,branch,source,fresh)
    folder=joinpath(cfg.out,"D$(D)_Dz$(Dz)_$(branch)","h$(h)");mkpath(folder)
    final=joinpath(folder,"final.jld2");stagefile=joinpath(folder,"stage.jld2");done=joinpath(folder,"observables.toml")
    isfile(done) && return final,TOML.parsefile(done)
    completed=0;history=Dict[]
    if isfile(final)
        s,data=K.load_checkpoint(final);completed=999;history=data["metadata"]["history"]
    elseif isfile(stagefile)
        s,data=K.load_checkpoint(stagefile);completed=data["metadata"]["completed"];history=data["metadata"]["history"]
    elseif !isnothing(source)
        s,_=K.load_checkpoint(source)
    else
        s=K.random_dense_state(D,cfg.seed+D+round(Int,Dz*1000)+(branch=="up" ? 0 : 10000);
            Lx=3,Ly=3,pattern=:sqrt3xsqrt3)
    end
    isempty(s.auxiliary_charges) || error("unrestricted dense state required; no projected-charge seed")
    # Match Python's PHYSICAL Hamiltonian -h Sz. The Julia API uses +h Sz.
    field=-h;schedule=fresh ? cfg.fresh : cfg.warm
    isfile(joinpath(folder,"initial.jld2")) || K.save_checkpoint(joinpath(folder,"initial.jld2"),s;J=1.,Dz,h=field)
    for (k,stage) in enumerate(schedule)
        k<=completed && continue
        start=time()
        K.evolve!(s;schedule=[stage],Dmax=D,J=1.,Dz,h=field,trun_tol=cfg.trun_tol,
            callback=(s,_,step,n)->begin
                if step%250==0 || step==n
                    println("JL D=$D Dz=$Dz h=$h $branch stage=$k $step/$n seconds=$(time()-start)");flush(stdout)
                end
            end)
        push!(history,Dict("stage"=>k,"dt"=>stage.dt,"tau"=>stage.tau,"seconds"=>time()-start))
        K.save_checkpoint(stagefile,s;J=1.,Dz,h=field,metadata=Dict("completed"=>k,"history"=>history))
    end
    K.save_checkpoint(final,s;J=1.,Dz,h=field,metadata=Dict("history"=>history,"source"=>string(source)))
    r=try
        result=K.measure(s;J=1.,Dz,h=field,chi=cfg.chi,iterations=cfg.ctm_max,tolerance=cfg.ctm_tol,verbose=false)
        result["valid_measurement"]=result["observables_hermiticity_ok"]
        result["m_normalized"]=2result["mean_Sz"] # physical positive-field convention
        result
    catch err
        err isa InterruptException && rethrow()
        open(io->showerror(io,err,catch_backtrace()),joinpath(folder,"measurement_error.txt"),"w")
        Dict{String,Any}("valid_measurement"=>false,"error"=>sprint(showerror,err))
    end
    merge!(r,Dict("D"=>D,"Dz"=>Dz,"h_physical"=>h,"julia_h_parameter"=>field,"branch"=>branch,
        "history"=>history,"symmetry"=>"none","chi"=>cfg.chi,"source"=>string(source)))
    N.write_json(joinpath(folder,"observables.json"),r);N.atomic_toml(done,r)
    println("JL POINT D=$D Dz=$Dz h=$h $branch E=$(get(r,"energy_per_spin",NaN)) m=$(get(r,"m_normalized",NaN)) valid=$(r["valid_measurement"])");flush(stdout)
    final,r
end
function collect_results(root)
    rows=Dict[]
    for (dir,_,files) in walkdir(root)
        "observables.toml" in files || continue
        r=TOML.parsefile(joinpath(dir,"observables.toml"));push!(rows,Dict(k=>v for (k,v) in r if !(k in ("cells","ctm","history"))))
    end
    N.write_csv(joinpath(root,"progress.csv"),rows)
end
function run(cfg)
    BLAS.set_num_threads(cfg.threads);N.prepare_device!("cpu");mkpath(cfg.out)
    for D in cfg.dims,Dz in cfg.Dz
        for (branch,fields) in (("up",cfg.fields),("down",reverse(cfg.fields)))
            source=nothing
            for (i,h) in enumerate(fields)
                source,_=point(cfg,D,Dz,h,branch,source,i==1);collect_results(cfg.out)
            end
        end
    end
    write(joinpath(cfg.out,"COMPLETED.txt"),"All bounded scans finished. Compare energy branches/convergence before interpreting plateaux.\n")
end
end
