"""Small CPU dense iPESS order test; no U1 projection and no enforced C3."""
module KagomeDMIOrderProbe
using TensorKit, LinearAlgebra, JLD2, TOML
if !isdefined(parentmodule(@__MODULE__),:KagomeDMI)
    Base.include(parentmodule(@__MODULE__),joinpath(@__DIR__,"KagomeDMI.jl"))
end
import ..KagomeDMI
const DMI=KagomeDMI
const N=DMI.N
const BS=DMI.BS

function weights(s)
    [sort(abs.(diag(convert(Array,getfield(c,n))))) for c in s.cells for n in (BS.U_FIELDS...,BS.D_FIELDS...) if !isnothing(getfield(c,n))]
end
function distance(a,b)
    maximum(begin
        k=max(length(x),length(y));z=zeros(k);z[1:length(x)].+=x;z[1:length(y)].-=y
        maximum(abs.(z))
    end for (x,y) in zip(a,b))
end
function run_case(cfg,D,branch)
    folder=joinpath(cfg.out,"D$(D)_$(branch)");mkpath(folder)
    final=joinpath(folder,"final.jld2");history=Dict[];completed=0
    if isfile(final)
        s,data=DMI.load_checkpoint(final);history=data["metadata"]["history"];completed=data["metadata"]["completed"]
    else
        s=branch=="random" ? DMI.random_dense_state(D,cfg.seed+D;Lx=3,Ly=3,pattern=:sqrt3xsqrt3) :
          branch=="ordered120" ? DMI.product_120(;Lx=3,Ly=3,pattern=:sqrt3xsqrt3) : error("unknown branch")
        DMI.save_checkpoint(joinpath(folder,"initial.jld2"),s;J=cfg.J,Dz=cfg.Dz,h=cfg.h)
    end
    for (stage,setting) in enumerate(cfg.schedule)
        stage<=completed && continue
        start=time();previous=nothing;last=Float64[];last_step=0
        DMI.evolve!(s;schedule=[setting],Dmax=D,J=cfg.J,Dz=cfg.Dz,h=cfg.h,trun_tol=cfg.trun_tol,
            callback=(s,_,step,n)->begin
                if step%20==0 || step==n
                    current=weights(s)
                    if !isnothing(previous)
                        push!(last,distance(previous,current)/((step-last_step)*setting.dt))
                    end
                    previous=current;last_step=step
                end
                if step%100==0 || step==n
                    println("JL D=$D $branch stage=$stage step=$step/$n seconds=$(time()-start)");flush(stdout)
                end
            end)
        push!(history,Dict("stage"=>stage,"dt"=>setting.dt,"tau"=>setting.tau,
            "seconds"=>time()-start,"weight_drift_per_tau_last20"=>last[end]))
        DMI.save_checkpoint(final,s;J=cfg.J,Dz=cfg.Dz,h=cfg.h,metadata=Dict("completed"=>stage,"history"=>history))
        N.atomic_toml(joinpath(folder,"su_progress.toml"),Dict("completed"=>stage,"history"=>history))
    end
    N.atomic_toml(joinpath(folder,"schmidt_spectra.toml"),Dict("weights"=>weights(s)))
    for chi in (D==maximum(cfg.dims) ? (cfg.chi,cfg.check_chi) : (cfg.chi,))
        target=joinpath(folder,"observables_chi$(chi).json");isfile(target) && continue
        result=DMI.measure(s;J=cfg.J,Dz=cfg.Dz,h=cfg.h,chi,iterations=cfg.ctm_max,tolerance=cfg.ctm_tol,verbose=false)
        result["D"]=D;result["branch"]=branch;result["chi"]=chi;result["seed"]=cfg.seed+D;result["su_history"]=history
        result["actual_D_max"]=maximum(dim(space(getfield(c,n),leg)) for c in s.cells for n in BS.B_FIELDS for leg in 1:2)
        N.write_json(target,result);N.atomic_toml(joinpath(folder,"observables_chi$(chi).toml"),result)
        println("JL FINAL D=$D $branch chi=$chi E=$(result["energy_per_spin"]) CTM=$(result["ctm"])");flush(stdout)
    end
end
function run_all(cfg)
    BLAS.set_num_threads(cfg.threads);N.prepare_device!("cpu");mkpath(cfg.out)
    settings=Dict("dims"=>cfg.dims,"branches"=>cfg.branches,"seed"=>cfg.seed,"J"=>cfg.J,"Dz"=>cfg.Dz,"h"=>cfg.h,
        "schedule"=>[Dict("dt"=>s.dt,"tau"=>s.tau) for s in cfg.schedule],"chi"=>cfg.chi,"check_chi"=>cfg.check_chi,
        "ctm_max"=>cfg.ctm_max,"ctm_tol"=>cfg.ctm_tol,"trun_tol"=>cfg.trun_tol)
    saved=joinpath(cfg.out,"settings.toml")
    isfile(saved) && TOML.parsefile(saved)!=settings && error("settings changed; use a new output")
    N.atomic_toml(saved,settings)
    for D in cfg.dims,branch in cfg.branches;run_case(cfg,D,branch);end
    println("JL ORDER PROBE COMPLETE");flush(stdout)
end
end
