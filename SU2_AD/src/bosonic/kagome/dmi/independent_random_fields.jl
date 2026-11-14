module KagomeDMIIndependentFields
using TensorKit,LinearAlgebra,TOML,JLD2,SHA,FileWatching
if !isdefined(parentmodule(@__MODULE__),:KagomeDMIPlateauExplore)
    Base.include(parentmodule(@__MODULE__),joinpath(@__DIR__,"plateau_explore.jl"))
end
import ..KagomeDMIPlateauExplore
const P=KagomeDMIPlateauExplore
seed_for(cfg,D,Dz,h,t)=cfg.seed_base+1000000000+t*10000000+D*1000000+round(Int,Dz*1000)*1000+round(Int,h*1000)
folder_for(cfg,D,Dz,h,t)=joinpath(cfg.out,"D$(D)_Dz$(Dz)_random$(t)","h$(h)")
function run(cfg)
    BLAS.set_num_threads(cfg.threads);P.N.prepare_device!("cpu");mkpath(cfg.out)
    cases=[(D,dz,h,t) for D in cfg.dims for dz in cfg.Dz for h in cfg.fields for t in cfg.trials]
    seeds=[seed_for(cfg,c...) for c in cases];allunique(seeds) || error("seed collision")
    settings=Dict("dims"=>cfg.dims,"Dz"=>cfg.Dz,"fields"=>cfg.fields,"trials"=>cfg.trials,"seed_base"=>cfg.seed_base,
        "chi"=>cfg.chi,"ctm_max"=>cfg.ctm_max,"ctm_tol"=>cfg.ctm_tol,"threads"=>cfg.threads,
        "schedule"=>[Dict("dt"=>s.dt,"tau"=>s.tau) for s in cfg.fresh],"trun_tol"=>cfg.trun_tol)
    FileWatching.Pidfile.mkpidlock(joinpath(cfg.out,"writer.pid");wait=false) do
        saved=joinpath(cfg.out,"settings.toml")
        isfile(saved) && TOML.parsefile(saved)!=settings && error("settings changed; choose new out")
        P.N.atomic_toml(saved,settings)
        P.N.write_json(joinpath(cfg.out,"plan.json"),[Dict("D"=>c[1],"Dz"=>c[2],"h"=>c[3],"trial"=>c[4],"seed"=>seed) for (c,seed) in zip(cases,seeds)])
        for ((D,Dz,h,trial),seed) in zip(cases,seeds)
            folder=folder_for(cfg,D,Dz,h,trial);mkpath(folder);done=joinpath(folder,"independent_result.json")
            isfile(done) && continue
            manifest=Dict("D"=>D,"Dz"=>Dz,"h"=>h,"trial"=>trial,"seed"=>seed,
                "initialization"=>"fresh independent complex Gaussian","previous_field_source"=>"none",
                "previous_D_source"=>"none","imposed_U1"=>false,"imposed_C3"=>false)
            p=joinpath(folder,"initialization.toml")
            isfile(p) && TOML.parsefile(p)!=manifest && error("initialization mismatch")
            P.N.atomic_toml(p,manifest)
            localcfg=merge(cfg,(seed=seed-D-round(Int,Dz*1000)-10000,))
            final,r=P.point(localcfg,D,Dz,h,"random$(trial)",nothing,true)
            s,_=P.K.load_checkpoint(joinpath(folder,"initial.jld2"))
            isempty(s.auxiliary_charges) || error("projected seed is forbidden")
            reference=P.K.random_dense_state(D,seed;Lx=3,Ly=3,pattern=:sqrt3xsqrt3)
            norm(s.cells[1].B_a-reference.cells[1].B_a)<1e-12 || error("actual initial tensor disagrees with recorded seed")
            audit=merge(manifest,Dict("initial_checkpoint_sha256"=>bytes2hex(sha256(read(joinpath(folder,"initial.jld2"))))))
            P.N.atomic_toml(joinpath(folder,"initialization_audit.toml"),audit)
            if !(r["valid_measurement"] && get(r,"ctm",Dict("converged"=>false))["converged"])
                state,_=P.K.load_checkpoint(final)
                try
                    more=P.K.measure(state;J=1.,Dz,h=-h,chi=24,iterations=cfg.ctm_max,tolerance=cfg.ctm_tol,verbose=false)
                    more["m_normalized"]=2more["mean_Sz"];more["chi"]=24
                    more["valid_measurement"]=more["observables_hermiticity_ok"]
                    P.N.write_json(joinpath(folder,"remeasure_chi24.json"),more)
                    if more["valid_measurement"] && more["ctm"]["converged"];r=more;end
                catch err
                    err isa InterruptException && rethrow()
                    r["remeasure_error"]=sprint(showerror,err)
                end
            end
            merge!(r,Dict("D"=>D,"Dz"=>Dz,"h_physical"=>h,"trial"=>trial,"seed"=>seed,
                "branch"=>"random$(trial)","initialization"=>"fresh independent random","source"=>"none","warm_start"=>false))
            P.N.write_json(done,r);P.N.atomic_toml(joinpath(folder,"independent_result.toml"),r)
            println("JULIA INDEPENDENT D=$D Dz=$Dz h=$h trial=$trial seed=$seed E=$(get(r,"energy_per_spin",NaN)) m=$(get(r,"m_normalized",NaN))");flush(stdout)
            rows=Dict[]
            for (dir,_,files) in walkdir(cfg.out)
                "independent_result.toml" in files || continue
                v=TOML.parsefile(joinpath(dir,"independent_result.toml"));push!(rows,Dict(k=>x for (k,x) in v if !(k in ("cells","ctm","history"))))
            end
            P.N.write_csv(joinpath(cfg.out,"progress.csv"),rows)
        end
        write(joinpath(cfg.out,"COMPLETED.txt"),"Independent random initializations at every field finished; inspect quality flags.\n")
    end
end
end
