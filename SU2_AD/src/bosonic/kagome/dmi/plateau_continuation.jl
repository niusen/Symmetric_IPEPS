"""Track a dense plateau branch in D or signed Dz, reusing the native SU/CTMRG.

These are candidate-state continuations, not a ground-state/plateau certificate.
No random initialization, noise, explicit U1 restriction, or field continuation.
"""
module KagomeDMIPlateauContinuation
using TensorKit, LinearAlgebra, TOML, SHA, FileWatching
if !isdefined(parentmodule(@__MODULE__), :KagomeDMIPlateauExplore)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "plateau_explore.jl"))
end
import ..KagomeDMIPlateauExplore
const P=KagomeDMIPlateauExplore
const K=P.K
const N=P.N
const BS=K.BS

function check_dense(s;Dmax)
    s isa N.DenseKagomeState || error("expected a dense kagome state")
    isempty(s.auxiliary_charges) || error("auxiliary-charge/U1 seeds are not used in these tasks")
    g=s.geometry
    g.pattern==:sqrt3xsqrt3 && length(s.cells)==3 || error("expected the nine-spin sqrt3 cell")
    dimensions=Int[]
    for (r,c) in enumerate(s.cells)
        for name in (:T_u,:T_d,BS.B_FIELDS...)
            t=getfield(c,name)
            isfinite(norm(t)) && norm(t)>0 || error("zero/nonfinite tensor")
        end
        for name in BS.B_FIELDS
            b=getfield(c,name)
            numind(b)==3 && dim(space(b,3))==2 || error("unexpected physical/auxiliary axes")
        end
        for orientation in (:up,:dn)
            t=orientation==:up ? c.T_u : c.T_d
            leg=orientation==:up ? 2 : 1
            for (axis,(owner,b)) in enumerate(BS.triangle_sites(g,r,orientation))
                space(t,axis)==space(getfield(s.cells[owner],BS.B_FIELDS[b]),leg)' || error("bond mismatch")
                d=dim(space(t,axis));d<=Dmax || error("retained dimension $d exceeds cap $Dmax")
                push!(dimensions,d)
            end
        end
    end
    maximum(dimensions)
end

"""Topological plan. Every h has its own D/Dz lineage; no neighboring-h seed."""
function plan(cfg)
    fields=Float64.(cfg.fields);dims=Int.(cfg.dims);dzs=Float64.(cfg.Dz_values)
    !isempty(fields) && allunique(fields) && all(isfinite,fields) || error("invalid fields")
    !isempty(dzs) && allunique(dzs) && all(isfinite,dzs) || error("invalid Dz values")
    !isempty(dims) && allunique(dims) && all(>(0),dims) || error("invalid dimensions")
    nodes=NamedTuple[]
    add(D,Dz,h,parent,branch)=begin
        Dz=Float64(Dz);h=Float64(h)
        id="D$(D)_Dz$(Dz)_$(branch)/h$(h)"
        push!(nodes,(id=id,D=Int(D),Dz=Dz,h=h,parent=Int(parent),branch=branch))
        length(nodes)
    end
    if cfg.mode==:D_chain
        dims==collect(cfg.seed_D:maximum(dims)) || error("D chain must start at seed_D and increase by one")
        dzs==[Float64(cfg.seed_Dz)] || error("D chain keeps Dz fixed at seed_Dz")
        previous=Dict{Float64,Int}()
        for D in dims,h in fields
            previous[h]=add(D,only(dzs),h,get(previous,h,0),"platform_D_chain")
        end
    elseif cfg.mode==:Dz_chain
        dims==[cfg.seed_D] || error("Dz chain keeps D fixed at seed_D")
        cfg.seed_Dz in dzs || error("Dz grid must include seed_Dz")
        base=Dict(h=>add(cfg.seed_D,cfg.seed_Dz,h,0,"platform_Dz_base") for h in fields)
        # Two independent continuations leave the same +0.02 base. The lower
        # branch passes through zero to negative Dz, without conjugating/resetting
        # the ket. This implements the requested same-platform-state experiment.
        for (branch,values) in (("platform_Dz_increasing",sort(filter(>(cfg.seed_Dz),dzs))),
                                ("platform_Dz_decreasing",sort(filter(<(cfg.seed_Dz),dzs);rev=true)))
            previous=copy(base)
            for Dz in values,h in fields
                previous[h]=add(cfg.seed_D,Dz,h,previous[h],branch)
            end
        end
    else
        error("mode must be :D_chain or :Dz_chain")
    end
    allunique(n.id for n in nodes) || error("duplicate point IDs")
    all(n.parent<i for (i,n) in enumerate(nodes)) || error("invalid dependency order")
    nodes
end

function load_seed(cfg)
    isfile(cfg.input) || error("missing platform checkpoint: $(abspath(cfg.input)); set input in the .jl settings")
    isfile(cfg.seed_observables) || error("missing platform measurement: $(abspath(cfg.seed_observables))")
    s,data=K.load_checkpoint(cfg.input)
    check_dense(s;Dmax=cfg.seed_D)
    get(data,"format","")=="kagome_dmi_v1" || error("expected a DMI checkpoint")
    isapprox(data["J"],1.;atol=1e-12) && isapprox(data["Dz"],cfg.seed_Dz;atol=1e-12) || error("seed couplings mismatch")
    isapprox(data["h"],-cfg.seed_h;atol=1e-12) || error("seed physical field mismatch")
    obs=TOML.parsefile(cfg.seed_observables)
    get(obs,"measurement_reliable",false) && get(obs,"valid_measurement",false) || error("seed measurement not certified")
    obs["D"]==cfg.seed_D && isapprox(obs["Dz"],cfg.seed_Dz;atol=1e-12) || error("seed observation parameters mismatch")
    isapprox(obs["h_physical"],cfg.seed_h;atol=1e-12) || error("seed observation field mismatch")
    abs(obs["m_normalized"]-cfg.seed_magnetization)<=cfg.seed_m_tolerance || error("seed is not the requested plateau candidate")
    s,obs
end

function settings(cfg)
    Dict("version"=>"dense_platform_continuation_v1","mode"=>String(cfg.mode),
        "dims"=>cfg.dims,"Dz_values"=>cfg.Dz_values,"fields"=>cfg.fields,
        "input"=>abspath(cfg.input),"input_sha256"=>bytes2hex(sha256(read(cfg.input))),
        "seed_observables"=>abspath(cfg.seed_observables),
        "seed_observables_sha256"=>bytes2hex(sha256(read(cfg.seed_observables))),
        "seed_D"=>cfg.seed_D,"seed_Dz"=>cfg.seed_Dz,"seed_h"=>cfg.seed_h,
        "seed_magnetization"=>cfg.seed_magnetization,"seed_m_tolerance"=>cfg.seed_m_tolerance,
        "chi"=>cfg.chi,"ctm_max"=>cfg.ctm_max,"ctm_tol"=>cfg.ctm_tol,
        "threads"=>cfg.threads,"trun_tol"=>cfg.trun_tol,
        "schedule"=>[Dict("dt"=>s.dt,"tau"=>s.tau) for s in cfg.schedule],
        "initial_noise"=>0.,"symmetry"=>"dense, no explicit U1/SU2 constraint",
        "initialization_policy"=>"same h previous D/Dz; no h continuation; preserve original ket")
end

function save_tables(folder,result,g)
    haskey(result,"cells") || return
    sites=NamedTuple[];bonds=NamedTuple[];triangles=NamedTuple[]
    lookup=Dict((c["x"],c["y"])=>c for c in result["cells"])
    for c in result["cells"]
        x=c["x"];y=c["y"];owner=N.owner(g,x,y)
        for i in 1:3
            push!(sites,(x=x,y=y,independent_cell=owner,site=i,Sx=c["Sx"][i],Sy=c["Sy"][i],Sz=c["Sz"][i],m_normalized=2c["Sz"][i]))
        end
        for orientation in ("up","down")
            ss=c["bonds_$(orientation)_SS"];kap=c["kappa_$(orientation)_CCW"]
            for i in 1:3
                push!(bonds,(x=x,y=y,independent_cell=owner,triangle=orientation,bond=i,
                    SS=ss[i],kappa_CCW=kap[i],exchange_energy=result["J"]*ss[i],
                    dmi_energy=result["Dz"]*kap[i],energy=c["bonds_$(orientation)_energy"][i]))
            end
            # Native down-triangle sites: B2(R), B3(R+ey), B1(R+ex).
            sz=orientation=="up" ? sum(c["Sz"]) :
                c["Sz"][2]+lookup[(x,mod1(y+1,g.Ly))]["Sz"][3]+lookup[(mod1(x+1,g.Lx),y)]["Sz"][1]
            exchange=result["J"]*sum(ss);dm=result["Dz"]*sum(kap)
            field=-result["h_physical"]*sz/2 # each site belongs to two triangles
            push!(triangles,(x=x,y=y,independent_cell=owner,triangle=orientation,
                Sz=sz,mean_m_normalized=2sz/3,SS=sum(ss),kappa_CCW=sum(kap),
                exchange_energy=exchange,dmi_energy=dm,field_energy=field,energy=exchange+dm+field))
        end
    end
    N.write_csv(joinpath(folder,"site_magnetization.csv"),sites)
    N.write_csv(joinpath(folder,"bond_energies.csv"),bonds)
    N.write_csv(joinpath(folder,"triangle_observables.csv"),triangles)
end

function collect_results(out)
    rows=Dict[]
    for (dir,_,files) in walkdir(out)
        "continuation_result.toml" in files || continue
        r=TOML.parsefile(joinpath(dir,"continuation_result.toml"))
        push!(rows,Dict(k=>v for (k,v) in r if !(k in ("cells","ctm","history"))))
    end
    sort!(rows;by=r->r["point_index"])
    N.write_csv(joinpath(out,"progress.csv"),rows)
end

function run(cfg)
    nodes=plan(cfg)
    cfg.chi>0 && 1<=cfg.ctm_max<=50 && cfg.ctm_tol>0 || error("invalid CTMRG settings (cap <=50)")
    cfg.threads>0 && 0<=cfg.trun_tol<1 || error("invalid runtime/truncation settings")
    all(s.dt>0 && s.tau>=0 && isfinite(s.dt) && isfinite(s.tau) for s in cfg.schedule) || error("invalid SU schedule")
    _,seed_obs=load_seed(cfg)
    config=settings(cfg)
    BLAS.set_num_threads(cfg.threads);N.prepare_device!("cpu");mkpath(cfg.out)
    FileWatching.Pidfile.mkpidlock(joinpath(cfg.out,"writer.pid");wait=false) do
        configfile=joinpath(cfg.out,"settings.toml")
        isfile(configfile) && TOML.parsefile(configfile)!=config && error("settings/source changed; choose a new out")
        N.atomic_toml(configfile,config)
        N.write_json(joinpath(cfg.out,"plan.json"),[Dict("point_index"=>i,"point_id"=>n.id,
            "D"=>n.D,"Dz"=>n.Dz,"h"=>n.h,"parent_index"=>n.parent,
            "parent_point"=>(n.parent==0 ? "original_platform_seed" : nodes[n.parent].id)) for (i,n) in enumerate(nodes)])
        println("$(length(nodes)) platform-branch points; mode=$(cfg.mode); CPU $(cfg.threads) threads; chi=$(cfg.chi); data only")
        println("No noise, no U1 projection, no h continuation. A flat M alone does not prove a stable plateau.");flush(stdout)
        collect_results(cfg.out)
        localcfg=merge(cfg,(fresh=cfg.schedule,warm=cfg.schedule))
        for (i,n) in enumerate(nodes)
            folder=joinpath(cfg.out,n.id);mkpath(folder)
            done=joinpath(folder,"continuation_result.toml")
            if isfile(done)
                isfile(joinpath(folder,"final.jld2")) || error("result exists but checkpoint is missing: $folder")
                println("SKIP saved $(n.id)");flush(stdout);continue
            end
            source=n.parent==0 ? abspath(cfg.input) : joinpath(cfg.out,nodes[n.parent].id,"final.jld2")
            isfile(source) || error("missing parent checkpoint: $source")
            parent,_=K.load_checkpoint(source)
            initialD=check_dense(parent;Dmax=n.D)
            manifest=Dict("point_index"=>i,"point_id"=>n.id,"parent_index"=>n.parent,
                "parent_point"=>(n.parent==0 ? "original_platform_seed" : nodes[n.parent].id),
                "source"=>source,"source_sha256"=>bytes2hex(sha256(read(source))),
                "initial_max_bond_dimension"=>initialD,"target_D"=>n.D,
                "initial_noise"=>0.,"h_continuation"=>false,"imposed_U1"=>false,
                "growth"=>"existing ket; native SU raises truncation cap without random padding")
            record=joinpath(folder,"initialization.toml")
            isfile(record) && TOML.parsefile(record)!=manifest && error("parent changed: $folder")
            N.atomic_toml(record,manifest)
            # Reuse the original gate sequence and stage checkpoint handling.
            # Smaller source spaces are valid: native HOSVD may grow Schmidt
            # channels up to Dmax. No artificial random/zero-padded state is made.
            final,result=P.point(localcfg,n.D,n.Dz,n.h,n.branch,source,true)
            state,_=K.load_checkpoint(final)
            actualD=check_dense(state;Dmax=n.D)
            r=merge(result,Dict("point_index"=>i,"point_id"=>n.id,"mode"=>String(cfg.mode),
                "parent_index"=>n.parent,"parent_point"=>manifest["parent_point"],
                "initialization"=>"continued dense plateau candidate","seed_source"=>abspath(cfg.input),
                "seed_source_trial"=>get(seed_obs,"trial",0),"seed_source_rng"=>get(seed_obs,"seed",0),
                "initial_max_bond_dimension"=>initialD,"actual_max_bond_dimension"=>actualD,
                "independent_cells"=>length(state.cells),"initial_noise"=>0.,"h_continuation"=>false,
                "measurement_seconds"=>get(result,"seconds",0.),
                "su_seconds"=>sum(get(v,"seconds",0.) for v in result["history"];init=0.)))
            ctm=get(r,"ctm",Dict{String,Any}())
            r["ctm_converged"]=get(ctm,"converged",false)
            r["measurement_reliable"]=get(r,"measurement_reliable",false)
            save_tables(folder,r,state.geometry)
            N.write_json(joinpath(folder,"continuation_result.json"),r)
            N.atomic_toml(done,r) # completion marker only after all data files
            collect_results(cfg.out)
            println("SAVED $(n.id) E=$(get(r,"energy_per_spin",NaN)) M=$(get(r,"m_normalized",NaN)) actualD=$actualD reliable=$(r["measurement_reliable"])");flush(stdout)
        end
        write(joinpath(cfg.out,"COMPLETED.txt"),"All plateau-branch points saved. Compare energies with independent competing states; flat M alone is not a plateau certificate.\n")
    end
end
end
