# Fixed-state chi scan. Reuse the existing TensorKit CTMRG/observable functions.
if !isdefined(@__MODULE__, :KagomeNativeMeasurement)
    include(joinpath(@__DIR__, "KagomeNativeMeasurement.jl"))
end
module KagomeChiScan
using TensorKit, LinearAlgebra, TOML, SHA, JLD2, FileWatching
import ..KagomeNativeMeasurement
const Native = KagomeNativeMeasurement

file_hash(path) = bytes2hex(sha256(read(path)))

function chi_scan_plan(cfg)
    !isempty(cfg.sources) && allunique(s.D for s in cfg.sources) || error("provide distinct checkpoint D values")
    !isempty(cfg.chis) && all(>(0), cfg.chis) && issorted(cfg.chis) && allunique(cfg.chis) || error("chis must be positive, increasing and distinct")
    cfg.threads>0 && 2<=cfg.ctm_iterations<=50 || error("invalid threads or CTMRG iteration limit")
    isfinite(cfg.h) && isfinite(cfg.ctm_tolerance) && cfg.ctm_tolerance>0 || error("invalid field/tolerance")
    sources = Dict[]
    for entry in cfg.sources
        input=abspath(entry.checkpoint)
        isfile(input) || error("missing D=$(entry.D) checkpoint: $input; set sources in the task file to the existing final.jld2 files")
        before=file_hash(input)
        state,_=Native.KagomeU1Cells.load_u1_checkpoint(input)
        actual=maximum(dim(space(getfield(c,n),leg)) for c in state.cells for n in Native.BS.B_FIELDS for leg in 1:2)
        actual==entry.D || error("checkpoint labelled D=$(entry.D) has actual maximum bond dimension $actual")
        file_hash(input)==before || error("checkpoint changed while being read; use a completed final.jld2")
        push!(sources,Dict("D"=>entry.D,"checkpoint"=>input,"sha256"=>before,
            "cell"=>string(state.geometry.pattern),"actual_D_max"=>actual))
    end
    Dict("format"=>"kagome_fixed_state_chi_scan_v1","sources"=>sources,"chis"=>collect(cfg.chis),
        "h"=>cfg.h,"threads"=>cfg.threads,"device"=>cfg.device,"ctm_iterations"=>cfg.ctm_iterations,
        "ctm_tolerance"=>cfg.ctm_tolerance,"representation"=>"u1","fresh_environment_per_chi"=>true,
        "TensorKit"=>string(pkgversion(TensorKit)),"JLD2"=>string(pkgversion(JLD2)))
end

bond_key(r)=(r["representative"],r["orientation"],r["vertex_a"],r["vertex_b"])
site_key(r)=(r["representative"],r["site"])
function max_difference(current,previous,key,value)
    isnothing(previous) && return ""
    old=Dict(key(r)=>r[value] for r in previous)
    Set(keys(old))==Set(key(r) for r in current) || error("observable labels changed across chi")
    maximum(abs(r[value]-old[key(r)]) for r in current)
end

"""Rewrite aggregate files after each completed point, retaining convergence flags."""
function collect_chi_outputs(cfg,plan)
    summaries=Dict[];bonds=Dict[];sites=Dict[];triangles=Dict[]
    for source in plan["sources"]
        D=source["D"];previous=nothing;previous_groups=nothing
        reference_path=joinpath(cfg.out,"D$D","chi40","observables.toml")
        reference=40 in cfg.chis && isfile(reference_path) ? TOML.parsefile(reference_path) : nothing
        for chi in cfg.chis
            path=joinpath(cfg.out,"D$D","chi$chi","observables.toml")
            isfile(path) || continue
            result=TOML.parsefile(path);settings=result["settings"];ctm=result["ctm"]
            settings["input_sha256"]==source["sha256"] && settings["chi"]==chi || error("cached checkpoint/chi mismatch")
            groups=Native.grouped_observables(result,D)
            nsites=length(groups.sites)
            # Internal consistency only: these identities do not independently
            # establish the accuracy of the CTMRG contraction.
            length(groups.bonds)==2nsites || error("incorrect kagome bond count")
            abs(sum(r["energy"] for r in groups.bonds)/nsites-result["exchange_energy_per_spin"])<1e-9 || error("bond sum/energy mismatch")
            abs(result["energy_per_spin"]-result["exchange_energy_per_spin"]-result["field_energy_per_spin"])<1e-10 || error("field/total mismatch")
            row=Dict{String,Any}("D"=>D,"chi"=>chi,"source_sha256"=>source["sha256"],
                "cell"=>source["cell"],"h"=>cfg.h,
                "energy_per_spin"=>result["energy_per_spin"],
                "exchange_energy_per_spin"=>result["exchange_energy_per_spin"],
                "field_energy_per_spin"=>result["field_energy_per_spin"],
                "m_normalized"=>result["m_normalized"],"mean_Sz"=>result["mean_Sz"],
                "ctm_converged"=>ctm["converged"],"ctm_error"=>ctm["error"],
                "ctm_iterations"=>ctm["iterations"],"max_corner_dimension"=>maximum(result["actual_corner_dimensions"]),
                "seconds"=>result["seconds"],"previous_chi"=>isnothing(previous) ? "" : previous["settings"]["chi"],
                "previous_ctm_converged"=>isnothing(previous) ? "" : previous["ctm"]["converged"],
                "delta_exchange_vs_previous_chi"=>isnothing(previous) ? "" : result["exchange_energy_per_spin"]-previous["exchange_energy_per_spin"],
                "delta_total_vs_chi40"=>isnothing(reference) ? "" : result["energy_per_spin"]-reference["energy_per_spin"],
                "chi40_ctm_converged"=>isnothing(reference) ? "" : reference["ctm"]["converged"],
                "max_bond_delta_vs_previous_chi"=>max_difference(groups.bonds,isnothing(previous_groups) ? nothing : previous_groups.bonds,bond_key,"energy"),
                "max_site_m_delta_vs_previous_chi"=>max_difference(groups.sites,isnothing(previous_groups) ? nothing : previous_groups.sites,site_key,"m_normalized"))
            push!(summaries,row)
            common=Dict("chi"=>chi,"h"=>cfg.h,"ctm_converged"=>ctm["converged"],"ctm_error"=>ctm["error"],
                "source_sha256"=>source["sha256"])
            old=isnothing(previous_groups) ? nothing : Dict(bond_key(r)=>r["energy"] for r in previous_groups.bonds)
            for r in groups.bonds
                push!(bonds,merge(r,common,Dict("previous_chi"=>row["previous_chi"],
                    "delta_energy_vs_previous_chi"=>isnothing(old) ? "" : r["energy"]-old[bond_key(r)])))
            end
            append!(sites,[merge(r,common) for r in groups.sites])
            append!(triangles,[merge(r,common) for r in groups.triangles])
            previous=result;previous_groups=groups
        end
    end
    for (name,rows) in (("summary",summaries),("bond_energies",bonds),
                        ("site_magnetization",sites),("triangle_observables",triangles))
        Native.write_csv(joinpath(cfg.out,name*".csv"),rows)
    end
    Native.atomic_toml(joinpath(cfg.out,"summary.toml"),Dict("results"=>summaries))
    Native.write_json(joinpath(cfg.out,"summary.json"),summaries)
    temporary=tempname(cfg.out)
    open(temporary,"w") do io
        println(io,"Fixed U1 states; J=1; H=Hex+h*sum(Sz); h=",cfg.h,"; fresh CTMRG environment for every chi.")
        println(io,"No SU or optimization. Bond energy=<Si.Sj>. CTMRG convergence is not chi convergence.")
        for r in summaries
            println(io,"D=",r["D"]," chi=",r["chi"]," E=",r["energy_per_spin"]," Eex=",r["exchange_energy_per_spin"],
                " m=",r["m_normalized"]," converged=",r["ctm_converged"]," iterations=",r["ctm_iterations"],
                " error=",r["ctm_error"]," retained=",r["max_corner_dimension"],
                " dEex(prev)=",r["delta_exchange_vs_previous_chi"]," max_dbond(prev)=",r["max_bond_delta_vs_previous_chi"])
        end
    end
    mv(temporary,joinpath(cfg.out,"summary.txt");force=true)
    summaries
end

function run_locked(cfg,plan)
    configuration=joinpath(cfg.out,"configuration.toml")
    if isfile(configuration)
        TOML.parsefile(configuration)==plan || error("input/settings/version changed; choose a new out directory")
    else
        Native.atomic_toml(configuration,plan)
    end
    println("Fixed-state chi check: D=",[s["D"] for s in plan["sources"]]," chi=",cfg.chis,
        "; ",cfg.device,"; threads=",cfg.threads,"; CTMRG <=",cfg.ctm_iterations);flush(stdout)
    BLAS.set_num_threads(cfg.threads)
    collect_chi_outputs(cfg,plan)
    for source in plan["sources"],chi in cfg.chis
        D=source["D"];input=source["checkpoint"]
        file_hash(input)==source["sha256"] || error("source checkpoint changed during the chi scan")
        println("MEASURE fixed D=$D chi=$chi; fresh environment (or reuse completed matching result)");flush(stdout)
        Native.measure_checkpoint(input,joinpath(cfg.out,"D$D","chi$chi");representation=:u1,
            h=cfg.h,chi,iterations=cfg.ctm_iterations,tolerance=cfg.ctm_tolerance,
            device=cfg.device,threads=cfg.threads,D)
        file_hash(input)==source["sha256"] || error("source checkpoint changed during measurement")
        collect_chi_outputs(cfg,plan)
        Native.CTM.square_J1_reclaim_device_memory!()
        GC.gc()
    end
    println("DONE fixed-state chi check: ",abspath(joinpath(cfg.out,"summary.csv")));flush(stdout)
    abspath(cfg.out)
end

function run_chi_scan(cfg)
    plan=chi_scan_plan(cfg)
    root=abspath(cfg.out)
    any(s->startswith(s["checkpoint"],root*string(Base.Filesystem.path_separator)),plan["sources"]) && error("put chi outputs outside the source checkpoint folders")
    mkpath(root)
    if Sys.isunix()
        open(joinpath(root,"writer.lock"),"a+") do io
            ccall(:flock,Cint,(Cint,Cint),fd(io),6)==0 || error("another live chi scan owns $root; stop that process before resuming")
            try
                seekstart(io);truncate(io,0);println(io,"PID=",getpid());flush(io)
                run_locked(cfg,plan)
            finally
                ccall(:flock,Cint,(Cint,Cint),fd(io),8)
            end
        end
    else
        FileWatching.Pidfile.mkpidlock(joinpath(root,"writer.pid");wait=false) do
            run_locked(cfg,plan)
        end
    end
end
end # module
