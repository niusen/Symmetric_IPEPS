# Long server runs remain sequential: one state and one norm CTM environment in
# memory. Completed case markers support restart without rerunning finished cases.
batch_utc()=Dates.format(Dates.now(Dates.UTC),dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
batch_seconds(t)=Float64(time_ns()-t)/1e9
batch_log(message)=(println("[",batch_utc(),"] ",message);flush(stdout))

function variance_code_sha256()
    files=[joinpath(@__DIR__,name) for name in ("KagomeEnergyVariance.jl",
        "variance_operators.jl","variance_contraction.jl","variance_array_sweeps.jl",
        "variance_workflow.jl","variance_batch_plan.jl","variance_batch.jl")]
    append!(files,[joinpath(native,name) for name in ("KagomeNativeMeasurement.jl",
        "conversion_layers.jl","factorized_ctm.jl","memory_bounded_ctm.jl","observables.jl")])
    bytes2hex(sha256(join([basename(p)*":"*Plan.file_sha256(p) for p in files],"\n")))
end

"""One SMALL-D speed run, reporting first-call compilation separately."""
function run_variance_benchmark(cfg;startup_seconds=0.)
    length(cfg.dims)==1 && length(cfg.chis)==1 || error("Benchmark accepts one D and one chi")
    ispath(cfg.out) && error("Benchmark requires a fresh directory to avoid timing a resume/skip")
    batch_log("BENCHMARK FIRST CALL; D=$(only(cfg.dims)), chi=$(only(cfg.chis)), L=$(cfg.sizes), contraction_Lmax=$(cfg.contraction_size)")
    timing=@timed run_variance_batch(cfg;startup_seconds)
    stats=Dict("wall_seconds"=>timing.time,"gc_seconds"=>timing.gctime,
        "allocated_bytes_cumulative"=>timing.bytes,"module_load_seconds"=>startup_seconds,
        "note"=>"FIRST CALL including Julia compilation; compilation timers are diagnostics, not an independently measured warm-run speed; allocated bytes are NOT peak memory")
    for field in (:compile_time,:recompile_time)
        hasproperty(timing,field) && (stats[string(field)*"_seconds"]=getproperty(timing,field))
    end
    N.atomic_toml(joinpath(cfg.out,"benchmark_timing.toml"),stats)
    batch_log("BENCHMARK DONE: $(stats)")
    timing.value
end
# Snapshot the implementation when it is loaded. Uploading new source files
# while a job is running must not relabel the already-loaded implementation.
const VARIANCE_CODE_SHA256=variance_code_sha256()

function completed_variance_case(job,settings)
    marker=joinpath(job.out,"completed.toml")
    isfile(marker) || return nothing
    data=TOML.parsefile(marker)
    data["settings"]==settings || error("Completed case has different settings/state/code; use a new output root: $(job.out)")
    for name in ("variance.json","variance.toml","variance_by_L.csv","connected_terms.csv","timing_by_anchor.csv")
        isfile(joinpath(job.out,name)) || error("Completed marker exists but $name is missing in $(job.out)")
    end
    data
end

function update_variance_summaries(out,records)
    summary=Dict[];anchors=Dict[]
    for data in records
        s=data["settings"];r=data["result"];t=data["run_timing"]
        base=Dict("D"=>s["D"],"chi"=>s["chi"],"contraction_Lmax"=>s["L"],"sweeps"=>s["sweeps"],
            "exchange_energy_per_spin"=>r["exchange_energy_per_spin"],"ctm_converged"=>r["ctm"]["converged"],
            "ctm_iterations"=>r["ctm"]["iterations"],"ctm_error"=>r["ctm"]["error"],
            "actual_corner_dimensions"=>join(r["ctm"]["actual_corner_dimensions"],";"),
            "all_L_case_wall_seconds_shared"=>t["case_wall_seconds"],"norm_ctm_seconds_shared"=>t["norm_ctm_seconds_shared"],
            "center_seconds"=>sum(a["center_seconds"] for a in r["timing"]["anchors"]),
            "sweep_seconds"=>sum(a["sweep_seconds"] for a in r["timing"]["anchors"]),
            "measurement_seconds"=>sum(a["measurement_seconds"] for a in r["timing"]["anchors"]),
            "started_utc"=>t["started_utc"],"finished_utc"=>t["finished_utc"],"case_directory"=>data["case_directory"])
        for row in r["cumulative"]
            push!(summary,merge(base,Dict("L"=>row["window_size"]),row))
        end
        for row in r["timing"]["anchors"]
            push!(anchors,merge(Dict("D"=>s["D"],"chi"=>s["chi"],"contraction_Lmax"=>s["L"]),row))
        end
    end
    N.write_csv(joinpath(out,"summary.csv"),summary)
    N.write_csv(joinpath(out,"variance_by_L_all.csv"),summary)
    N.write_csv(joinpath(out,"timing_by_anchor_all.csv"),anchors)
end

function run_variance_case(cfg,job,settings,f,prepared,shared)
    mkpath(job.out)
    settingsfile=joinpath(job.out,"settings.toml")
    if isfile(settingsfile)
        TOML.parsefile(settingsfile)==settings || error("Unfinished case has changed settings/state/code; use a new output root: $(job.out)")
    else
        N.atomic_toml(settingsfile,settings)
    end
    started=batch_utc();clock=time_ns()
    batch_log("START D=$(job.D) chi=$(job.chi) contraction Lmax=$(job.L); extract L=$(cfg.sizes)")
    result=exchange_variance(f;L=job.L,window_sizes=cfg.sizes,chi=job.chi,
        ctm_iterations=cfg.ctm_iterations,ctm_tolerance=cfg.ctm_tolerance,
        sweeps=cfg.sweeps,source_shift=cfg.source_shift,norm_environment=prepared,verbose=true)
    Plan.file_sha256(job.checkpoint)==settings["source_sha256"] || error("Source state changed during measurement")
    result["source_checkpoint"]=job.checkpoint;result["source_sha256"]=settings["source_sha256"]
    result["D"]=job.D;result["settings"]=settings
    N.write_json(joinpath(job.out,"variance.json"),result)
    N.write_csv(joinpath(job.out,"connected_terms.csv"),result["connected_terms"])
    N.write_csv(joinpath(job.out,"variance_by_L.csv"),result["cumulative"])
    N.write_csv(joinpath(job.out,"timing_by_anchor.csv"),result["timing"]["anchors"])
    N.atomic_toml(joinpath(job.out,"variance.toml"),result)
    timing=merge(shared,Dict("started_utc"=>started,"finished_utc"=>batch_utc(),
        "case_wall_seconds"=>batch_seconds(clock),"compute_seconds"=>result["seconds"],
        "note"=>"case time includes variance calculation and result export; shared load/factors/norm CTM counted separately; first calls may compile"))
    brief=Dict(k=>result[k] for k in ("D","L","chi","sweeps","exchange_energy_per_spin","ctm","timing","cumulative","full_anchor_sum","anchor_count","n_physical_sites"))
    data=Dict("settings"=>settings,"result"=>brief,"run_timing"=>timing,"case_directory"=>job.out)
    N.atomic_toml(joinpath(job.out,"run_timing.toml"),timing)
    N.atomic_toml(joinpath(job.out,"completed.toml"),data) # completion marker LAST
    batch_log("DONE D=$(job.D) chi=$(job.chi); extracted L=$(cfg.sizes); seconds=$(round(timing["case_wall_seconds"];digits=3)); E=$(result["exchange_energy_per_spin"]); variance=$(last(result["cumulative"])["variance_per_spin"])")
    data
end

function run_variance_D(cfg,jobs,code_hash,records)
    file=first(jobs).checkpoint;source_hash=Plan.file_sha256(file)
    settings=[Plan.case_settings(cfg,j,source_hash,code_hash) for j in jobs]
    completed=[completed_variance_case(j,s) for (j,s) in zip(jobs,settings)]
    if all(x->!isnothing(x),completed)
        for data in completed;push!(records,data);end
        update_variance_summaries(cfg.out,records)
        batch_log("SKIP completed D=$(first(jobs).D); saved results/settings/hash verified")
        return
    end
    batch_log("LOAD D=$(first(jobs).D): $file")
    clock=time_ns();state,_=N.KagomeU1Cells.load_u1_checkpoint(file);load_seconds=batch_seconds(clock)
    actual=maximum(dim(space(getfield(c,n),leg)) for c in state.cells for n in N.BS.B_FIELDS for leg in 1:2)
    actual==first(jobs).D || error("Checkpoint label D=$(first(jobs).D), actual maximum virtual dimension=$actual")
    state.geometry.pattern==:sqrt3xsqrt3 || error("This batch is configured for the sqrt3xsqrt3 cell")
    clock=time_ns();f=N.build_factors(state);factor_seconds=batch_seconds(clock)
    for chi in cfg.chis
        indices=findall(i->jobs[i].chi==chi,eachindex(jobs))
        pending=filter(i->isnothing(completed[i]),indices)
        for i in indices
            isnothing(completed[i]) || push!(records,completed[i])
        end
        if !isempty(pending)
            # Scoped helper releases this chi's large environment before the next.
            run_variance_chi(cfg,jobs,settings,completed,pending,f,load_seconds,factor_seconds,records)
        end
        update_variance_summaries(cfg.out,records)
        GC.gc()
    end
end

function run_variance_chi(cfg,jobs,settings,completed,pending,f,load_seconds,factor_seconds,records)
    chi=jobs[first(pending)].chi
    batch_log("NORM CTMRG D=$(jobs[first(pending)].D) chi=$chi; shared by all extracted L values")
    clock=time_ns()
    env,info=N.native_ctmrg(f;chi,iterations=cfg.ctm_iterations,tolerance=cfg.ctm_tolerance,verbose=true)
    norm_seconds=batch_seconds(clock)
    shared=Dict("source_load_seconds_shared"=>load_seconds,"factor_seconds_shared"=>factor_seconds,"norm_ctm_seconds_shared"=>norm_seconds)
    prepared=(f=f,environment=env,ctm=info,chi=chi,iterations=cfg.ctm_iterations,tolerance=cfg.ctm_tolerance)
    N.atomic_toml(joinpath(cfg.out,"D$(jobs[first(pending)].D)","chi$chi","norm_ctm_timing.toml"),
        merge(shared,Dict("ctm"=>info,"finished_utc"=>batch_utc())))
    for i in pending
        data=run_variance_case(cfg,jobs[i],settings[i],f,prepared,shared)
        push!(records,data)
        update_variance_summaries(cfg.out,records) # save summary after EVERY case
        GC.gc()
    end
end

function run_variance_batch(cfg;startup_seconds=0.)
    jobs=Plan.variance_jobs(cfg);mkpath(cfg.out)
    code_hash=VARIANCE_CODE_SHA256;started=batch_utc();clock=time_ns()
    batch_log("BATCH START: $(length(jobs)) contractions; $(length(jobs)*length(cfg.sizes)) L estimates; CPU; $(cfg.threads) threads; all 18 central bonds")
    BLAS.set_num_threads(cfg.threads);N.prepare_device!("cpu")
    N.atomic_toml(joinpath(cfg.out,"batch_run.toml"),Dict("started_utc"=>started,
        "module_load_seconds"=>startup_seconds,"planned_cases"=>length(jobs),"code_sha256"=>code_hash,
        "dims"=>cfg.dims,"chis"=>cfg.chis,"sizes"=>cfg.sizes,"contraction_size"=>cfg.contraction_size,"threads"=>cfg.threads))
    run_id=Dates.format(Dates.now(Dates.UTC),dateformat"yyyymmddTHHMMSSsss")*"_"*string(getpid())
    run_dir=joinpath(cfg.out,"runs",run_id);mkpath(run_dir)
    records=Dict[]
    for D in cfg.dims
        d_started=batch_utc();d_clock=time_ns();n_before=length(records)
        batch_log("D START: D=$D")
        run_variance_D(cfg,filter(j->j.D==D,jobs),code_hash,records)
        d_seconds=batch_seconds(d_clock)
        N.atomic_toml(joinpath(run_dir,"D$D.toml"),Dict("D"=>D,"started_utc"=>d_started,
            "finished_utc"=>batch_utc(),"D_wall_seconds"=>d_seconds,
            "completed_cases"=>length(records)-n_before,"run_id"=>run_id,
            "note"=>"This invocation's wall time, including shared CTMRG, calculation, I/O and resume checks"))
        batch_log("D DONE: D=$D; wall seconds=$(round(d_seconds;digits=3))")
        GC.gc()
    end
    elapsed=batch_seconds(clock)
    run_info=Dict("started_utc"=>started,
        "finished_utc"=>batch_utc(),"batch_wall_seconds"=>elapsed,
        "module_load_seconds"=>startup_seconds,"completed_cases"=>length(records),
        "L_estimates"=>length(records)*length(cfg.sizes),"run_id"=>run_id)
    N.atomic_toml(joinpath(run_dir,"completed.toml"),run_info)
    N.atomic_toml(joinpath(cfg.out,"batch_completed.toml"),run_info)
    batch_log("BATCH DONE: $(length(records)) cases; wall seconds=$(round(elapsed;digits=3)); module load seconds=$startup_seconds")
    records
end
