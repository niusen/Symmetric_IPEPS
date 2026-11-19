# New continuation orchestration; reuse the original U1 SU, checkpoint,
# TensorKit CTMRG and complete observable functions without modifying them.
if !isdefined(@__MODULE__, :u1_server_evolve_D)
    include(joinpath(@__DIR__, "u1_server_workflow.jl"))
end

function u1_continuation_plan(cfg)
    isempty(cfg.dims) && error("continuation D sequence is empty")
    first(cfg.dims)==cfg.source_D+1 && cfg.dims==collect(first(cfg.dims):last(cfg.dims)) ||
        error("continuation must start at source_D+1 and increase D one at a time")
    isfile(cfg.initial_state) || error("missing saved U1 initial state: $(cfg.initial_state)")
    digest=bytes2hex(sha256(read(cfg.initial_state)))
    digest==lowercase(cfg.initial_sha256) || error("saved initial state SHA256 mismatch")
    source=load_u1_seed(cfg.initial_state)
    ServerU1.check_u1(source;Dmax=cfg.source_D)
    actual=maximum(dim(space(getfield(c,n),leg)) for c in source.cells for n in ServerBase.B_FIELDS for leg in 1:2)
    actual==cfg.source_D || error("initial state's maximum virtual dimension $actual is not source_D=$(cfg.source_D)")
    if endswith(lowercase(cfg.initial_state),".jld2")
        _,data=ServerU1.load_u1_checkpoint(cfg.initial_state)
        metadata=get(data,"metadata",Dict{String,Any}())
        get(metadata,"Dmax",cfg.source_D)==cfg.source_D || error("source checkpoint Dmax mismatch")
        get(metadata,"h",cfg.h)==cfg.h || error("source checkpoint magnetic field mismatch")
    end
    # Do not allow either the output root or a new D folder to contain the
    # original source checkpoint. The source tree stays read-only.
    source_path=realpath(cfg.initial_state)
    output=abspath(cfg.out)
    for candidate in [output; [joinpath(output,"D$D") for D in cfg.dims]]
        rel=relpath(source_path,candidate)
        (isabspath(rel) || rel==".." || startswith(rel,".."*string(Base.Filesystem.path_separator))) ||
            error("new output directory must not contain the saved source state")
    end
    plan=u1_server_plan(cfg)
    merge!(plan,Dict("format"=>"u1_continuation_plan_v1","source_D"=>cfg.source_D,
                    "initialization"=>"saved_U1_final_state","initial_sha256"=>digest))
    plan
end

function u1_continuation_check_parent(folder,input,plan_id,D)
    for name in ("latest.jld2","initial.jld2","final.jld2")
        path=joinpath(folder,name)
        isfile(path) || continue
        _,data=ServerU1.load_u1_checkpoint(path)
        metadata=data["metadata"]
        metadata["plan_id"]==plan_id && metadata["Dmax"]==D || error("continuation checkpoint settings mismatch")
        metadata["initial_source_sha256"]==bytes2hex(sha256(read(input))) ||
            error("parent state changed for D=$D; refusing to mix continuation branches")
    end
end

function u1_continuation_run_locked(cfg,plan)
    planpath=joinpath(cfg.out,"configuration.toml")
    if isfile(planpath)
        TOML.parsefile(planpath)==plan || error("settings/seed/version changed; choose a new out directory")
    else
        server_atomic_toml(planpath,plan)
    end
    plan_id=bytes2hex(sha256(read(planpath)))
    BLAS.set_num_threads(cfg.threads)
    ServerNative.prepare_device!(cfg.device)
    println("U1 continuation: READ saved D=$(cfg.source_D); evolve D=$(cfg.dims); no random initialization")
    println("Source: ",cfg.initial_state," SHA256=",plan["initial_sha256"])
    println("h=$(cfg.h), sqrt3xsqrt3, chi=$(cfg.chi), CTMRG <=$(cfg.ctm_iterations), $(cfg.device)")
    println("Full Trotter steps per NEW D: ",sum(round(Int,s.tau/s.dt) for s in cfg.schedule),
            "; ONE final CTMRG + complete observables per D; fixed SU schedule")
    flush(stdout)
    input=cfg.initial_state
    source_hash=plan["initial_sha256"]
    for D in cfg.dims
        folder=joinpath(cfg.out,"D$D")
        u1_continuation_check_parent(folder,input,plan_id,D)
        parent_hash=bytes2hex(sha256(read(input)))
        println("D=$D loads D=$(D-1) parent SHA256=$parent_hash");flush(stdout)
        final=u1_server_evolve_D(cfg,D,input,folder,plan_id)
        u1_continuation_check_parent(folder,input,plan_id,D)
        server_atomic_toml(joinpath(folder,"lineage.toml"),Dict("D"=>D,"parent_D"=>D-1,
            "parent_sha256"=>parent_hash,"source_D"=>cfg.source_D,
            "source_sha256"=>source_hash,"final_sha256"=>bytes2hex(sha256(read(final)))))
        ServerNative.measure_checkpoint(final,joinpath(folder,"ctm");representation=:u1,h=cfg.h,
            chi=cfg.chi,iterations=cfg.ctm_iterations,tolerance=cfg.ctm_tolerance,
            device=cfg.device,threads=cfg.threads,D=D)
        server_collect_outputs(cfg.out;reference_file=cfg.dense_reference,evolution_symmetry=:u1)
        ServerNative.CTM.square_J1_reclaim_device_memory!()
        input=final
    end
    bytes2hex(sha256(read(cfg.initial_state)))==source_hash || error("source checkpoint unexpectedly changed")
    println("DONE: ",joinpath(cfg.out,"summary.csv"));flush(stdout)
    cfg.out
end

function run_u1_continuation(cfg)
    plan=u1_continuation_plan(cfg)
    mkpath(cfg.out)
    if Sys.isunix()
        open(joinpath(cfg.out,"writer.lock"),"a+") do io
            if ccall(:flock,Cint,(Cint,Cint),fd(io),6)!=0
                seekstart(io);owner=strip(read(io,String))
                error("another live U1 continuation owns $(cfg.out): $owner; stop it first, do not delete its lock")
            end
            try
                seekstart(io);truncate(io,0);println(io,"PID=",getpid());flush(io)
                u1_continuation_run_locked(cfg,plan)
            finally
                ccall(:flock,Cint,(Cint,Cint),fd(io),8)
            end
        end
    else
        FileWatching.Pidfile.mkpidlock(joinpath(cfg.out,"writer.pid");wait=false) do
            u1_continuation_run_locked(cfg,plan)
        end
    end
end
