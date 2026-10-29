const STAR_HELP="""
SU(4) honeycomb: exact D=9 star PEPS -> two-site simple update.
The standard configuration runs SU and then CTMRG observables in the SAME job.

  julia --startup-file=no run_star_simple_update.jl [options]

  --config FILE    TOML configuration (default: ../config/StarSimpleUpdate.toml)
  --output DIR     Override output directory (relative to the configuration file)
  --resume         Resume SU, then CTMRG; completed SU is not repeated
  --auto-resume    Resume if state_latest.jld2 exists (the sh launcher enables this)
  --init-checkpoint FILE   Start a NEW output/schedule from a saved server checkpoint
  --measure-only   Optional diagnostic: CTMRG on a saved checkpoint only
  --dry-run        Validate and print configuration, without constructing tensors
  --help           Print this message

Uses the existing shared source library, selected with SU4_SRC_DIR / SU2_AD_ROOT if needed.
No SU2_AD directory name or fixed relative location is required.
JULIA_PROJECT is honored; packages are never installed or updated by this script.
""";

function star_source_dir(;entry_dir=@__DIR__,work_dir=pwd(),env=ENV)
    function probe(base)
        base=abspath(expanduser(base));
        for suffix in ("", "src/bosonic/honeycomb/star", "bosonic/honeycomb/star",
                       "honeycomb/star", "star")
            path=normpath(joinpath(base,suffix));
            if isfile(joinpath(path,"StarRunConfig.jl")) &&
               isfile(joinpath(path,"StarSimpleUpdateRun.jl")) &&
               isfile(joinpath(path,"../Honeycomb_SU4.jl"))
                return path
            end
        end
        return nothing
    end
    for name in ("SU4_SRC_DIR","SU2_AD_ROOT")
        value=get(env,name,"");
        if !isempty(value)
            path=probe(value);
            isnothing(path) && error("$name=$value does not contain the honeycomb source library. Point it to src, the project root, or bosonic/honeycomb/star.");
            return path
        end
    end
    seen=Set{String}();
    for start in (entry_dir,work_dir)
        base=abspath(start);
        for _=1:8
            if !(base in seen)
                push!(seen,base);path=probe(base);
                isnothing(path) || return path
            end
            parent=dirname(base);parent==base && break;base=parent;
        end
    end
    error("Cannot find the shared honeycomb library. Set SU4_SRC_DIR=/path/to/your/existing/src; no source copy is needed.");
end

function star_default_config(;entry_dir=@__DIR__,work_dir=pwd())
    paths=(joinpath(entry_dir,"config/StarSimpleUpdate.toml"),
        joinpath(entry_dir,"StarSimpleUpdate.toml"),
        joinpath(entry_dir,"../config/StarSimpleUpdate.toml"),
        joinpath(work_dir,"config/StarSimpleUpdate.toml"),joinpath(work_dir,"StarSimpleUpdate.toml"));
    for path in paths
        isfile(path) && return normpath(abspath(path))
    end
    error("Configuration not found; pass --config /path/to/StarSimpleUpdate.toml");
end

function star_cli(args)
    config=nothing;output=nothing;
    resume=false;measure=false;dry=false;initial=nothing;auto=false;i=1;
    while i<=length(args)
        a=args[i];
        if a in ("--config","--output","--init-checkpoint")
            i<length(args) || error("$a requires a value");
            if a=="--config";config=abspath(args[i+1]);
            elseif a=="--output";output=args[i+1];
            else;initial=abspath(args[i+1]);end
            i+=2;
        elseif a=="--resume"
            resume=true;i+=1;
        elseif a=="--auto-resume"
            auto=true;i+=1;
        elseif a=="--measure-only"
            measure=true;i+=1;
        elseif a=="--dry-run"
            dry=true;i+=1;
        else
            error("Unknown option: $a. Use --help.");
        end
    end
    isnothing(config) && (config=star_default_config());
    !isnothing(initial) && (resume || measure) && error("--init-checkpoint requires a new run, not --resume/--measure-only");
    return (;config,output,resume,measure,dry,initial,auto)
end

star_should_resume(options,c)=options.resume || (options.auto && isnothing(options.initial) &&
    isfile(joinpath(c.output_dir,"state_latest.jld2")));

if abspath(PROGRAM_FILE)==@__FILE__
    if "--help" in ARGS
        println(STAR_HELP);
    else
        options=star_cli(ARGS);
        source_dir=star_source_dir();
        # Configuration-only dry runs do not load TensorKit or build CGCs.
        if options.dry
            using TOML
            include(joinpath(source_dir,"StarRunConfig.jl"));
            describe_star_run(read_star_config(options.config;output_dir=options.output));
            isnothing(options.initial) || println("Continue from checkpoint (not loaded in dry-run): ",options.initial);
        else
            include(joinpath(source_dir,"StarSimpleUpdateRun.jl"));
            using .HoneycombStarRun
            c=HoneycombStarRun.read_star_config(options.config;output_dir=options.output);
            if options.measure
                rows=measure_star_server(c);
                any(r->r.chi_stable,rows) || exit(2);
            else
                status=run_star_server(c;resume=star_should_resume(options,c),initial_checkpoint=options.initial);
                if status==:finished && c.ctm_enabled
                    rows=measure_star_server(c);
                    any(r->r.chi_stable,rows) || exit(2);
                end
            end
        end
    end
end
