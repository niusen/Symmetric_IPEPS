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

Requires the complete repository and the Julia environment used by its SU(4) code.
JULIA_PROJECT is honored; packages are never installed or updated by this script.
""";

function star_cli(args)
    config=joinpath(@__DIR__,"../config/StarSimpleUpdate.toml");output=nothing;
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
        # Configuration-only dry runs do not load TensorKit or build CGCs.
        if options.dry
            using TOML
            include(joinpath(@__DIR__,"../../../src/bosonic/honeycomb/star/StarRunConfig.jl"));
            describe_star_run(read_star_config(options.config;output_dir=options.output));
            isnothing(options.initial) || println("Continue from checkpoint (not loaded in dry-run): ",options.initial);
        else
            include(joinpath(@__DIR__,"../../../src/bosonic/honeycomb/star/StarSimpleUpdateRun.jl"));
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
