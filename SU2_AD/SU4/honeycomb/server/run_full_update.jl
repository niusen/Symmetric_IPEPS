# Reuse the source locator and common CLI from the existing single-job launcher.
include("run_star_simple_update.jl")
const FULL_HELP="""
Honeycomb SU4 two-site full update, followed by CTMRG energy in the same job.
  sh run_full_update.sh [--config FILE] [--resume] [--dry-run]
  --auto-resume / --output DIR / --init-checkpoint FILE / --measure-only
space_policy = "fixed_space" or "tensorkit" in the TOML configuration.
Uses the existing shared src and the user's existing Julia environment.
""";

function full_default_config()
    for p in (joinpath(@__DIR__,"../config/FullUpdate.toml"),joinpath(@__DIR__,"config/FullUpdate.toml"),
              joinpath(@__DIR__,"FullUpdate.toml"),joinpath(pwd(),"config/FullUpdate.toml"))
        isfile(p) && return abspath(p)
    end
    error("FullUpdate.toml not found; use --config")
end

function full_main(args)
    "--help" in args && (println(FULL_HELP);return :help);
    options=star_cli("--config" in args ? args : vcat(["--config",full_default_config()],args));
    source=star_source_dir();
    if !isdefined(@__MODULE__,:HoneycombFullRunConfig)
        include(joinpath(source,"FullRunConfig.jl"));
    end
    config=Base.invokelatest(read_full_config,options.config;output_dir=options.output);
    options.dry && (Base.invokelatest(describe_full_run,config);return :preview);
    include(joinpath(source,"FullUpdateRun.jl"));
    c=Base.invokelatest(HoneycombFullRun.read_full_config,options.config;output_dir=options.output);
    auto=options.auto && isfile(joinpath(c.output_dir,"state_latest.jld2"));
    return Base.invokelatest(HoneycombFullRun.run_full_server,c;
        resume=options.resume || auto,initial_checkpoint=options.initial,measure_only=options.measure)
end

if abspath(PROGRAM_FILE)==@__FILE__
    full_main(ARGS);
end
