# Pure configuration parsing, also usable by lightweight tests without TensorKit.
struct StarRunConfig
    Lx::Int
    Ly::Int
    J::Float64
    stages::Vector{NamedTuple{(:steps,:dt,:Dmax),Tuple{Int,Float64,Int}}}
    output_dir::String
    identity_sweeps::Int
    checkpoint_every::Int
    keep_every::Int
    local_energy_every::Int
    blas_threads::Int
    max_seconds::Float64
    ctm_enabled::Bool
    ctm_chis::Vector{Int}
    ctm_iterations::Int
    ctm_tol::Float64
    ctm_energy_tol::Float64
    ctm_threads::Int
    raw::Dict{String,Any}
end

function _known_keys(table,allowed,label)
    extra=setdiff(Set(keys(table)),Set(allowed));
    isempty(extra) || throw(ArgumentError("unknown $label keys: $(join(sort!(collect(extra)),", "))"));
end

function read_star_config(path;output_dir=nothing)
    raw=TOML.parsefile(path);
    _known_keys(raw,("run","schedule","ctm"),"top-level");
    r=get(raw,"run",Dict{String,Any}());c=get(raw,"ctm",Dict{String,Any}());
    _known_keys(r,("Lx","Ly","J","Dmax","output_dir","identity_sweeps",
        "checkpoint_every","keep_every","local_energy_every","blas_threads","max_seconds"),"run");
    _known_keys(c,("enabled","chis","iterations","tol","energy_tol","blas_threads"),"ctm");
    lx=Int(get(r,"Lx",4));ly=Int(get(r,"Ly",4));
    lx>=2 && ly>=2 && iseven(lx) && iseven(ly) || throw(ArgumentError("cell dimensions must be even and >=2"));
    cap=Int(get(r,"Dmax",24));cap>=9 || throw(ArgumentError("Dmax must be >=9 for the exact initial state"));
    stages=NamedTuple{(:steps,:dt,:Dmax),Tuple{Int,Float64,Int}}[];
    for s in get(raw,"schedule",Any[])
        _known_keys(s,("steps","dt","Dmax"),"schedule");
        steps=Int(s["steps"]);dt=Float64(s["dt"]);d=Int(get(s,"Dmax",cap));
        steps>0 && isfinite(dt) && dt>0 && d>=9 || throw(ArgumentError("require steps>0, finite dt>0, Dmax>=9"));
        push!(stages,(steps=steps,dt=dt,Dmax=d));
    end
    isempty(stages) && throw(ArgumentError("at least one [[schedule]] stage is required"));
    out=isnothing(output_dir) ? get(r,"output_dir","server_runs/star_D24") : output_dir;
    out=abspath(isabspath(out) ? out : joinpath(dirname(abspath(path)),out));
    gauge=Int(get(r,"identity_sweeps",0));checkpoint=Int(get(r,"checkpoint_every",20));
    keep=Int(get(r,"keep_every",200));energy=Int(get(r,"local_energy_every",100));
    threads=parse(Int,get(ENV,"SU4_STAR_BLAS_THREADS",string(get(r,"blas_threads",1))));
    seconds=Float64(get(r,"max_seconds",0.));j=Float64(get(r,"J",1.));
    gauge>=0 && checkpoint>0 && keep>=0 && energy>=0 && threads>0 &&
        isfinite(seconds) && seconds>=0 && isfinite(j) || throw(ArgumentError("invalid run controls"));
    enabled=Bool(get(c,"enabled",true));chis=Int.(get(c,"chis",[48,64,96]));
    !isempty(chis) && all(>(0),chis) && issorted(chis) && allunique(chis) || throw(ArgumentError("ctm.chis must be positive and strictly increasing"));
    iterations=Int(get(c,"iterations",100));tol=Float64(get(c,"tol",1e-7));etol=Float64(get(c,"energy_tol",1e-6));
    ct=Int(get(c,"blas_threads",threads));
    iterations>0 && isfinite(tol) && tol>0 && isfinite(etol) && etol>0 && ct>0 || throw(ArgumentError("invalid CTMRG controls"));
    enabled && !((lx,ly) in ((2,2),(4,4))) && throw(ArgumentError("the existing CTMRG interface requires 2x2 or 4x4"));
    return StarRunConfig(lx,ly,j,stages,out,gauge,checkpoint,keep,energy,threads,seconds,
        enabled,chis,iterations,tol,etol,ct,raw)
end

star_physics(c::StarRunConfig)=(Lx=c.Lx,Ly=c.Ly,J=c.J,initial="strict_star_D9",
    identity_sweeps=c.identity_sweeps,stages=c.stages,convention="permutation");

function describe_star_run(c::StarRunConfig)
    println("Ansatz origin: exact SU(4) star PEPS, V=1+4+4bar, D=9; cell=",(c.Lx,c.Ly));
    println("Schedule: ",c.stages,"; total imaginary time=",sum(s.steps*s.dt for s in c.stages));
    println("Output: ",c.output_dir,"; checkpoint every ",c.checkpoint_every," sweeps");
    println("Evolution BLAS threads: ",c.blas_threads,"; automatic CTMRG: ",c.ctm_enabled);
end
