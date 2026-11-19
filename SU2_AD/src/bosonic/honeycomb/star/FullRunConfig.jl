using TOML

struct HoneycombFullRunConfig
    Lx::Int
    Ly::Int
    J::Float64
    Dmax::Int
    space_policy::Symbol
    initial::Symbol
    dense_D::Int
    seed::Int
    stages::Vector{NamedTuple{(:steps,:dt),Tuple{Int,Float64}}}
    output_dir::String
    chi::Int
    ctm_iterations::Int
    ctm_tol::Float64
    blas_threads::Int
    optimizer::NamedTuple
    raw::Dict{String,Any}
end

function read_full_config(path;output_dir=nothing)
    raw=TOML.parsefile(path);
    function checked(table,allowed,label)
        extra=setdiff(keys(table),allowed);isempty(extra) || error("unknown $label keys: $extra");table
    end
    checked(raw,("run","schedule","ctm","optimizer"),"full-update");
    r=checked(get(raw,"run",Dict()),("Lx","Ly","J","Dmax","space_policy","initial","dense_D","seed","output_dir","blas_threads"),"run");
    c=checked(get(raw,"ctm",Dict()),("chi","iterations","tol"),"ctm");
    o=checked(get(raw,"optimizer",Dict()),("maxiter","gradient_tolerance","loss_tolerance","initial_step","verbose"),"optimizer");
    nx=Int(get(r,"Lx",4));ny=Int(get(r,"Ly",4));
    nx in (2,4) && ny in (2,4) && nx==ny || error("existing CTMRG requires a 2x2 or 4x4 cell");
    policy=Symbol(get(r,"space_policy","fixed_space"));
    policy in (:fixed_space,:tensorkit) || error("space_policy must be fixed_space or tensorkit");
    initial=Symbol(get(r,"initial","star"));initial in (:star,:dense) || error("initial must be star or dense");
    cap=Int(get(r,"Dmax",9));d=Int(get(r,"dense_D",1));
    cap>=1 && 1<=d<=cap || error("invalid Dmax/dense_D");
    initial==:star && cap<9 && error("the original star state requires Dmax >= 9");
    J=Float64(get(r,"J",1.));isfinite(J) || error("J must be finite");
    stages=NamedTuple{(:steps,:dt),Tuple{Int,Float64}}[];
    for q in get(raw,"schedule",Any[])
        checked(q,("steps","dt"),"schedule");n=Int(q["steps"]);dt=Float64(q["dt"]);
        n>0 && isfinite(dt) && dt>0 || error("invalid schedule");push!(stages,(steps=n,dt=dt));
    end
    isempty(stages) && error("a schedule is required");
    chi=Int(get(c,"chi",16));iters=Int(get(c,"iterations",100));tol=Float64(get(c,"tol",1e-7));
    threads=Int(get(r,"blas_threads",1));
    chi>0 && iters>0 && isfinite(tol) && tol>0 && threads>0 || error("invalid CTMRG/thread controls");
    optimizer=(maxiter=Int(get(o,"maxiter",20)),gradient_tolerance=Float64(get(o,"gradient_tolerance",1e-8)),
        loss_tolerance=Float64(get(o,"loss_tolerance",1e-12)),initial_step=Float64(get(o,"initial_step",.2)),
        verbose=Bool(get(o,"verbose",false)));
    optimizer.maxiter>=1 && all(isfinite,(optimizer.gradient_tolerance,optimizer.loss_tolerance,optimizer.initial_step)) &&
        optimizer.gradient_tolerance>=0 && optimizer.loss_tolerance>=0 && optimizer.initial_step>0 || error("invalid optimizer settings");
    out=isnothing(output_dir) ? String(get(r,"output_dir","../server_runs/full_fixed_D9")) : output_dir;
    out=abspath(joinpath(dirname(abspath(path)),out));
    return HoneycombFullRunConfig(nx,ny,J,cap,policy,initial,d,Int(get(r,"seed",142)),stages,out,chi,iters,tol,threads,optimizer,raw)
end

function full_physics(c)
    return Dict("Lx"=>c.Lx,"Ly"=>c.Ly,"J"=>c.J,"Dmax"=>c.Dmax,"space_policy"=>String(c.space_policy),
        "initial"=>String(c.initial),"dense_D"=>c.dense_D,"seed"=>c.seed,"chi"=>c.chi,
        "ctm_iterations"=>c.ctm_iterations,"ctm_tol"=>c.ctm_tol,"optimizer"=>Dict(pairs(c.optimizer)),
        "schedule"=>[Dict("steps"=>s.steps,"dt"=>s.dt) for s in c.stages],"coordinate_system"=>"ctm_down")
end

function describe_full_run(c)
    println("Honeycomb two-site full update; initial=",c.initial,"; cell=",(c.Lx,c.Ly));
    println("Space policy: ",c.space_policy,"; Dmax=",c.Dmax,"; CTMRG chi=",c.chi);
    println("Schedule: ",c.stages,"; CTMRG rebuilt after every bond; energy in the same job");
    println("Output: ",c.output_dir);flush(stdout);
end
