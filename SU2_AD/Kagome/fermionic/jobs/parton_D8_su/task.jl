if abspath(PROGRAM_FILE)==@__FILE__
    println("node=",gethostname(),"  pid=",getpid(),"  task=",join(ARGS," "))
    flush(stdout)
end

using LinearAlgebra

include("settings.jl")
const ROOT=normpath(joinpath(@__DIR__,"../../../.."))
const SEEDS=abspath(get(ENV,"KAGOME_SEED_ROOT",joinpath(ROOT,"Kagome/fermionic/iPESS/results/gfpeps_validation")))
const RUN_ROOT=abspath(get(ENV,"KAGOME_RUN_ROOT",joinpath(ROOT,"Kagome/fermionic/iPESS/results/parton_D8_su")))
const RUNS=joinpath(RUN_ROOT,join(UNIT_CELL,"x"))
const KAGOME_SRC=joinpath(ROOT,"src/fermionic/kagome")
include(joinpath(KAGOME_SRC,"workflows/parton_su_io.jl"))

caseid(c)=string(c[1],"_",c[2])
casedir(c)=joinpath(RUNS,caseid(c))
function findcase(family,model)
    i=findfirst(c->(c[1],c[2])==(Symbol(family),Symbol(model)),CASES)
    isnothing(i) && error("choose ipeps|ipess and orig|mild|zero")
    CASES[i]
end

function check_seed(c)
    family,model,folder,name,source_hash,mat_hash=c
    read_parton_seed(joinpath(SEEDS,folder,"tensors.mat"); family,model,name,source_hash,mat_hash,D=8)
end

function read_case(c)
    expected=(;family=c[1],model=c[2],Dmax=8,t=(1.,1.),onsite_U=0.,schedule=SU_SCHEDULE,
              MEASUREMENT...,trun_tol=SU_TRUNCATION_TOL)
    result=read_parton_measurements(casedir(c),expected;mat_hash=c[6],source_hash=c[5],initial_D=8)
    result.a.cell==UNIT_CELL || error("expected cell $UNIT_CELL in $(casedir(c)), found $(result.a.cell)")
    result
end

function case_report(c)
    a,b,p=read_case(c)
    write_parton_comparison(casedir(c),a,b,p)
end

function check_settings()
    Set((c[1],c[2]) for c in CASES)==Set(Iterators.product((:ipeps,:ipess),(:orig,:mild,:zero))) &&
        length(CASES)==6 || error("expected the six distinct jobs")
    BLAS_THREADS>0 && MEASUREMENT.chi>0 && MEASUREMENT.maxiter>0 && MEASUREMENT.tol>0 &&
        0<=SU_TRUNCATION_TOL<1 || error("invalid SU/CTMRG settings")
    !isempty(SU_SCHEDULE) || error("empty SU schedule")
    for (tau,dt) in SU_SCHEDULE
        isfinite(tau) && isfinite(dt) && tau>=dt>0 && isapprox(round(tau/dt)*dt,tau;rtol=1e-10) ||
            error("each stage needs a positive integral number of steps: $((tau,dt))")
    end
end

function run_case(c)
    check_settings()
    seed=check_seed(c)
    dest=casedir(c)
    if ispath(dest)
        if all(isfile(joinpath(dest,stage*".jld2")) for stage in ("initial","final"))
            report=case_report(c)
            println("Existing result ",caseid(c)," (",report.status,"): ",joinpath(dest,"comparison.tsv"))
            return nothing
        end
        error("partial output in $dest; inspect its log and state.jld2, or rerun in a fresh KAGOME_RUN_ROOT")
    end
    BLAS.set_num_threads(BLAS_THREADS)
    println("Starting ",caseid(c)," seed=",seed.name," Dmax=8 unprojected; μ=",seed.fields,
            " schedule=",SU_SCHEDULE," CTM=",MEASUREMENT," BLAS=",BLAS.get_num_threads())
    flush(stdout)
    # Load the SU/CTMRG code after checking this case's input.
    if !isdefined(@__MODULE__,:KagomeIPEPSSU)
        include(joinpath(KAGOME_SRC,"ipeps/bootstrap.jl"))
    end
    Base.invokelatest(KagomeIPEPSSU.run_native_su,seed.family,seed.path,dest,seed.model;
        schedule=SU_SCHEDULE,Dmax=8,unitcell=UNIT_CELL,mu=seed.fields,t=(1.,1.),onsite_U=0.,
        trun_tol=SU_TRUNCATION_TOL,MEASUREMENT...,
        provenance=(;seed.name,seed.source_hash,seed.mat_hash,projected=false,blas_threads=BLAS_THREADS))
    case_report(c)
    println("Comparison: ",joinpath(dest,"comparison.tsv"))
    nothing
end

function summarize_cases()
    file=joinpath(RUNS,"summary.tsv")
    columns=("SU_steps","chi","D_before","D_after","grand_before","grand_after","delta_grand",
             "kinetic_before","kinetic_after","delta_kinetic","nup_before","nup_after","ndown_before","ndown_after",
             "Sz_before","Sz_after","M_over_Msat_before","M_over_Msat_after","ctm_iter_before","ctm_iter_after",
             "ctm_error_before","ctm_error_after","ctm_converged_before","ctm_converged_after")
    invalid=false
    write_table(file) do io
        row(io,"ansatz","model","seed","status",columns...)
        for c in CASES
            dir=casedir(c)
            status="invalid"
            if all(isfile(joinpath(dir,s)) for s in ("initial.jld2","final.jld2"))
                try
                    r=case_report(c)
                    row(io,c[1],c[2],c[4],r.status,r.values...)
                    println(caseid(c),"  ",r.status,"  E/site ",r.energy_before," -> ",r.energy_after,
                            "  delta=",r.energy_after-r.energy_before)
                    continue
                catch err
                    @warn "Cannot compare this result" dir exception=err
                    invalid=true
                end
            else
                status=isdir(dir) ? "incomplete" : "not_run"
            end
            row(io,c[1],c[2],c[4],status,fill("",length(columns))...)
            println(caseid(c),"  ",status)
        end
    end
    println(file)
    invalid && error("invalid case(s) present in summary")
end

function task_main(args)
    check_settings()
    println("unitcell=",join(UNIT_CELL,"x"),"  kagome_sites=",3prod(UNIT_CELL),"  results=",RUNS)
    flush(stdout)
    if args==["check"]
        include(joinpath(KAGOME_SRC,"ipeps/bootstrap.jl"))
        println("Julia ",VERSION,"  TensorKit ",Base.pkgversion(TensorKit),"  project=",Base.active_project())
        for c in CASES
            s=check_seed(c);println("OK  ",caseid(c),"  ",s.name,"  D=8  μ=",s.fields)
        end
    elseif args==["report"]
        summarize_cases()
    elseif length(args)==3 && first(args)=="run"
        run_case(findcase(args[2],args[3]))
    else
        error("usage: task.jl check | run {ipeps|ipess} {orig|mild|zero} | report")
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    task_main(ARGS)
end
