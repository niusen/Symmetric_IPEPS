using TOML, SHA

"""Read and preflight a GFPEPS job without loading TensorKit or doing a contraction.

All relative paths are relative to the TOML file. Inputs and outputs are separate.
"""
function read_gfpeps_job(path)
    path=abspath(path); config=TOML.parsefile(path); base=dirname(path)
    job=config["job"]; ctm=config["ctm"]
    mode=get(job,"mode","heisenberg")
    mode in ("heisenberg","gaussian") || error("mode must be heisenberg or gaussian")
    family=job["family"]; family in ("ipess","ipeps") || error("family must be ipess or ipeps")
    D=job["D"]; D in (4,8) || error("The exported fixtures use D=4 or D=8")
    input=normpath(joinpath(base,job["input"]))
    output=normpath(joinpath(base,job["output"]))
    input==output && error("Use separate input and output directories")
    required=mode=="gaussian" ? ("tensors.mat","reference_N192.mat") : ("tensors.mat",)
    for file in required
        isfile(joinpath(input,file)) || error("Missing input: $(joinpath(input,file))")
    end
    chis=Int.(ctm["chi"])
    !isempty(chis) && all(>(0),chis) && issorted(chis) && allunique(chis) || error("chi must be a nonempty increasing list of positive integers")
    get(ctm,"maxiter",100)>0 || error("maxiter must be positive")
    get(ctm,"tol",1e-7)>0 || error("tol must be positive")
    get(ctm,"cutoff",1e-8)>0 || error("cutoff must be positive")
    get(ctm,"gauge_sweeps",0)>=0 || error("gauge_sweeps must be nonnegative")
    get(job,"blas_threads",1)>0 || error("blas_threads must be positive")
    get(ctm,"update_method","continuous_update") in ("continuous_update","together_update") || error("Unknown CTMRG update_method")
    # Optional explicit restart; otherwise each chi starts from PBC. This avoids
    # silently treating a stale or differently gauged environment as compatible.
    warmstart=get(job,"warmstart",nothing)
    if !isnothing(warmstart)
        warmstart=normpath(joinpath(base,warmstart))
        isfile(warmstart) || error("Missing warmstart: $warmstart")
    end
    tensor_sha256=bytes2hex(open(sha256,joinpath(input,"tensors.mat")))
    (;path,config,job,ctm,mode,family,D,input,output,chis,warmstart,tensor_sha256)
end

function check_gfpeps_job(path)
    plan=read_gfpeps_job(path)
    println("mode=$(plan.mode) family=$(plan.family) D=$(plan.D) chi=$(plan.chis)")
    println("input=$(plan.input)\noutput=$(plan.output)")
    println("tensors.mat sha256=$(plan.tensor_sha256)")
    println("Configuration and input files OK; no contraction performed.")
    plan
end
