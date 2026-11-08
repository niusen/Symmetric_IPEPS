"""Lightweight batch preflight; no TensorKit loading or numerical calculation."""
module KagomeVariancePlan
using SHA
export variance_jobs, case_settings, file_sha256

file_sha256(path)=open(io->bytes2hex(sha256(io)),path)

function variance_jobs(cfg;check_inputs=true)
    for (label,values) in (("dims",cfg.dims),("chis",cfg.chis),("sizes",cfg.sizes))
        !isempty(values) && all(>(0),values) && length(unique(values))==length(values) ||
            error("$label must contain distinct positive integers")
    end
    issorted(cfg.sizes) || error("L values must be increasing")
    cfg.threads>0 && cfg.sweeps>0 && 2<=cfg.ctm_iterations<=50 && cfg.ctm_tolerance>0 && cfg.source_shift>=1 || error("invalid measurement settings")
    all(L->L%3==0,cfg.sizes) || error("Reported L values must be multiples of 3")
    cfg.contraction_size%3==0 && cfg.contraction_size>=maximum(cfg.sizes)+6 ||
        error("Contraction size must be a multiple of 3, at least max(reported L)+6 for patch/boundary padding")
    jobs=NamedTuple[]
    for D in cfg.dims
        haskey(cfg.checkpoints,D) || error("No checkpoint configured for D=$D")
        file=abspath(cfg.checkpoints[D])
        check_inputs && !isfile(file) && error("Missing saved D=$D state: $file")
        for chi in cfg.chis
            push!(jobs,(D=Int(D),chi=Int(chi),L=Int(cfg.contraction_size),checkpoint=file,
                out=abspath(joinpath(cfg.out,"D$D","chi$chi"))))
        end
    end
    jobs
end

function case_settings(cfg,job,source_hash,code_hash)
    Dict("format"=>"kagome_variance_batch_case_v1","D"=>job.D,"chi"=>job.chi,"L"=>job.L,
        "source_checkpoint"=>job.checkpoint,"source_sha256"=>source_hash,"code_sha256"=>code_hash,
        "window_sizes"=>collect(cfg.sizes),"threads"=>cfg.threads,"sweeps"=>cfg.sweeps,
        "ctm_iterations"=>cfg.ctm_iterations,"ctm_tolerance"=>cfg.ctm_tolerance,
        "source_shift"=>cfg.source_shift,"device"=>"cpu")
end
end
