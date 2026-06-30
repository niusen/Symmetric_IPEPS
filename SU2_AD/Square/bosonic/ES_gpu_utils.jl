if !isdefined(@__MODULE__, :run_device)
    @show run_device = "cpu"
end

USE_GPU_ES = lowercase(strip(run_device)) != "cpu"

if USE_GPU_ES
    using CUDA, cuTENSOR, Adapt
end

if !isdefined(@__MODULE__, :ROOT_DIR)
    const ROOT_DIR = normpath(joinpath(@__DIR__, "..", ".."))
end

if !isdefined(@__MODULE__, :ipeps_select_device!)
    include(joinpath(ROOT_DIR, "src", "device_utils.jl"))
end

ipeps_select_device!(run_device)

to_es_device(x) = USE_GPU_ES ? ipeps_to_device(x) : x
to_es_cpu(x) = USE_GPU_ES ? ipeps_to_cpu(x) : x

function es_synchronize()
    USE_GPU_ES && CUDA.synchronize()
    return nothing
end
