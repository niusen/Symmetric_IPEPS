const IPESS_DEVICE_SPEC = Ref{String}("cpu")
const IPESS_STORAGE = Ref{Any}(Array)
const IPESS_CUDA_DEVICE = Ref{Any}(nothing)

const IPESS_CTM_DEVICE = Ref{String}("cpu")
const IPESS_FULL_UPDATE_DEVICE = Ref{String}("cpu")
const IPESS_OBSERVABLE_DEVICE = Ref{String}("cpu")
const IPESS_CONTRACT_TRIANGLE_ENV_DEVICE = Ref{String}("full_update")
const IPESS_ENV_GAUGE_SVD_DEVICE = Ref{String}("full_update")
const IPESS_ENV_GAUGE_SVD_DEBUG_BLOCKS = Ref{Bool}(false)
const IPESS_CONTRACT_TRIANGLE_ENV_PROJECTOR = Ref{Bool}(false)
const IPESS_MEMORY_INFO = Ref{Bool}(false)

function _ipeps_modcall(mod::Module, fname::Symbol, args...; kwargs...)
    return getfield(mod, fname)(args...; kwargs...)
end

function _ipeps_require_module!(name::Symbol)
    isdefined(@__MODULE__, name) && return getfield(@__MODULE__, name)
    error(
        "Module $name is not loaded. For GPU runs, load CUDA support before selecting a GPU: " *
        "`using CUDA, cuTENSOR, Adapt`."
    )
end

function _ipeps_require_adapt!()
    return _ipeps_require_module!(:Adapt)
end

function _ipeps_require_cuda!()
    _ipeps_require_adapt!()
    cuda_mod = _ipeps_require_module!(:CUDA)
    cutensor_mod = _ipeps_require_module!(:cuTENSOR)
    _ipeps_modcall(cuda_mod, :functional) || error("CUDA.functional() is false; no working CUDA device was found.")
    _ipeps_modcall(cutensor_mod, :functional) ||
        error("cuTENSOR.functional() is false; TensorKit CUDA contractions need cuTENSOR.")
    return cuda_mod
end

function _ipeps_cuda_id(device_spec::AbstractString)
    spec = lowercase(strip(device_spec))
    spec == "gpu" && return 0
    spec == "cuda" && return 0
    startswith(spec, "gpu:") && return parse(Int, split(spec, ":")[2])
    startswith(spec, "cuda:") && return parse(Int, split(spec, ":")[2])
    error("Unknown CUDA device spec: $device_spec. Use \"cpu\", \"cuda\", \"cuda:0\", or \"cuda:1\".")
end

function ipeps_select_device!(device_spec::AbstractString)
    spec = lowercase(strip(device_spec))
    if spec == "cpu"
        IPESS_DEVICE_SPEC[] == "cpu" && return IPESS_STORAGE[]
        IPESS_DEVICE_SPEC[] = "cpu"
        IPESS_STORAGE[] = Array
        IPESS_CUDA_DEVICE[] = nothing
        println("iPEPS tensor device = CPU")
        return IPESS_STORAGE[]
    end

    if spec == "gpu" || spec == "cuda" || startswith(spec, "gpu:") || startswith(spec, "cuda:")
        cuda_mod = _ipeps_require_cuda!()
        device_id = _ipeps_cuda_id(spec)
        if IPESS_DEVICE_SPEC[] == "cuda:$device_id"
            ipeps_use_selected_device!()
            return IPESS_STORAGE[]
        end
        devices = collect(getfield(cuda_mod, :devices)())
        0 <= device_id < length(devices) ||
            error("Requested cuda:$device_id, but only cuda:0 through cuda:$(length(devices) - 1) are visible.")

        selected_device = devices[device_id + 1]
        _ipeps_modcall(cuda_mod, :device!, selected_device)
        _ipeps_modcall(cuda_mod, :allowscalar, false)
        _ipeps_modcall(cuda_mod, :synchronize)

        IPESS_DEVICE_SPEC[] = "cuda:$device_id"
        IPESS_STORAGE[] = getfield(cuda_mod, :CuArray)
        IPESS_CUDA_DEVICE[] = selected_device
        println("iPEPS tensor device = cuda:$device_id, ", _ipeps_modcall(cuda_mod, :name, _ipeps_modcall(cuda_mod, :device)))
        return IPESS_STORAGE[]
    end

    error("Unknown run_device=\"$device_spec\". Use \"cpu\", \"cuda\", \"cuda:0\", or \"cuda:1\".")
end

function ipeps_use_selected_device!()
    IPESS_DEVICE_SPEC[] == "cpu" && return nothing
    cuda_mod = _ipeps_require_cuda!()
    isnothing(IPESS_CUDA_DEVICE[]) && error("No CUDA device has been selected. Call ipeps_select_device!(\"cuda:i\") first.")
    _ipeps_modcall(cuda_mod, :device!, IPESS_CUDA_DEVICE[])
    return nothing
end

function ipeps_set_step_devices!(; ctm=IPESS_DEVICE_SPEC[], full_update=IPESS_DEVICE_SPEC[],
    observable=IPESS_DEVICE_SPEC[], contract_triangle_env="full_update", env_gauge_svd="full_update")
    IPESS_CTM_DEVICE[] = lowercase(strip(String(ctm)))
    IPESS_FULL_UPDATE_DEVICE[] = lowercase(strip(String(full_update)))
    IPESS_OBSERVABLE_DEVICE[] = lowercase(strip(String(observable)))
    IPESS_CONTRACT_TRIANGLE_ENV_DEVICE[] = lowercase(strip(String(contract_triangle_env)))
    IPESS_ENV_GAUGE_SVD_DEVICE[] = lowercase(strip(String(env_gauge_svd)))
    println("iPEPS step devices: CTM=", IPESS_CTM_DEVICE[], ", full_update=", IPESS_FULL_UPDATE_DEVICE[],
        ", observable=", IPESS_OBSERVABLE_DEVICE[],
        ", contract_triangle_env=", IPESS_CONTRACT_TRIANGLE_ENV_DEVICE[],
        ", env_gauge_svd=", IPESS_ENV_GAUGE_SVD_DEVICE[])
    return nothing
end

function ipeps_set_memory_info!(flag::Bool)
    IPESS_MEMORY_INFO[] = flag
    println("iPEPS memory info = ", flag)
    return nothing
end

function ipeps_set_contract_triangle_env_projector!(flag::Bool)
    IPESS_CONTRACT_TRIANGLE_ENV_PROJECTOR[] = flag
    println("iPEPS contract_triangle_env projector = ", flag)
    return nothing
end

function ipeps_set_env_gauge_svd_debug_blocks!(flag::Bool)
    IPESS_ENV_GAUGE_SVD_DEBUG_BLOCKS[] = flag
    println("iPEPS env_gauge_svd block debug = ", flag)
    return nothing
end

function _ipeps_format_bytes(bytes::Real)
    bytes < 1024 && return string(bytes, " B")
    units = ("KiB", "MiB", "GiB", "TiB")
    value = Float64(bytes)
    unit = units[1]
    for u in units
        value /= 1024
        unit = u
        abs(value) < 1024 && break
    end
    return string(round(value; digits=3), " ", unit)
end

function ipeps_tensor_data_bytes(t::TensorKit.AbstractTensorMap)
    data = getproperty(t, :data)
    return sizeof(data)
end

function ipeps_print_tensor_memory(label::AbstractString, t::TensorKit.AbstractTensorMap)
    bytes = ipeps_tensor_data_bytes(t)
    println(label, " actual tensor data = ", _ipeps_format_bytes(bytes),
        " (", bytes, " bytes), storage = ", TensorKit.storagetype(t))
    return bytes
end

function ipeps_print_device_memory(label::AbstractString)
    IPESS_DEVICE_SPEC[] == "cpu" && return nothing
    if isdefined(@__MODULE__, :CUDA)
        cuda_mod = getfield(@__MODULE__, :CUDA)
        println(label)
        if isdefined(cuda_mod, :memory_status)
            getfield(cuda_mod, :memory_status)()
        elseif isdefined(cuda_mod, :pool_status)
            getfield(cuda_mod, :pool_status)()
        elseif isdefined(cuda_mod, :available_memory) && isdefined(cuda_mod, :total_memory)
            free_bytes = getfield(cuda_mod, :available_memory)()
            total_bytes = getfield(cuda_mod, :total_memory)()
            used_bytes = total_bytes - free_bytes
            println("CUDA memory used = ", _ipeps_format_bytes(used_bytes),
                " / ", _ipeps_format_bytes(total_bytes),
                ", free = ", _ipeps_format_bytes(free_bytes))
        else
            println("CUDA memory status is unavailable in this CUDA.jl version.")
        end
    end
    return nothing
end

function ipeps_to_storage(storage, t::TensorKit.AbstractTensorMap)
    if storage === Array && TensorKit.storagetype(t) <: Array
        return t
    end
    adapt_mod = _ipeps_require_adapt!()
    return _ipeps_modcall(adapt_mod, :adapt, storage, t)
end

function _ipeps_storage_family(storage::Type)
    storage <: Array && return Array
    if isdefined(@__MODULE__, :CUDA)
        cuda_mod = getfield(@__MODULE__, :CUDA)
        cuarray = getfield(cuda_mod, :CuArray)
        storage <: cuarray && return cuarray
    end
    return storage
end

function _ipeps_to_scalartype_like(t::TensorKit.AbstractTensorMap, ref::TensorKit.AbstractTensorMap)
    T = TensorKit.scalartype(ref)
    TensorKit.scalartype(t) === T && return t
    y = similar(t, T)
    copy!(y, t)
    return y
end

function ipeps_to_storage_like(x, ref::TensorKit.AbstractTensorMap)
    y = x isa TensorKit.AbstractTensorMap ? _ipeps_to_scalartype_like(x, ref) : x
    return ipeps_to_storage(_ipeps_storage_family(TensorKit.storagetype(ref)), y)
end

ipeps_to_storage(storage, x::Number) = x
ipeps_to_storage(storage, x::AbstractString) = x
ipeps_to_storage(storage, x::Symbol) = x
ipeps_to_storage(storage, x::Nothing) = nothing

function ipeps_to_storage(storage, x::Tuple)
    return map(y -> ipeps_to_storage(storage, y), x)
end

function ipeps_to_storage(storage, x::NamedTuple{names}) where {names}
    return NamedTuple{names}(map(y -> ipeps_to_storage(storage, y), Tuple(x)))
end

function _ipeps_cpu_tensor_array(xs::AbstractArray)
    for ind in CartesianIndices(xs)
        if isassigned(xs, ind)
            xs[ind] isa TensorKit.AbstractTensorMap || return false
            TensorKit.storagetype(xs[ind]) <: Array || return false
        end
    end
    return true
end

function ipeps_to_storage(storage, xs::AbstractArray)
    if storage === Array
        if !(eltype(xs) <: TensorKit.AbstractTensorMap) && eltype(xs) !== Any
            return Array(xs)
        elseif eltype(xs) <: TensorKit.AbstractTensorMap && _ipeps_cpu_tensor_array(xs)
            return xs
        end
    end
    ys = Array{Any}(undef, size(xs))
    for ind in CartesianIndices(xs)
        isassigned(xs, ind) && (ys[ind] = ipeps_to_storage(storage, xs[ind]))
    end
    return ys
end

function ipeps_to_storage(storage, x::Cset_struc)
    return Cset_struc(
        ipeps_to_storage(storage, x.C1),
        ipeps_to_storage(storage, x.C2),
        ipeps_to_storage(storage, x.C3),
        ipeps_to_storage(storage, x.C4),
    )
end

function ipeps_to_storage(storage, x::Tset_struc)
    return Tset_struc(
        ipeps_to_storage(storage, x.T1),
        ipeps_to_storage(storage, x.T2),
        ipeps_to_storage(storage, x.T3),
        ipeps_to_storage(storage, x.T4),
    )
end

function ipeps_to_storage(storage, x::CTM_struc)
    return CTM_struc(ipeps_to_storage(storage, x.Cset), ipeps_to_storage(storage, x.Tset))
end

function ipeps_to_device(device_spec::AbstractString, x)
    ipeps_select_device!(device_spec)
    return ipeps_to_storage(IPESS_STORAGE[], x)
end

ipeps_to_device(x) = ipeps_to_storage(IPESS_STORAGE[], x)
ipeps_to_cpu(x) = ipeps_to_storage(Array, x)

function ipeps_reclaim_device_memory!(; aggressive::Bool=false)
    IPESS_DEVICE_SPEC[] == "cpu" && return nothing
    if isdefined(@__MODULE__, :CUDA)
        cuda_mod = getfield(@__MODULE__, :CUDA)
        isdefined(cuda_mod, :synchronize) && getfield(cuda_mod, :synchronize)()
        GC.gc(aggressive)
        isdefined(cuda_mod, :reclaim) && getfield(cuda_mod, :reclaim)()
    else
        GC.gc(aggressive)
    end
    return nothing
end

function ipeps_print_storage(label::AbstractString, t::TensorMap)
    println(label, " storage = ", TensorKit.storagetype(t))
    return nothing
end

function ipeps_print_storage(label::AbstractString, xs::AbstractArray)
    for ind in CartesianIndices(xs)
        if isassigned(xs, ind)
            ipeps_print_storage("$label[$ind]", xs[ind])
            return nothing
        end
    end
    println(label, " storage = <no assigned tensors>")
    return nothing
end

function ipeps_print_storage(label::AbstractString, x::Tuple)
    isempty(x) && return println(label, " storage = <empty tuple>")
    ipeps_print_storage(label * "[1]", first(x))
    return nothing
end
