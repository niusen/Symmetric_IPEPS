"""CUDA device and storage helpers for the square-J1 Full Update."""

const SQUARE_J1_DEVICE_SPEC = Ref{String}("cpu")
const SQUARE_J1_DEVICE_STORAGE = Ref{Any}(Array)
const SQUARE_J1_CUDA_DEVICE = Ref{Any}(nothing)

function square_J1_cuda_device_id(device_spec::AbstractString)
    spec = lowercase(strip(device_spec))
    spec in ("gpu", "cuda") && return 0
    startswith(spec, "gpu:") && return parse(Int, split(spec, ":")[2])
    startswith(spec, "cuda:") && return parse(Int, split(spec, ":")[2])
    throw(ArgumentError(
        "unknown CUDA device $device_spec; use cpu, cuda, cuda:0, or cuda:1",
    ))
end

function square_J1_select_device!(device_spec::AbstractString)
    spec = lowercase(strip(device_spec))
    if spec == "cpu"
        SQUARE_J1_DEVICE_SPEC[] = "cpu"
        SQUARE_J1_DEVICE_STORAGE[] = Array
        SQUARE_J1_CUDA_DEVICE[] = nothing
        return Array
    end

    isdefined(@__MODULE__, :CUDA) || error(
        "CUDA is not loaded; install CUDA, cuTENSOR, and Adapt for GPU runs",
    )
    isdefined(@__MODULE__, :cuTENSOR) || error(
        "cuTENSOR is not loaded; TensorKit CUDA contractions require cuTENSOR",
    )
    isdefined(@__MODULE__, :Adapt) || error("Adapt is not loaded")
    CUDA.functional() || error("CUDA.functional() is false")
    cuTENSOR.functional() || error("cuTENSOR.functional() is false")

    device_id = square_J1_cuda_device_id(spec)
    canonical_spec = "cuda:$device_id"
    if SQUARE_J1_DEVICE_SPEC[] == canonical_spec &&
       !isnothing(SQUARE_J1_CUDA_DEVICE[])
        square_J1_use_selected_device!()
        return SQUARE_J1_DEVICE_STORAGE[]
    end
    devices = collect(CUDA.devices())
    0 <= device_id < length(devices) || error(
        "requested cuda:$device_id, but visible devices are cuda:0 through " *
        "cuda:$(length(devices) - 1)",
    )
    selected_device = devices[device_id + 1]
    CUDA.device!(selected_device)
    CUDA.allowscalar(false)
    CUDA.synchronize()
    SQUARE_J1_DEVICE_SPEC[] = canonical_spec
    SQUARE_J1_DEVICE_STORAGE[] = CUDA.CuArray
    SQUARE_J1_CUDA_DEVICE[] = selected_device
    println("square-J1 tensor device = cuda:$device_id, $(CUDA.name(CUDA.device()))")
    flush(stdout)
    return CUDA.CuArray
end

function square_J1_use_selected_device!()
    SQUARE_J1_DEVICE_SPEC[] == "cpu" && return nothing
    isnothing(SQUARE_J1_CUDA_DEVICE[]) && error("no CUDA device has been selected")
    CUDA.device!(SQUARE_J1_CUDA_DEVICE[])
    return nothing
end

function square_J1_to_storage(storage, tensor::TensorKit.AbstractTensorMap)
    if storage === Array && TensorKit.storagetype(tensor) <: Array
        return tensor
    end
    isdefined(@__MODULE__, :Adapt) || error("Adapt is required to move TensorMap storage")
    return Adapt.adapt(storage, tensor)
end

function square_J1_storage_family(storage::Type)
    storage <: Array && return Array
    if isdefined(@__MODULE__, :CUDA) && storage <: CUDA.CuArray
        return CUDA.CuArray
    end
    return storage
end

function square_J1_to_scalartype_like(
    tensor::TensorKit.AbstractTensorMap,
    reference::TensorKit.AbstractTensorMap,
)
    target_type = TensorKit.scalartype(reference)
    source_type = TensorKit.scalartype(tensor)
    source_type === target_type && return tensor
    if target_type <: Real && source_type <: Complex
        real_type = typeof(real(zero(source_type)))
        imag_norm = norm(imag(tensor))
        scale = max(norm(tensor), one(real_type))
        imag_norm <= 100 * eps(real_type) * scale || throw(ArgumentError(
            "cannot convert a genuinely complex TensorMap to $target_type: " *
            "relative imaginary norm=$(imag_norm / scale)",
        ))
        return real(tensor)
    elseif target_type <: Complex && source_type <: Real
        return complex(tensor)
    end
    converted = similar(tensor, target_type)
    copy!(converted, tensor)
    return converted
end

"""Match both scalar type and storage family to `reference`, as in triangular FU."""
function square_J1_to_storage_like(
    tensor::TensorKit.AbstractTensorMap,
    reference::TensorKit.AbstractTensorMap,
)
    tensor = square_J1_to_scalartype_like(tensor, reference)
    storage = square_J1_storage_family(TensorKit.storagetype(reference))
    return square_J1_to_storage(storage, tensor)
end

square_J1_to_storage(storage, value::Number) = value
square_J1_to_storage(storage, value::AbstractString) = value
square_J1_to_storage(storage, value::Symbol) = value
square_J1_to_storage(storage, value::Nothing) = nothing
square_J1_to_storage(storage, value::Missing) = value

square_J1_to_storage(storage, values::Tuple) =
    map(value -> square_J1_to_storage(storage, value), values)

function square_J1_to_storage(storage, values::NamedTuple{names}) where {names}
    converted = map(value -> square_J1_to_storage(storage, value), Tuple(values))
    return NamedTuple{names}(converted)
end

function square_J1_to_storage(storage, values::AbstractArray)
    if storage === Array && !(eltype(values) <: TensorKit.AbstractTensorMap) &&
       eltype(values) !== Any
        return Array(values)
    end
    converted = Array{Any}(undef, size(values))
    for index in CartesianIndices(values)
        isassigned(values, index) &&
            (converted[index] = square_J1_to_storage(storage, values[index]))
    end
    return converted
end

function square_J1_to_storage(storage, value::Cset_struc)
    return Cset_struc(
        square_J1_to_storage(storage, value.C1),
        square_J1_to_storage(storage, value.C2),
        square_J1_to_storage(storage, value.C3),
        square_J1_to_storage(storage, value.C4),
    )
end

function square_J1_to_storage(storage, value::Tset_struc)
    return Tset_struc(
        square_J1_to_storage(storage, value.T1),
        square_J1_to_storage(storage, value.T2),
        square_J1_to_storage(storage, value.T3),
        square_J1_to_storage(storage, value.T4),
    )
end

square_J1_to_storage(storage, value::CTM_struc) = CTM_struc(
    square_J1_to_storage(storage, value.Cset),
    square_J1_to_storage(storage, value.Tset),
)

function square_J1_to_device(device_spec::AbstractString, value)
    storage = square_J1_select_device!(device_spec)
    return square_J1_to_storage(storage, value)
end

square_J1_to_cpu(value) = square_J1_to_storage(Array, value)

function square_J1_reclaim_device_memory!(; aggressive::Bool=false)
    SQUARE_J1_DEVICE_SPEC[] == "cpu" && return GC.gc(aggressive)
    square_J1_use_selected_device!()
    CUDA.synchronize()
    GC.gc(aggressive)
    CUDA.reclaim()
    return nothing
end

function square_J1_print_device_memory(label::AbstractString)
    SQUARE_J1_DEVICE_SPEC[] == "cpu" && return nothing
    square_J1_use_selected_device!()
    println(label)
    if isdefined(CUDA, :memory_status)
        CUDA.memory_status()
    else
        CUDA.pool_status()
    end
    flush(stdout)
    return nothing
end
