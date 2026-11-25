using TensorKit, LinearAlgebra, Random, MAT, JLD2, TOML, Dates
if !isdefined(@__MODULE__, :KAGOME_WORKSPACE)
    const KAGOME_WORKSPACE = normpath(joinpath(@__DIR__, "../../../.."))
end
if !isdefined(@__MODULE__, :KagomeFermionicZ2)
    include(joinpath(@__DIR__, "../KagomeFermionicZ2.jl"))
end
using .KagomeFermionicZ2
if !isdefined(@__MODULE__, :KagomeFermionicCTM)
    include(joinpath(@__DIR__, "../ctmrg.jl"))
end
using .KagomeFermionicCTM
if !isdefined(KagomeFermionicCTM.Legacy, :compact_double_layer)
    Base.include(KagomeFermionicCTM.Legacy, joinpath(@__DIR__, "ctm/compact_double_layer.jl"))
end
if get(ENV, "KAGOME_COMPACT_DOUBLE_LAYER", "1") == "1"
    KagomeFermionicCTM.Legacy.install_compact_double_layer!()
end
if !isdefined(@__MODULE__, :KagomeGFPEPSValidation)
    include(joinpath(@__DIR__, "import_ipess.jl"))
end
using .KagomeGFPEPSValidation
