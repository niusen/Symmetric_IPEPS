include("../gfpeps/bootstrap.jl")
include("../spin/bootstrap.jl")
if !isdefined(@__MODULE__,:KagomeIPEPSSU)
    include("KagomeIPEPSSU.jl")
end
