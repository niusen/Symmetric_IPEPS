using Test, TensorKit, LinearAlgebra, Random
include(joinpath(@__DIR__,"..","..","..","..","src","fermionic","kagome","KagomeFermionicZ2.jl"))
using .KagomeFermionicZ2
const KF = KagomeFermionicZ2
BLAS.set_num_threads(1)

include("fixtures.jl")

include("test_core.jl")

if get(ENV,"KAGOME_TEST_CTM","0") == "1"
    include("test_ctmrg.jl")
end
