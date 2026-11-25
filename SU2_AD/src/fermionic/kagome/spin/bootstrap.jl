if !isdefined(@__MODULE__, :KagomeFermionicZ2)
    include("../KagomeFermionicZ2.jl")
end
if !isdefined(@__MODULE__, :KagomeFermionicCTM)
    include("../ctmrg.jl")
end
if !isdefined(@__MODULE__, :KagomeGFPEPSValidation)
    include("../gfpeps/import_ipess.jl")
end
if !isdefined(@__MODULE__, :KagomeFermionicOptimization)
    include("../optimization/KagomeFermionicOptimization.jl")
end
if !isdefined(@__MODULE__, :KagomeSpinOptimization)
    include("KagomeSpinOptimization.jl")
end
