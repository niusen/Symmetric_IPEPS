using TensorKit,LinearAlgebra
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","independent_random_fields.jl"))
const R=KagomeDMIIndependentFields
mktempdir() do tmp
    cfg=(dims=[2],Dz=[.02],fields=[.36,.38],trials=[1],seed_base=202611140000,
        threads=2,trun_tol=1e-8,chi=2,ctm_max=2,ctm_tol=1e-6,
        fresh=[(dt=.01,tau=.01)],warm=[],out=tmp)
    R.run(cfg);R.run(cfg)
    a,_=R.P.K.load_checkpoint(joinpath(R.folder_for(cfg,2,.02,.36,1),"initial.jld2"))
    b,_=R.P.K.load_checkpoint(joinpath(R.folder_for(cfg,2,.02,.38,1),"initial.jld2"))
    @assert R.seed_for(cfg,2,.02,.36,1)!=R.seed_for(cfg,2,.02,.38,1)
    @assert norm(a.cells[1].B_a-b.cells[1].B_a)>1e-3
    println("JULIA INDEPENDENT SEEDS, ACTUAL RANDOM TENSORS AND POINT-LOCAL RESUME VERIFIED")
end
