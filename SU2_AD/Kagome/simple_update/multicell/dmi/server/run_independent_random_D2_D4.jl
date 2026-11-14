import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","independent_random_fields.jl"))
options=(dims=[2,3,4],Dz=[.02,.05,.1],fields=[.28,.32,.36,.38,.40,.44,.48],trials=[1,2,3],
    seed_base=202611140000,threads=10,trun_tol=1e-8,chi=12,ctm_max=50,ctm_tol=1e-6,
    fresh=[(dt=.03,tau=40.),(dt=.01,tau=20.),(dt=.003,tau=5.),(dt=.001,tau=2.),(dt=.0005,tau=1.)],warm=[],
    out=joinpath(@__DIR__,"..","..","results","DMI_independent_random3_D2_D4_20261114"))
KagomeDMIIndependentFields.run(options)
