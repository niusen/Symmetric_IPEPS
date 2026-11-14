import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","plateau_explore.jl"))
options=(dims=[2,3,4],Dz=[.02,.05,.1],fields=[.28,.32,.36,.38,.40,.44,.48],seed=20261114,
    fresh=[(dt=.03,tau=40.),(dt=.01,tau=20.),(dt=.003,tau=5.),(dt=.001,tau=2.),(dt=.0005,tau=1.)],
    warm=[(dt=.03,tau=10.),(dt=.01,tau=5.),(dt=.003,tau=1.),(dt=.001,tau=.5),(dt=.0005,tau=.2)],
    threads=10,trun_tol=1e-8,chi=12,ctm_max=50,ctm_tol=1e-6,
    out=joinpath(@__DIR__,"..","..","results","DMI_plateau_dense_D2_D4_20261113"))
KagomeDMIPlateauExplore.run(options)
