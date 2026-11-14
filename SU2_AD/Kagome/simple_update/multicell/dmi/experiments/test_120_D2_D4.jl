# CPU only; all options are here. Random and ordered-reference starts, independent D.
import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","order_probe.jl"))
options=(dims=[2,3,4],branches=["random","ordered120"],seed=20261113,J=0.,Dz=1.,h=0.,
    threads=10,trun_tol=1e-8,schedule=[(dt=.05,tau=8.),(dt=.02,tau=4.),(dt=.005,tau=1.),(dt=.001,tau=.2)],
    chi=12,check_chi=20,ctm_max=50,ctm_tol=1e-6,
    out=joinpath(@__DIR__,"..","..","results","pure_DMI_D2_D4_order_20261113"))
KagomeDMIOrderProbe.run_all(options)
