# Independent unrestricted dense center-field checks; no continuation seed.
import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","plateau_explore.jl"))
const P=KagomeDMIPlateauExplore
P.LinearAlgebra.BLAS.set_num_threads(10);P.N.prepare_device!("cpu")
base=(seed=20261114,fresh=[(dt=.03,tau=40.),(dt=.01,tau=20.),(dt=.003,tau=5.),(dt=.001,tau=2.),(dt=.0005,tau=1.)],
    warm=[(dt=.001,tau=1.)],trun_tol=1e-8,chi=16,ctm_max=50,ctm_tol=1e-7,
    out=joinpath(@__DIR__,"..","..","results","DMI_plateau_fresh_centers_20261114"))
for (D,Dz) in ((2,.02),(3,.02),(4,.02),(4,.05)),trial in 1:2
    cfg=merge(base,(seed=base.seed+100000trial,))
    P.point(cfg,D,Dz,.38,"fresh$(trial)",nothing,true)
    P.collect_results(cfg.out)
end
write(joinpath(base.out,"COMPLETED.txt"),"Eight independent random center-field checks complete.\n")
