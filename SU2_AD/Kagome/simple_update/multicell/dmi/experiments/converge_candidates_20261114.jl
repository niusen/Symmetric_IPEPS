# Continue independent dense random-origin candidates; no U1 restrictions.
import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","plateau_explore.jl"))
const P=KagomeDMIPlateauExplore
P.LinearAlgebra.BLAS.set_num_threads(10);P.N.prepare_device!("cpu")
parent=joinpath(@__DIR__,"..","..","results")
cfg=(seed=20261114,fresh=[(dt=.003,tau=30.),(dt=.001,tau=3.),(dt=.0005,tau=1.)],
    warm=[(dt=.003,tau=30.),(dt=.001,tau=3.),(dt=.0005,tau=1.)],trun_tol=1e-8,
    chi=20,ctm_max=50,ctm_tol=1e-6,out=joinpath(parent,"DMI_plateau_convergence_checks_20261114"))
for (D,trial) in ((4,1),(2,2),(4,2))
    source=joinpath(parent,"DMI_plateau_fresh_centers_20261114","D$(D)_Dz0.02_fresh$(trial)","h0.38","final.jld2")
    final,r=P.point(cfg,D,.02,.38,"refined_fresh$(trial)",source,false)
    if !r["ctm"]["converged"]
        s,_=P.K.load_checkpoint(final)
        more=P.K.measure(s;J=1.,Dz=.02,h=-.38,chi=24,iterations=50,tolerance=1e-6,verbose=false)
        merge!(more,Dict("D"=>D,"Dz"=>.02,"h_physical"=>.38,"branch"=>"refined_fresh$(trial)","chi"=>24,
            "m_normalized"=>2more["mean_Sz"],"valid_measurement"=>more["observables_hermiticity_ok"],"source"=>source))
        P.N.write_json(joinpath(dirname(final),"observables_chi24.json"),more)
    end
    P.collect_results(cfg.out)
end
write(joinpath(cfg.out,"COMPLETED.txt"),"Dense candidate continuation and chi20/24 checks finished.\n")
