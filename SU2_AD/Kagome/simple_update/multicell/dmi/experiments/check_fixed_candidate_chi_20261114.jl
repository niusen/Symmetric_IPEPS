# Fixed-ket CTM checks: both environment runs must converge before comparing.
import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","KagomeDMI.jl"))
const K=KagomeDMI
K.LinearAlgebra.BLAS.set_num_threads(10);K.N.prepare_device!("cpu")
root=joinpath(@__DIR__,"..","..","results","DMI_plateau_convergence_checks_20261114")
for (D,trial,chi) in ((4,1,28),(2,2,12))
    folder=joinpath(root,"D$(D)_Dz0.02_refined_fresh$(trial)","h0.38")
    s,_=K.load_checkpoint(joinpath(folder,"final.jld2"))
    r=K.measure(s;J=1.,Dz=.02,h=-.38,chi,iterations=50,tolerance=1e-6,verbose=true)
    merge!(r,Dict("D"=>D,"Dz"=>.02,"h_physical"=>.38,"branch"=>"refined_fresh$(trial)","chi"=>chi,
        "m_normalized"=>2r["mean_Sz"],"valid_measurement"=>r["observables_hermiticity_ok"]))
    K.N.write_json(joinpath(folder,"observables_chi$(chi).json"),r)
    println("FIXED KET CHI CHECK D=$D chi=$chi E=$(r["energy_per_spin"]) m=$(r["m_normalized"]) CTM=$(r["ctm"])");flush(stdout)
end
write(joinpath(root,"FIXED_CHI_CHECKS_COMPLETED.txt"),"Fixed-ket chi checks finished; inspect individual convergence flags.\n")
