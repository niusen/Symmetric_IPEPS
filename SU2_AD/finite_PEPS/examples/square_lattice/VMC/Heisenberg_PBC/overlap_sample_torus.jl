using LinearAlgebra:I,diagm,diag
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using BenchmarkTools
cd(@__DIR__)

include("../../../../state/iPEPS_ansatz.jl")
include("../../../../setting/Settings.jl")
include("../../../../setting/linearalgebra.jl")
include("../../../../setting/tuple_methods.jl")
include("../../../../environment/MC/contract_torus.jl")
include("../../../../environment/MC/build_degenerate_states.jl")
include("../../../../environment/MC/sampling.jl")
include("../../../../environment/MC/sampling_eliminate_physical_leg.jl")







####################
import LinearAlgebra.BLAS as BLAS
using Base.Threads

n_cpu=1;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);

Base.Sys.set_process_title("C"*string(n_cpu)*"_MPS")
pid=getpid();
println("pid="*string(pid));;flush(stdout);
####################



"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

D=5;
Lx=12;
Ly=12;

chi=5;

##############################################
filenm="Heisenber_SU_D_"*string(Lx)*"x"*string(Ly)*"_D"*string(D)*".jld2"
data=load(filenm);
psi0=data["T_set"];
# A=TensorMap(A.data,A.codom,A.dom);




#psi=generate_obc_from_iPEPS(A,Lx,Ly);
psi=Matrix{TensorMap}(undef,Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        psi[cx,cy]=psi0[cx,cy];
    end
end

Vp=space(psi[2,2],5);



global projector_method
projector_method="1";#"1" or "2"
normalize_PEPS!(psi,Vp,contract_whole_torus);#normalize psi such that the amplitude of a single config is close to 1


#initial spin config, total sz=0
config=initial_Neel_config(Lx,Ly,1);

psi_decomposed=decompose_physical_legs(psi,Vp);
psi_sample=pick_sample(psi_decomposed,config);




#do contraction

@btime Norm,trun_err=contract_whole_torus(psi_sample,chi);
Norm,trun_err=contract_whole_torus(psi_sample,chi);
@show [Norm,sum(abs.(trun_err))]

@btime Norm,trun_err=contract_whole_torus_boundaryMPS(psi_sample,chi);
Norm,trun_err=contract_whole_torus_boundaryMPS(psi_sample,chi);
@show [Norm,sum(abs.(trun_err))]
# ##############################

# @btime Norm,trun_err=contract_sample(psi,Lx,Ly,config,Vp,contract_whole_torus);

@btime Norm,trun_err=contract_sample(psi_decomposed,Lx,Ly,config,Vp,contract_whole_torus);




