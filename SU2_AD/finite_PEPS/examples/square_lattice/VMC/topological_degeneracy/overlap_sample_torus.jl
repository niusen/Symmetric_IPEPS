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

include("../../../../environment/MC/contract_torus.jl")
include("../../../../environment/MC/build_degenerate_states.jl")
include("../../../../environment/MC/sampling.jl")
include("../../../../environment/MC/sampling_eliminate_physical_leg.jl")

include("../../../../setting/Settings.jl")
include("../../../../setting/linearalgebra.jl")
include("../../../../setting/tuple_methods.jl")





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

Lx=6;
Ly=6;

chi=10;

##############################################

data=load("CSL_D3_U1.jld2");
A=data["A"];
# A=TensorMap(A.data,A.codom,A.dom);


Vv=U₁Space(0=>1,1/2=>1,-1/2=>1);
Vp=U₁Space(1/2=>1,-1/2=>1);
# Vv=ℤ₂Space(0=>1,1=>2);
# Vp=ℤ₂Space(1=>2);
A=TensorMap(convert(Array,A),Vv*Vv,Vv*Vv*Vp);
A=permute(A,(1,2,3,4,5,));


#psi=generate_obc_from_iPEPS(A,Lx,Ly);
psi=Matrix{TensorMap}(undef,Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        psi[cx,cy]=A;
    end
end

psi_00,psi_0pi,psi_pi0,psi_pipi =construct_4_states(psi,Vv);#four states
#################################
global projector_method
projector_method="1";#"1" or "2"
normalize_PEPS!(psi,Vp,contract_whole_torus);#normalize psi such that the amplitude of a single config is close to 1


#initial spin config, total sz=0
config=initial_Neel_config(Lx,Ly);



#apply projector to obtain sample
psi_sample=apply_sampling_projector(psi,Lx,Ly,config,Vp);
#remove physical leg when U1 symmetry is used
psi_sample=shift_pleg(psi_sample);





#do contraction

@btime Norm,trun_err=contract_whole_torus(psi_sample,chi);
Norm,trun_err=contract_whole_torus(psi_sample,chi);
@show [Norm,sum(abs.(trun_err))]
# ##############################

