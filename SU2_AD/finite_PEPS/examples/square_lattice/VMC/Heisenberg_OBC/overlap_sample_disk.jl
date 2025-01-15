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

include("../../../../environment/MC/contract_disk.jl")
include("../../../../environment/MC/sampling.jl")


####################
import LinearAlgebra.BLAS as BLAS
using Base.Threads

n_cpu=1;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);

Base.Sys.set_process_title("C"*string(n_cpu)*"_fPEPS")
pid=getpid();
println("pid="*string(pid));;flush(stdout);
####################

"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

D=2;
Lx=6;
Ly=6;

filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
data=load(filenm*".jld2");
psi0=data["psi"];
psi=Matrix{TensorMap}(undef,Lx,Ly);
psi[:]=psi0[:];
Vp=space(psi[2,2],5);


normalize_PEPS!(psi,Vp,contract_whole_disk);#normalize psi such that the amplitude of a single config is close to 1


config=initial_Neel_config(Lx,Ly);


psi_sample=apply_sampling_projector(psi,Lx,Ly,config,Vp);



chi=10;

@btime Norm,trun_err=contract_whole_disk(psi_sample,chi);
Norm,trun_err=contract_whole_disk(psi_sample,chi);
@show Norm,trun_err


# Norm_exact=exact_contraction(psi_sample);
# 

@btime Norm,trun_err=contract_sample(psi,Lx,Ly,config,Vp,contract_whole_disk)

