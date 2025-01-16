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

D=3;
Lx=6;
Ly=6;

filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
psi,Vp=load_fPEPS(Lx,Ly,filenm);


normalize_PEPS!(psi,Vp,contract_whole_disk);#normalize psi such that the amplitude of a single config is close to 1


config=initial_Neel_config(Lx,Ly,1);

psi_decomposed=decompose_physical_legs(psi,Vp);
psi_sample=pick_sample(psi_decomposed,config);
# psi_sample=apply_sampling_projector(psi,Lx,Ly,config,Vp);



chi=10;

@btime Norm,trun_err=contract_whole_disk(psi_sample,chi);
Norm,trun_err=contract_whole_disk(psi_sample,chi);
@show Norm,trun_err


@btime Norm,trun_err=contract_sample(psi,Lx,Ly,config,Vp,contract_whole_disk);

@btime Norm,trun_err=contract_sample(psi_decomposed,Lx,Ly,config,Vp,contract_whole_disk);
# Norm_exact=exact_contraction(psi_sample);
# 



contract_history=disk_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly),Matrix{TensorMap}(undef,Lx,Ly));#create empty contract_history
@btime  contract_partial_disk(psi_sample,config,contract_history, chi);
Norm,trun_err,contract_history= contract_partial_disk(psi_sample,config,contract_history, chi);
@show Norm,trun_err

#############################
#test flip
coord=reshape(Vector(1:Lx*Ly),Lx,Ly);

pos1=[3,3];
pos2=[3,4];
config_new=deepcopy(config);
config_new[coord[pos1[1],pos1[2]]]=config[coord[pos2[1],pos2[2]]];
config_new[coord[pos2[1],pos2[2]]]=config[coord[pos1[1],pos1[2]]];
Norm1,_=contract_sample(psi_decomposed,Lx,Ly,config_new,Vp,contract_whole_disk);
Norm2,trun_err,contract_history= contract_partial_disk(pick_sample(psi_decomposed,config_new),config_new,contract_history, chi);
@assert abs((Norm1-Norm2)/Norm1)<1e-10;
@show Norm1


pos1=[1,1];
pos2=[1,2];
config_new=deepcopy(config);
config_new[coord[pos1[1],pos1[2]]]=config[coord[pos2[1],pos2[2]]];
config_new[coord[pos2[1],pos2[2]]]=config[coord[pos1[1],pos1[2]]];
Norm1,_=contract_sample(psi_decomposed,Lx,Ly,config_new,Vp,contract_whole_disk);
Norm2,trun_err,contract_history= contract_partial_disk(pick_sample(psi_decomposed,config_new),config_new,contract_history, chi);
@assert abs((Norm1-Norm2)/Norm1)<1e-10;
@show Norm1


pos1=[1,1];
pos2=[2,1];
config_new=deepcopy(config);
config_new[coord[pos1[1],pos1[2]]]=config[coord[pos2[1],pos2[2]]];
config_new[coord[pos2[1],pos2[2]]]=config[coord[pos1[1],pos1[2]]];
Norm1,_=contract_sample(psi_decomposed,Lx,Ly,config_new,Vp,contract_whole_disk);
Norm2,trun_err,contract_history= contract_partial_disk(pick_sample(psi_decomposed,config_new),config_new,contract_history, chi);
@assert abs((Norm1-Norm2)/Norm1)<1e-10;
@show Norm1

pos1=[6,1];
pos2=[6,2];
config_new=deepcopy(config);
config_new[coord[pos1[1],pos1[2]]]=config[coord[pos2[1],pos2[2]]];
config_new[coord[pos2[1],pos2[2]]]=config[coord[pos1[1],pos1[2]]];
Norm1,_=contract_sample(psi_decomposed,Lx,Ly,config_new,Vp,contract_whole_disk);
Norm2,trun_err,contract_history= contract_partial_disk(pick_sample(psi_decomposed,config_new),config_new,contract_history, chi);
@assert abs((Norm1-Norm2)/Norm1)<1e-10;
@show Norm1

pos1=[6,1];
pos2=[5,1];
config_new=deepcopy(config);
config_new[coord[pos1[1],pos1[2]]]=config[coord[pos2[1],pos2[2]]];
config_new[coord[pos2[1],pos2[2]]]=config[coord[pos1[1],pos1[2]]];
Norm1,_=contract_sample(psi_decomposed,Lx,Ly,config_new,Vp,contract_whole_disk);
Norm2,trun_err,contract_history= contract_partial_disk(pick_sample(psi_decomposed,config_new),config_new,contract_history, chi);
@assert abs((Norm1-Norm2)/Norm1)<1e-10;
@show Norm1


pos1=[4,5];
pos2=[4,6];
config_new=deepcopy(config);
config_new[coord[pos1[1],pos1[2]]]=config[coord[pos2[1],pos2[2]]];
config_new[coord[pos2[1],pos2[2]]]=config[coord[pos1[1],pos1[2]]];
Norm1,_=contract_sample(psi_decomposed,Lx,Ly,config_new,Vp,contract_whole_disk);
Norm2,trun_err,contract_history= partial_contract_sample(psi_decomposed,config_new,Vp,contract_history);
@btime contract_sample(psi_decomposed,Lx,Ly,config_new,Vp,contract_whole_disk);
@btime partial_contract_sample(psi_decomposed,config_new,Vp,contract_history);
@assert abs((Norm1-Norm2)/Norm1)<1e-10;
@show Norm1