# Simple monte carlo code that computes the energy for the square lattice Heisenberg model with nearest neighbor interactions.

using LinearAlgebra:I,diagm,diag
using TensorKit
using Random
using Printf
using DelimitedFiles
using CSV
using DataFrames
using Pkg
using JLD2
using BenchmarkTools
using Profile
# using ProfileView
cd(@__DIR__)



include("../../../../state/iPEPS_ansatz.jl")

include("../../../../setting/Settings.jl")
include("../../../../setting/linearalgebra.jl")
include("../../../../setting/tuple_methods.jl")

include("../../../../environment/MC/contract_disk.jl")
include("../../../../environment/MC/sampling.jl")

const Lx = 6      # number of sites along x / number of columns in the lattice
const Ly = 6      # number of sites along y / number of rows in the lattice
const D=2;#bond dimension of state
const chi=10;#bond dimension of environment

include("sq_constants.jl")

####################
#use single core
import LinearAlgebra.BLAS as BLAS
using Base.Threads

n_cpu=1;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);

Base.Sys.set_process_title("C"*string(n_cpu)*"_fPEPS")
pid=getpid();
println("pid="*string(pid));;flush(stdout);
####################





global contract_fun,psi_decomposed,Vp
contract_fun=contract_whole_disk;

data=load("test_contract.jld2");
psi_decomposed=data["psi_decomposed"]
iconf_new_flip=data["iconf_new_flip"]
Vp=data["Vp"]
contract_history=data["contract_history"]

@show reshape(iconf_new_flip-contract_history.config,Lx,Ly)



# amplitude_flip,_,mps_top_set_standard,mps_bot_set_standard=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,Vp,contract_fun);
amplitude_flip,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,Vp,contract_fun);
amplitude_flip_,_,contract_history_flip= partial_contract_sample(psi_decomposed,iconf_new_flip,Vp,contract_history);

jldsave("test_contract.jld2";psi_decomposed,iconf_new_flip,Vp);
@assert abs((amplitude_flip-amplitude_flip_)/amplitude_flip_)<1e-10   string(amplitude_flip)*", "*string(amplitude_flip_);


empty_contract_history=disk_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly),Matrix{TensorMap}(undef,Lx,Ly));
amplitude_flip_correct,_,contract_history_flip_correct= partial_contract_sample(psi_decomposed,iconf_new_flip,Vp,empty_contract_history);


history1=contract_history_flip_correct;
history2=contract_history
for cx=1:Lx
    for cy=1:3
        pos=[cx,cy]
        println(norm(history1.mps_bot_set[pos[1],pos[2]]-history2.mps_bot_set[pos[1],pos[2]]))
    end
end

for cx=1:Lx
    for cy=4:6
        pos=[cx,cy]
        println(norm(history1.mps_top_set[pos[1],pos[2]]-history2.mps_top_set[pos[1],pos[2]]))
    end
end

# for cx=1:Lx
#     for cy=4:6
# pos=[cx,cy]
# println(norm(history1.mps_top_set[pos[1],pos[2]]-mps_top_set_standard[pos[1],pos[2]]))
#     end
# end


