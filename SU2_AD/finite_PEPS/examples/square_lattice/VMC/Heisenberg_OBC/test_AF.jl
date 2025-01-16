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

const Lx = 4      # number of sites along x / number of columns in the lattice
const Ly = 4      # number of sites along y / number of rows in the lattice
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











filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
psi,Vp=load_fPEPS(Lx,Ly,filenm);
global contract_fun,psi_decomposed,Vp
contract_fun=contract_whole_disk;
normalize_PEPS!(psi,Vp,contract_whole_disk);#normalize psi such that the amplitude of a single config is close to 1
psi_decomposed=decompose_physical_legs(psi,Vp);


##########################################

coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours(Lx,Ly,"OBC");

initial_iconf =initial_Neel_config(Lx,Ly,-1);
#Recall that iconf here has elements 1 (up spin) and -1 (down spin), unlike in our C++ code where we have 1 and 2.



# Initialize variables
iconf_new = copy(initial_iconf)





    for i in 1:10  # Number of Monte Carlo steps, usually 1 million
        # if mod(i,100)==0;@show i;flush(stdout);end
        for j in 1:Nbra  # Inner loop to create uncorrelated samples
            randl = rand(1:L)  # Picking a site at random; "l"
            rand2 = rand(1:length(NN_tuple[randl]))  # Picking randomly one of the 4 neighbors
            randK = NN_tuple[randl][rand2]  # Picking a neighbor at random to which electron wants to hop; "K"

            

            if iconf_new[randl] != iconf_new[randK]
                amplitude,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,Vp,contract_fun);

                iconf_new_flip=flip_config(iconf_new,randl,randK);

                amplitude_flip,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,Vp,contract_fun);


                probratio = abs2(amplitude_flip/amplitude)  # Probability of accepting configuration
                eta_rand = rand()  # Random number from 0 to 1; "eta"

                if eta_rand < probratio  # We accept the configuration

                    iconf_new= deepcopy(iconf_new_flip);
                    amplitude=deepcopy(amplitude_flip);
                    iconf_new_=reshape(iconf_new,Lx,Ly);
                    @show iconf_new_
                    @show amplitude
                end
            end
        end
        

   

 


    end







