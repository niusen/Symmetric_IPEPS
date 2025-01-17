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

const Lx = 8      # number of sites along x / number of columns in the lattice
const Ly = 8      # number of sites along y / number of rows in the lattice
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








function localenergy(iconf_new::Vector,  NN_tuple_reduced::Vector{Tuple}, contract_history_::Contract_History)
    # Compute the expectation value of the permutation operator
    elocal = Complex{Float64}(0.0, 0.0)  # Initialize local energy
    global contract_fun,psi_decomposed,Vp

    if contraction_path=="verify"
        amplitude,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,Vp,contract_fun);
        amplitude_,_,contract_history_= partial_contract_sample(psi_decomposed,iconf_new,Vp,contract_history_);
        @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;
    elseif contraction_path=="full"
        amplitude,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,Vp,contract_fun);
    elseif contraction_path=="recycle"
        amplitude,_,contract_history_= partial_contract_sample(psi_decomposed,iconf_new,Vp,contract_history_);
    end

    for i in 1:L
        for randK in NN_tuple_reduced[i]  # Loop over half of the nearest neighbors
            randl = i
            # randK = NN_tuple_reduced[randl][j]  # Neighbor site

            if iconf_new[randl] == iconf_new[randK]
                elocal += 0.25;  # Diagonal term ⟨x|H|x⟩
            else
                iconf_new_flip=flip_config(iconf_new,randl,randK);

                if contraction_path=="verify"
                    amplitude_flip,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,Vp,contract_fun);
                    amplitude_flip_,_,_= partial_contract_sample(psi_decomposed,iconf_new_flip,Vp,contract_history_);
                    @assert abs(norm(amplitude_flip-amplitude_flip_)/amplitude_flip)<1e-10   string(amplitude_flip)*", "*string(amplitude_flip_);
                elseif contraction_path=="full"
                    amplitude_flip,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,Vp,contract_fun);
                elseif contraction_path=="recycle"
                    amplitude_flip,_,= partial_contract_sample(psi_decomposed,iconf_new_flip,Vp,contract_history_);
                end
                elocal += 0.5*amplitude_flip/amplitude -0.25;
            end
        end
    end

    return elocal
end



function main()
    #load saved fPEPS data

    contraction_path="recycle";#"verify","full","recycle"

    filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
    psi,Vp=load_fPEPS(Lx,Ly,filenm);
    global contraction_path, contract_fun, psi_decomposed, Vp
    contract_fun=contract_whole_disk;
    normalize_PEPS!(psi,Vp,contract_whole_disk);#normalize psi such that the amplitude of a single config is close to 1
    psi_decomposed=decompose_physical_legs(psi,Vp);

    
    ##########################################

    coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours(Lx,Ly,"OBC");

    initial_iconf =initial_Neel_config(Lx,Ly,1);
    #Recall that iconf here has elements 1 (up spin) and -1 (down spin), unlike in our C++ code where we have 1 and 2.

    #start from the config in test.csv


    #create empty contract_history
    contract_history=disk_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly),Matrix{TensorMap}(undef,Lx,Ly));


    # Initialize variables
    iconf_new = copy(initial_iconf)

    ebin1 = zeros(Complex{Float64}, binn)

    outputname = "test.csv"

    if isfile(outputname)
        rm(outputname)
    end

    open(outputname, "a") do file # "a" is for append

        # @inbounds for i in 1:Nsteps  # Number of Monte Carlo steps, usually 1 million
        #     @inbounds for j in 1:Nbra  # Inner loop to create uncorrelated samples
        for i in 1:50  # Number of Monte Carlo steps, usually 1 million
            # if mod(i,100)==0;@show i;flush(stdout);end


            if contraction_path=="verify"
                amplitude,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,Vp,contract_fun);
                amplitude_,_,contract_history= partial_contract_sample(psi_decomposed,iconf_new,Vp,contract_history);
                @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;
            elseif contraction_path=="full"
                amplitude,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,Vp,contract_fun);
            elseif contraction_path=="recycle"
                amplitude,_,contract_history= partial_contract_sample(psi_decomposed,iconf_new,Vp,contract_history);
            end

            for j in 1:Nbra  # Inner loop to create uncorrelated samples
                randl = rand(1:L)  # Picking a site at random; "l"
                rand2 = rand(1:length(NN_tuple[randl]))  # Picking randomly one of the 4 neighbors
                randK = NN_tuple[randl][rand2]  # Picking a neighbor at random to which electron wants to hop; "K"

                

                if iconf_new[randl] != iconf_new[randK]
                    

                    iconf_new_flip=flip_config(iconf_new,randl,randK);
                    if contraction_path=="verify"
                        amplitude_flip,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,Vp,contract_fun);
                        amplitude_flip_,_,contract_history_flip= partial_contract_sample(psi_decomposed,iconf_new_flip,Vp,contract_history);
                        @assert abs((amplitude_flip-amplitude_flip_)/amplitude_flip_)<1e-10   string(amplitude_flip)*", "*string(amplitude_flip_);
                    elseif contraction_path=="full"
                        amplitude_flip,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,Vp,contract_fun);
                    elseif contraction_path=="recycle"
                        amplitude_flip,_,contract_history_flip= partial_contract_sample(psi_decomposed,iconf_new_flip,Vp,contract_history);
                    end


                    probratio = abs2(amplitude_flip/amplitude)  # Probability of accepting configuration
                    eta_rand = rand()  # Random number from 0 to 1; "eta"

                    if eta_rand < probratio  # We accept the configuration
                        # println("accept")
                        iconf_new= deepcopy(iconf_new_flip);
                        amplitude=deepcopy(amplitude_flip);
                        if contraction_path in ("verify","recycle")
                            contract_history=deepcopy(contract_history_flip);
                        end
                    end
                end
            end
            
            energyl1 = localenergy(iconf_new,NN_tuple_reduced,contract_history)

            rems = mod1(i, binn)  # Binning to store fewer numbers, usually binn is order of 1000
            ebin1[rems] = energyl1

            if rems == binn
                #CSV.write(outputname, mean(ebin); append=true) 
                println(file, real(mean(ebin1)));flush(stdout);
            end

            # Optional: Uncomment to print configuration every 999 steps
            #save a good initial config for next time
            if mod(i + 1, 999) == 0
                # println(outfile, "\n\n", iconf_new, "\n\n\n")
                # println(file, "\n\n", iconf_new, "\n\n\n");flush(stdout);
            end

            if mod(i + 1, 200) == 0
                println(real(mean(ebin1)));flush(stdout);
            end

        end

    end

end

# Profile.clear()
# @btime @profview main()

@time main();


