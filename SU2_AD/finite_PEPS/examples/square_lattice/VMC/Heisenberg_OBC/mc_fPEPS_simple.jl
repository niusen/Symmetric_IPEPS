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








function localenergy(KEL::Matrix, W::Matrix, iconf_new::Vector,  NN_tuple_reduced::Vector{Tuple})
    # Compute the expectation value of the permutation operator
    elocal = Complex{Float64}(0.0, 0.0)  # Initialize local energy

    amplitude,_=contract_sample(psi,Lx,Ly,iconf_new,Vp,contract_fun);

    for i in 1:L
        for j in NN_tuple_reduced[i]  # Loop over half of the nearest neighbors
            randl = i
            randK = NN_tuple_reduced[randl][j]  # Neighbor site

            if iconf_new[randl] == iconf_new[randK]
                elocal += 1.0  # Diagonal term ⟨x|H|x⟩
            else
                iconf_new_flip=deepcopy(iconf_new);
                iconf_new_flip[randl]=iconf_new[randK];
                iconf_new_flip[randK]=iconf_new[randl];

                amplitude_flip,_=contract_sample(psi,Lx,Ly,iconf_new_flip,Vp,contract_fun);

    
                # overlap = -W[randK, d5] * W[randl, d6]  # Ratio ⟨y|ψ⟩ / ⟨x|ψ⟩
                # elocal += overlap


                #coefficient? J? -J? J/2? -J/2?
                elocal += coe*amplitude_flip/amplitude;
            end
        end
    end

    return elocal
end



function main()
    #load saved fPEPS data


    filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
    psi,Vp=load_fPEPS(Lx,Ly,filenm);
    
    normalize_PEPS!(psi,Vp,contract_whole_disk);#normalize psi such that the amplitude of a single config is close to 1
    
    
    contract_fun=contract_whole_disk;
    ##########################################

    coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours(Lx,Ly,"OBC");
    # neighbor_file = "Neighbor_matrix/all_nearest_neighbors_$(Lx)x$(Ly).jld2"
    # NNmatrix = JLD2.load(neighbor_file, "NNmatrix")

    # iconf_file = "Therm_iconf/initial_iconf_$(Lx).jld2"
    # initial_iconf = JLD2.load(iconf_file, "random_array")
    initial_iconf =initial_Neel_config(Lx,Ly);
    #Recall that iconf here has elements 1 (up spin) and -1 (down spin), unlike in our C++ code where we have 1 and 2.

    # evecs_file = "Ansatz/evecs_$(Lx)_dirac_pbcpbc.jld2"
    # U = JLD2.load(evecs_file, "evecs")


    KEL = zeros(Int, L, N)  # Initialize KEL as a matrix of integers

    #######################
    # W_prime = zeros(Complex{Float64}, L, L_N)
    # W_oneflavor = zeros(Complex{Float64}, L, L_N)
    #######################



    for j in 1:N
        ll = (-1)^(j - 1)  # Calculate ll for this column
        mask = (initial_iconf .== ll)  # Create a logical mask where iconf equals ll
        # .== means do an operation for every element of the array. Mask is an array of same size as initial_iconf, with elements true or false.

        indices = findall(mask)  # Get the indices where the condition is true
        KEL[indices, j] = 1:length(indices)  # Fill in the count for matching indices
    end

    #display(KEL)
    #display(initial_iconf)

    # redU = U[1:L, 1:(L_N)]


    W = zeros(Complex{Float64}, L,L)
    println(typeof(W))

    # ftW!(redU, KEL, W)

    # Initialize variables
    iconf_new = copy(initial_iconf)

    ebin1 = zeros(Complex{Float64}, binn)

    outputname = "test.csv"

    if isfile(outputname)
        rm(outputname)
    end

    open(outputname, "a") do file # "a" is for append

        @inbounds for i in 1:Nsteps  # Number of Monte Carlo steps, usually 1 million
            @inbounds for j in 1:Nbra  # Inner loop to create uncorrelated samples
                randl = rand(1:L)  # Picking a site at random; "l"
                rand2 = rand(1:length(NN_tuple[randl]))  # Picking randomly one of the 4 neighbors
                randK = NN_tuple[randl][rand2]  # Picking a neighbor at random to which electron wants to hop; "K"

                amplitude,_=contract_sample(psi,Lx,Ly,iconf_new,Vp,contract_fun);

                if iconf_new[randl] != iconf_new[randK]

                    iconf_new_flip=deepcopy(iconf_new);
                    iconf_new_flip[randl]=iconf_new[randK];
                    iconf_new_flip[randK]=iconf_new[randl];

                    amplitude_flip,_=contract_sample(psi,Lx,Ly,iconf_new_flip,Vp,contract_fun);


                    probratio = abs2(amplitude_flip/amplitude_flip)  # Probability of accepting configuration
                    eta_rand = rand()  # Random number from 0 to 1; "eta"

                    if eta_rand < probratio  # We accept the configuration

                        # Update KEL arrays and iconf
                        # Swap elements based on dl and dK values
                        KEL[randK, 1 + (dl == -1)], KEL[randl, 1 + (dl == -1)] = KEL[randl, 1 + (dl == -1)], KEL[randK, 1 + (dl == -1)]
                        KEL[randl, 1 + (dK == -1)], KEL[randK, 1 + (dK == -1)] = KEL[randK, 1 + (dK == -1)], KEL[randl, 1 + (dK == -1)]
                        iconf_new= iconf_new_flip;
                        amplitude=amplitude_flip;

                    end
                end
            end

            energyl1 = (localenergy(KEL, W, iconf_new, NN_tuple,NN_tuple_reduced) - 1.0 * L) / 2.0
            # The constant L/2 is required because the localenergy function computes the expectation value of the permutation operator.
            # The permutation operator is related to the S.S as follows: P_{ij} = 2 S_i . S_j + 1/2

            rems = mod1(i, binn)  # Binning to store fewer numbers, usually binn is order of 1000
            ebin1[rems] = energyl1

            if rems == binn
                #CSV.write(outputname, mean(ebin); append=true) 
                println(file, real(mean(ebin1)))
            end

            # Optional: Uncomment to print configuration every 999 steps
            if mod(i + 1, 999) == 0
                # println(outfile, "\n\n", iconf_new, "\n\n\n")
                println(file, "\n\n", iconf_new, "\n\n\n")
            end

            # if mod(i + 1, Nscra) == 0  # Recalculate W matrix from scratch to avoid numerical errors
            #     ftW!(redU, KEL, W)
            # end
        end

    end

end

# Profile.clear()
# @btime @profview main()

@time main()



