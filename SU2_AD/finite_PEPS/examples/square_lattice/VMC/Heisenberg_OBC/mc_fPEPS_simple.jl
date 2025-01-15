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
include("sq_constants.jl")


include("../../../../state/iPEPS_ansatz.jl")

include("../../../../setting/Settings.jl")
include("../../../../setting/linearalgebra.jl")
include("../../../../setting/tuple_methods.jl")

include("../../../../environment/MC/contract_disk.jl")
include("../../../../environment/MC/sampling.jl")


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



function ftW!(redU::Matrix, KEL::Matrix, W::Matrix)
    # Constants and initialization
    cc = 1.00
    # Iterate over each flavor (column of KEL)
    for i in 1:N
        dummy = L_N
        
        # Create the initial dumM matrix (filled with zeros initially)
        dumM = zeros(Complex{Float64}, dummy, dummy)
        # Fill dumM based on KEL and redU (find indices where KEL == j + 1)
        for j in 1:dummy
            dummy2 = findfirst(KEL[:, i] .== j)  # Find the row index where KEL == j


            dumM[j, :] = redU[dummy2, 1:dummy] # j'th row of dumM is given by the dummy2'th row of redU. .= is element wise assigning.
        end

        # Inverse of dumM
        dumMinverse = inv(dumM)

        # Update W matrix
        for j in 1:L
            for k in 1:dummy
                dummy2 = (i - 1) * dummy + k
                W[j, dummy2] = sum(redU[j, :] .* dumMinverse[:, k])
            end
        end
    end
end


function updateW!(W::Matrix, d3::Number, randK::Number, flavor::Number,W_prime::Matrix,W_oneflavor::Matrix)
    # Initialize W_prime and W_oneflavor
    # W_prime = zeros(Complex{Float64}, L, L_N)
    # W_oneflavor = zeros(Complex{Float64}, L, L_N)

    W_prime=W_prime*0;
    W_oneflavor=W_oneflavor*0;
    
    # Define the flavor-specific slice of W
    x = (flavor == 1) ? 0 : L_N # Unlike C++, here flavor can take values +1 or -1.
    W_oneflavor = W[1:L, x+1:x + L_N]
    
    W_oneflavor_randK=W_oneflavor[randK, :];

    # Loop over the matrix and perform the update
    # for II in 1:L
        # for j in 1:(L_N)
            # check = 0 
            # if j == d3
            #     check = 1
            # end
            # W_prime[II, j] = W_oneflavor[II, j] - (W_oneflavor[II, d3] / W_oneflavor[randK, d3]) * (W_oneflavor[randK, j] - check)
            
            # check=(j == d3);
            check=((1:L_N).==d3);

            W_prime[1:L, 1:(L_N)] = W_oneflavor[1:L, 1:(L_N)] - (W_oneflavor[1:L, d3] / W_oneflavor_randK[d3]) * permutedims((W_oneflavor_randK[1:(L_N)] - check))
        # end
    # end
    
    # Update the original W matrix with the new values
    W[1:L, x+1:x + L_N] = W_prime
end


function localenergy(KEL::Matrix, W::Matrix, iconf_new::Vector,  NN_tuple_reduced::Vector{Tuple})
    # Compute the expectation value of the permutation operator
    elocal = Complex{Float64}(0.0, 0.0)  # Initialize local energy

    for i in 1:L
        for j in NN_tuple_reduced[i]  # Loop over half of the nearest neighbors
            randl = i
            randK = NN_tuple_reduced[randl][j]  # Neighbor site

            if iconf_new[randl] == iconf_new[randK]
                elocal += 1.0  # Diagonal term ⟨x|H|x⟩
            else
                dl = iconf_new[randl]
                dK = iconf_new[randK]
                
                d3 = KEL[randl, 1 + (dl == -1)]  # If dl == 1, use 1; if dl == -1, use 2
                d5 = d3 + (dl == -1) * (L_N)  # If dl == -1, add L_N
                
                d4 = KEL[randK, 1 + (dK == -1)]  # If dK == 1, use 1; if dK == -1, use 2
                d6 = d4 + (dK == -1) * (L_N)  # If dK == -1, add L_N
    
                overlap = -W[randK, d5] * W[randl, d6]  # Ratio ⟨y|ψ⟩ / ⟨x|ψ⟩
                elocal += overlap
            end
        end
    end

    return elocal
end



function main()

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
    W_prime = zeros(Complex{Float64}, L, L_N)
    W_oneflavor = zeros(Complex{Float64}, L, L_N)
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

    redU = U[1:L, 1:(L_N)]


    W = zeros(Complex{Float64}, L,L)
    println(typeof(W))

    ftW!(redU, KEL, W)

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
                rand2 = rand(1:length(NN_tuple[rand1]))  # Picking randomly one of the 4 neighbors
                randK = NN_tuple[randl][rand2]  # Picking a neighbor at random to which electron wants to hop; "K"

                if iconf_new[randl] != iconf_new[randK]
                    dl = iconf_new[randl]
                    dK = iconf_new[randK]
                    
                    d3 = KEL[randl, 1 + (dl == -1)]  # If dl == 1, use 1; if dl == -1, use 2
                    d5 = d3 + (dl == -1) * (L_N)  # If dl == -1, add L_N
                    
                    d4 = KEL[randK, 1 + (dK == -1)]  # If dK == 1, use 1; if dK == -1, use 2
                    d6 = d4 + (dK == -1) * (L_N)  # If dK == -1, add L_N
                    
                    #println("d3 index: ", 1 + (dl == -1), " d4 index: ", 1 + (dK == -1))
                    #println("i: ", i, "  j: ", j, " dl: ", dl, "  dK: ", dK, " d5: ", d5, "  d6: ", d6, " d3: ", d3, "  d4: ", d4)
                    probratio = abs2(W[randK, d5] * W[randl, d6])  # Probability of accepting configuration
                    eta_rand = rand()  # Random number from 0 to 1; "eta"

                    if eta_rand < probratio  # We accept the configuration
                        updateW!(W, d3, randK, iconf_new[randl],W_prime,W_oneflavor)  # Update W matrix
                        updateW!(W, d4, randl, iconf_new[randK],W_prime,W_oneflavor)  # Modify the second W matrix for the other flavor
                        #println("dl index: ", dl, " dK index: ", dK)
                        # Update KEL arrays and iconf
                        # Swap elements based on dl and dK values
                        KEL[randK, 1 + (dl == -1)], KEL[randl, 1 + (dl == -1)] = KEL[randl, 1 + (dl == -1)], KEL[randK, 1 + (dl == -1)]
                        KEL[randl, 1 + (dK == -1)], KEL[randK, 1 + (dK == -1)] = KEL[randK, 1 + (dK == -1)], KEL[randl, 1 + (dK == -1)]
                        iconf_new[randl], iconf_new[randK] = iconf_new[randK], iconf_new[randl]
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



