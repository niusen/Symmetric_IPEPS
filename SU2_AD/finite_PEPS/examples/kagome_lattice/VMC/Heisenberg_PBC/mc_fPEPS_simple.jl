using Distributed
#number of workers to add and soft restrict of memory
addprocs(50; exeflags=["--heap-size-hint=6G"])

@everywhere using LinearAlgebra:I,diagm,diag
@everywhere using TensorKit
@everywhere using Random
@everywhere using Printf
@everywhere using DelimitedFiles
@everywhere using CSV
@everywhere using DataFrames
@everywhere using JLD2
using Statistics
using MAT

@everywhere cd(@__DIR__)


@everywhere include("../../../../state/iPEPS_ansatz.jl")
@everywhere include("../../../../setting/Settings.jl")
@everywhere include("../../../../setting/linearalgebra.jl")
@everywhere include("../../../../setting/tuple_methods.jl")

@everywhere include("../../../../environment/MC/contract_torus.jl")
@everywhere include("../../../../environment/MC/sampling.jl")
@everywhere include("../../../../environment/MC/sampling_eliminate_physical_leg.jl")
@everywhere include("../../../../environment/MC/build_degenerate_states.jl")

@everywhere begin
@show const Lx = 10      # number of sites along x / number of columns in the lattice
@show const Ly = 10      # number of sites along y / number of rows in the lattice
@show const D=3;#bond dimension of state
@show const chi=40;#bond dimension of environment

const L = 3*Lx*Ly # total number of lattice sites
const Nbra = L             # Inner loop size, to generate uncorrelated samples, usually must be of size O(L).
const Ne = L            # Number of electrons on the lattice (for spin models this will always be equal to L)
@show const Nsteps = 100000       # Total Monte Carlo steps
@show const binn = 200          # Bin size to store the data during the monte carlo run. 
const GC_spacing = 200          # garbage collection
end

###################
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
dir=hostnm*"_"*string(Lx)*"x"*string(Ly)*"_D"*string(D)*"/";
isdir(dir) || mkdir(dir)
###################

@everywhere include("sq_constants.jl")


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

#input  TensorKit.   and then press Tab to show all properties that may occupy memory

# @everywhere TensorKit.usebraidcache_abelian[] = false 
# @everywhere TensorKit.usebraidcache_nonabelian[] = false
@everywhere TensorKit.braidcache.maxsize=1000
@everywhere TensorKit.transposecache.maxsize=1000
# @everywhere TensorKit.usetransposecache
@everywhere TensorKit.treepermutercache.maxsize=1000
@everywhere TensorKit.GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE.maxsize=1000
# Base.summarysize(TensorKit.treepermutercache)
# Base.summarysize(TensorKit.GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE)

@everywhere function localenergy(psi_decomposed::Array{TensorMap},iconf_new::Vector,NN_tuple_reduced::Vector{Tuple}, sample_::Matrix{TensorMap}, contract_history_::Contract_History)
    # Compute the expectation value of the permutation operator
    elocal = Complex{Float64}(0.0, 0.0)  # Initialize local energy
    global contract_fun,Vp

    # if contraction_path=="verify"
    #     amplitude,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,Vp,contract_fun);
    #     amplitude_,_,contract_history_= partial_contract_sample(psi_decomposed,iconf_new,Vp,contract_history_);
    #     @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;
    # elseif contraction_path=="full"
    #     amplitude,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,Vp,contract_fun);
    # elseif contraction_path=="recycle"
    #     amplitude,_,contract_history_= partial_contract_sample(psi_decomposed,iconf_new,Vp,contract_history_);
    # end

    if contraction_path=="verify"
        amplitude,sample_,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample_, Vp,contract_fun);
        amplitude_,sample_,_,contract_history_= partial_contract_sample(psi_decomposed,iconf_new,sample_, Vp,contract_history_);
        @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;
    elseif contraction_path=="full"
        amplitude,sample_,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample_, Vp,contract_fun);
    elseif contraction_path=="recycle"
        amplitude,sample_,_,contract_history_= partial_contract_sample(psi_decomposed,iconf_new,sample_, Vp,contract_history_);
    end

    for i in 1:L
        for randK in NN_tuple_reduced[i]  # Loop over half of the nearest neighbors
            randl = i
            # randK = NN_tuple_reduced[randl][j]  # Neighbor site

            if iconf_new[randl] == iconf_new[randK]
                elocal += 0.25;  # Diagonal term ⟨x|H|x⟩
            else
                iconf_new_flip=flip_config(iconf_new,randl,randK);

                # if contraction_path=="verify"
                #     amplitude_flip,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,Vp,contract_fun);
                #     amplitude_flip_,_,_= partial_contract_sample(psi_decomposed,iconf_new_flip,Vp,contract_history_);
                #     @assert abs(norm(amplitude_flip-amplitude_flip_)/amplitude_flip)<1e-10   string(amplitude_flip)*", "*string(amplitude_flip_);
                # elseif contraction_path=="full"
                #     amplitude_flip,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,Vp,contract_fun);
                # elseif contraction_path=="recycle"
                #     amplitude_flip,_,= partial_contract_sample(psi_decomposed,iconf_new_flip,Vp,contract_history_);
                # end
                if contraction_path=="verify"
                    amplitude_flip,sample_,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample_, Vp,contract_fun);
                    amplitude_flip_,sample_,_,_= partial_contract_sample(psi_decomposed,iconf_new_flip,sample_, Vp,contract_history_);
                    @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;
                elseif contraction_path=="full"
                    amplitude_flip,sample_,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample_, Vp,contract_fun);
                elseif contraction_path=="recycle"
                    amplitude_flip,sample_,_,_= partial_contract_sample(psi_decomposed,iconf_new_flip,sample_, Vp,contract_history_);
                end
                elocal += 0.5*amplitude_flip/amplitude -0.25;
            end
        end
    end

    return elocal,sample_
end





@everywhere function main(dir::String, worker_id::Int, ntask::Int)
    #load saved fPEPS data
    @show Nsteps_worker=Int(round(Nsteps/ntask));
    contraction_path="recycle";#"verify","full","recycle"
 
    filenm="CSL_D"*string(D)*"_U1";
    @show to_dense=false;#convert to dense
    psi0,Vp,Vv=load_fPEPS_from_iPEPS(Lx,Ly,filenm,to_dense);

    global contraction_path, contract_fun, Vp, projector_method
    projector_method="1";#"1" or "2"
    contract_fun=contract_whole_torus_boundaryMPS;
    normalize_PEPS!(psi0,Vp,contract_whole_torus_boundaryMPS);#normalize psi0 such that the amplitude of a single config is close to 1
    #psi_00,psi_0pi,psi_pi0,psi_pipi =construct_4_states(psi0,Vv);#four states
    psi_BC_set =construct_4_states(psi0,Vv);#four states

    psi=psi_BC_set[BC1];
    psi_decomposed=decompose_physical_legs(psi,Vp);


    sample=Matrix{TensorMap}(undef,Lx,Ly);

    ##########################################

    coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours(Lx,Ly,"PBC");

    initial_iconf =initial_Neel_config(Lx,Ly,1);
    #Recall that iconf here has elements 1 (up spin) and -1 (down spin), unlike in our C++ code where we have 1 and 2.

    #start from the config in test.csv


    #create empty contract_history
    contract_history=torus_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly));


    # Initialize variables
    iconf_new = copy(initial_iconf)

    ebin1 = zeros(Complex{Float64}, binn)

    
    outputname = dir*"id_"*string(worker_id)*"_chi"*string(chi)*".csv"

    if isfile(outputname)
        rm(outputname)
    end

    open(outputname, "a") do file # "a" is for append

        # @inbounds for i in 1:Nsteps_worker  # Number of Monte Carlo steps, usually 1 million
        #     @inbounds for j in 1:Nbra  # Inner loop to create uncorrelated samples
        for i in 1:Nsteps_worker  # Number of Monte Carlo steps, usually 1 million
            global ite_num
            ite_num=i;
            # @show i
            # if mod(i,100)==0;@show i;flush(stdout);end


            if contraction_path=="verify"
                amplitude,sample, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample,Vp,contract_fun);
                amplitude_, sample, _,contract_history= partial_contract_sample(psi_decomposed,iconf_new,sample,Vp,contract_history);
                @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;
            elseif contraction_path=="full"
                amplitude,sample, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample, Vp,contract_fun);
            elseif contraction_path=="recycle"
                amplitude,sample, _,contract_history= partial_contract_sample(psi_decomposed,iconf_new,sample, Vp,contract_history);
                # _,sample1,_,contract_history1= partial_contract_sample(psi_decomposed1,iconf_new,sample1,Vp,contract_history1);
            end

            for j in 1:Nbra  # Inner loop to create uncorrelated samples
                randl = rand(1:L)  # Picking a site at random; "l"
                rand2 = rand(1:length(NN_tuple[randl]))  # Picking randomly one of the 4 neighbors
                randK = NN_tuple[randl][rand2]  # Picking a neighbor at random to which electron wants to hop; "K"

                

                if iconf_new[randl] != iconf_new[randK]
                    

                    iconf_new_flip=flip_config(iconf_new,randl,randK);
                    if contraction_path=="verify"
                        amplitude_flip,sample, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample, Vp,contract_fun);
                        amplitude_flip_,sample, _,contract_history_flip= partial_contract_sample(psi_decomposed,iconf_new_flip,sample, Vp,contract_history);
                        @assert abs((amplitude_flip-amplitude_flip_)/amplitude_flip_)<1e-10   string(amplitude_flip)*", "*string(amplitude_flip_);
                    elseif contraction_path=="full"
                        amplitude_flip,sample, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample, Vp,contract_fun);
                    elseif contraction_path=="recycle"
                        amplitude_flip,sample, _, contract_history_flip= partial_contract_sample(psi_decomposed,iconf_new_flip,sample, Vp,contract_history);
                    end

                    probratio = abs2(amplitude_flip/amplitude)  # Probability of accepting configuration
                    eta_rand = rand()  # Random number from 0 to 1; "eta"

                    if eta_rand < probratio  # We accept the configuration
                        # println("accept")
                        iconf_new= iconf_new_flip;
                        amplitude=amplitude_flip;
                        if contraction_path in ("verify","recycle")
                            contract_history=contract_history_flip;
                        end
                    end
                end
            end

            energy,sample= localenergy(psi_decomposed, iconf_new, NN_tuple_reduced, sample, contract_history);


            rems = mod1(i, binn)  # Binning to store fewer numbers, usually binn is order of 1000
            ebin[rems] = energy


            if rems == binn
                #println( mean(ebin1))
                #CSV.write(outputname, mean(ebin1); append=true) 
                println(file, real(mean(ebin1)));flush(file);
            end

            # Optional: Uncomment to print configuration every 999 steps
            #save a good initial config for next time
            if mod(i + 1, 999) == 0
                # println(outfile, "\n\n", iconf_new, "\n\n\n")
                # println(file, "\n\n", iconf_new, "\n\n\n");flush(stdout);
            end

            if mod(i + 1, 100) == 0
                GC.gc(true)
                if malloc_trim()
                    #println("Memory trimmed successfully.")
                else
                    println("No memory trimmed.")
                end
            end

        end

    end
    GC.gc(true);

end



    


ntask=nworkers();
# main(dir, 1, ntask, BC1,BC2);
@sync begin
    for cp=1:ntask
        worker_id=workers()[cp]
        @spawnat worker_id main(dir, worker_id, ntask);
    end
end

data_set=Vector{ComplexF64}(undef,0);
for cp =1:ntask
    outputname = dir*"id_"*string(workers()[cp])*"_chi"*string(chi)*".csv";
    # Read the list of numbers from the CSV file
    data = open(outputname, "r") do file
        [parse(ComplexF64, line) for line in readlines(file)]
    end
    data_set=vcat(data_set,data);
    #rm(outputname)
end


#error analysis
# Output file

bin_size_set=Vector{Int}(undef,0);
energy_set=Vector{ComplexF64}(undef,0); 
std_dev_set=Vector{Float64}(undef,0);

bin_size = 1

total_data_size = length(data_set)
while bin_size < total_data_size
    # Bin the data
    binned = [mean(data_set[i:min(i+bin_size-1, total_data_size)]) for i in 1:bin_size:total_data_size]
    
    # Compute mean energy per site
    energy = mean(binned) 
    
    # Compute standard deviation
    std_dev = std(binned; corrected=false) / (sqrt(length(binned)))

    # Write results to the output file
    push!(bin_size_set,bin_size);
    push!(energy_set,energy);
    push!(std_dev_set,std_dev);

    # Double the bin size
    bin_size *= 2
end


matnm=string(Lx)*"x"*string(Ly)*"_D"*string(D)*"_chi"*string(chi)*".mat"
matwrite(matnm, Dict(
"bin_size_set"=>bin_size_set,
"energy_set"=>energy_set,
"std_dev_set"=>std_dev_set,
"data_set"=>data_set[BC2]
); compress = false)    

    






