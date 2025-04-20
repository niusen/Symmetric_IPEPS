using Distributed
#number of workers to add and soft restrict of memory
#addprocs(50; exeflags=["--heap-size-hint=6G"])
#addprocs(1; exeflags=["--heap-size-hint=6G"])

@everywhere using LinearAlgebra:I,diagm,diag
@everywhere using TensorKit
@everywhere using Random
@everywhere using Printf
@everywhere using DelimitedFiles
@everywhere using CSV
@everywhere using DataFrames
@everywhere using JLD2,MAT

@everywhere cd(@__DIR__)



@everywhere include("../../../../state/iPEPS_ansatz.jl")
@everywhere include("../../../../setting/Settings.jl")
@everywhere include("../../../../setting/linearalgebra.jl")
@everywhere include("../../../../setting/tuple_methods.jl")
@everywhere include("../../../../environment/MC/finite_clusters.jl")

@everywhere include("../../../../environment/MC/contract_disk.jl")
@everywhere include("../../../../environment/MC/sampling.jl")
@everywhere include("../../../../environment/MC/mps_sweep.jl")

@everywhere begin
@show const Lattice="square";#"kagome", "square"
@show const Lx = 4      # number of sites along x / number of columns in the lattice
@show const Ly = 4      # number of sites along y / number of rows in the lattice
@show const D=2;#bond dimension of state
@show const chi=10;#bond dimension of environment
@show const use_mps_sweep=true;
@show const n_mps_sweep=5;

const L = Lx*Ly # total number of lattice sites
const Nbra = L             # Inner loop size, to generate uncorrelated samples, usually must be of size O(L).
const Ne = L            # Number of electrons on the lattice (for spin models this will always be equal to L)
@show const Nsteps = 600       # Total Monte Carlo steps
@show const binn = 200          # Bin size to store the data during the monte carlo run. 
@show const step_prepare = 200          # number of steps to get equilibrium. 
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



@everywhere function localenergy(psi::Matrix{TensorMap},iconf_new::Vector,sample_::Matrix{TensorMap},  NN_tuple_reduced::Vector{Tuple}, contract_history_::Contract_History)
    #fixed path contraction for computing energy, and unfixed path contraction for gradient

    # Compute the expectation value of the permutation operator
    global contract_fun,psi_decomposed,Vp
    config_grad=contract_disk_derivative(sample_,iconf_new, chi);
    config_grad=set_grad_config(config_grad,iconf_new,psi);

    elocal =zeros(ComplexF64,sum(length.(NN_tuple_reduced)));   # Initialize local energy
    if contraction_path=="verify"
        amplitude,sample_,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample_,Vp,contract_fun);
        amplitude_,sample_,_,contract_history_= partial_contract_sample(psi_decomposed,iconf_new,sample_,Vp,contract_history_);
        @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;
    elseif contraction_path=="full"
        amplitude,sample_,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample_,Vp,contract_fun);
    elseif contraction_path=="recycle"
        amplitude,sample_,_,contract_history_= partial_contract_sample(psi_decomposed,iconf_new,sample_,Vp,contract_history_);
    end

    ########
    config_grad=config_grad/amplitude;
    ########
    
    step=1;
    for i in 1:L
        for randK in NN_tuple_reduced[i]  # Loop over half of the nearest neighbors
            randl = i
            # randK = NN_tuple_reduced[randl][j]  # Neighbor site

            if iconf_new[randl] == iconf_new[randK]
                elocal[step] = 0.25;  # Diagonal term ⟨x|H|x⟩
                step=step+1;
            else
                iconf_new_flip=flip_config(iconf_new,randl,randK);

                if contraction_path=="verify"
                    amplitude_flip,sample_,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample_,Vp,contract_fun);
                    amplitude_flip_,sample_,_,_= partial_contract_sample(psi_decomposed,iconf_new_flip,sample_,Vp,contract_history_);
                    @assert abs(norm(amplitude_flip-amplitude_flip_)/amplitude_flip)<1e-10   string(amplitude_flip)*", "*string(amplitude_flip_);
                elseif contraction_path=="full"
                    amplitude_flip,sample_,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample_,Vp,contract_fun);
                elseif contraction_path=="recycle"
                    amplitude_flip,sample_,_,= partial_contract_sample(psi_decomposed,iconf_new_flip,sample_,Vp,contract_history_);
                end
                elocal[step] = 0.5*amplitude_flip/amplitude -0.25;#second term corresponds to diagonal term when two spins are opposite
                step=step+1
            end
        end
    end



    return elocal,sample_, config_grad
end



@everywhere function main(dir, worker_id, ntask)
    #load saved fPEPS data
    @show Nsteps_worker=Int(round(Nsteps/ntask));
    @assert step_prepare>=0;
    contraction_path="recycle";#"verify","full","recycle"

    filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
    psi,Vp=load_fPEPS(Lx,Ly,filenm);
    @show Vp
    global contraction_path, contract_fun, psi_decomposed, Vp
    contract_fun=contract_whole_disk;
    config_max=normalize_PEPS!(psi,Vp,contract_whole_disk);#normalize psi such that the amplitude of a single config is close to 1
    psi_decomposed=decompose_physical_legs(psi,Vp);

    sample=Matrix{TensorMap}(undef,Lx,Ly);
    ##########################################

    coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours_square(Lx,Ly,"OBC");

    initial_iconf =initial_Neel_config_square(Lx,Ly,1);
    #Recall that iconf here has elements 1 (up spin) and -1 (down spin), unlike in our C++ code where we have 1 and 2.

    #start from the config in test.csv


    #create empty contract_history
    contract_history=disk_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly),Matrix{TensorMap}(undef,Lx,Ly));


    # Initialize variables
    iconf_new = config_max;

    ebin1 = zeros(Complex{Float64}, binn, sum(length.(NN_tuple_reduced)));
    gradbin1 = Vector{Matrix{TensorMap}}(undef, binn);
    E_gradbin1=Vector{Matrix{TensorMap}}(undef, binn);
    ebin_file=zeros(Complex{Float64}, 0, sum(length.(NN_tuple_reduced)));


    outputname = dir*"id_"*string(worker_id)*"_chi"*string(chi)*".jld2"

    if isfile(outputname)
        rm(outputname)
    end

    #write empty variables in file
    jldopen(outputname, "w") do file
        file["E_terms"]=Matrix{ComplexF64}(undef,0,sum(length.(NN_tuple_reduced)));
        file["grads"]=Vector{Matrix{TensorMap}}(undef,0);
        file["E_grads"]=Vector{Matrix{TensorMap}}(undef,0);
    end

    #open(outputname, "a") do file # "a" is for append
    jldopen(outputname, "r+") do file

        # @inbounds for i in 1:Nsteps  # Number of Monte Carlo steps, usually 1 million
        #     @inbounds for j in 1:Nbra  # Inner loop to create uncorrelated samples
        for i in -step_prepare:Nsteps_worker  # Number of Monte Carlo steps, usually 1 million
            global ite_num
            ite_num=i;
            # if mod(i,100)==0;@show i;flush(stdout);end


            if contraction_path=="verify"
                amplitude,sample, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample,Vp,contract_fun);
                amplitude_,sample, _,contract_history= partial_contract_sample(psi_decomposed,iconf_new,sample,Vp,contract_history);
                @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;
            elseif contraction_path=="full"
                amplitude,sample, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new,sample,Vp,contract_fun);
            elseif contraction_path=="recycle"
                amplitude,sample,_,contract_history= partial_contract_sample(psi_decomposed,iconf_new,sample,Vp,contract_history);
            end

            for j in 1:Nbra  # Inner loop to create uncorrelated samples
                randl = rand(1:L)  # Picking a site at random; "l"
                rand2 = rand(1:length(NN_tuple[randl]))  # Picking randomly one of the 4 neighbors
                randK = NN_tuple[randl][rand2]  # Picking a neighbor at random to which electron wants to hop; "K"

                

                if iconf_new[randl] != iconf_new[randK]
                    

                    iconf_new_flip=flip_config(iconf_new,randl,randK);
                    if contraction_path=="verify"
                        amplitude_flip,sample, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample,Vp,contract_fun);
                        amplitude_flip_,sample, _,contract_history_flip= partial_contract_sample(psi_decomposed,iconf_new_flip,sample,Vp,contract_history);
                        @assert abs((amplitude_flip-amplitude_flip_)/amplitude_flip_)<1e-10   string(amplitude_flip)*", "*string(amplitude_flip_);
                    elseif contraction_path=="full"
                        amplitude_flip,sample, _=contract_sample(psi_decomposed,Lx,Ly,iconf_new_flip,sample,Vp,contract_fun);
                    elseif contraction_path=="recycle"
                        amplitude_flip,sample, _,contract_history_flip= partial_contract_sample(psi_decomposed,iconf_new_flip,sample,Vp,contract_history);
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
            
            if i>0
                energyl1,sample, grad_config_ = localenergy(psi,iconf_new,sample,NN_tuple_reduced,contract_history)

                rems = mod1(i, binn)  # Binning to store fewer numbers, usually binn is order of 1000
                ebin1[rems,:] = energyl1;
                gradbin1[rems] = grad_config_;
                E_gradbin1[rems] = conj(sum(energyl1))*grad_config_;

                if rems == binn
                    #println(file, join(mean(ebin1,dims=1), ","));flush(file);
                    E_terms=file["E_terms"];
                    grads=file["grads"];
                    E_grads=file["E_grads"];
                    if haskey(file, "E_terms")
                        delete!(file, "E_terms")
                        delete!(file, "grads")
                        delete!(file, "E_grads")
                    end
                    #@show mean(ebin1,dims=1)
                    
                    file["grads"]=push!(grads,average_grad(gradbin1));
                    file["E_grads"]=push!(E_grads,average_grad(E_gradbin1));
                    file["E_terms"]= vcat(E_terms,mean(ebin1,dims=1));
                    
                end

                # Optional: Uncomment to print configuration every 999 steps
                #save a good initial config for next time
                if mod(i + 1, 999) == 0
                    # println(outfile, "\n\n", iconf_new, "\n\n\n")
                    # println(file, "\n\n", iconf_new, "\n\n\n");flush(stdout);
                end

                if mod(i + 1, GC_spacing) == 0
                    GC.gc(true)
                    if malloc_trim()
                        #println("Memory trimmed successfully.")
                    else
                        println("No memory trimmed.")
                    end
                end
            end

        end

    end

end

# Profile.clear()
# @btime @profview main()


ntask=nworkers();
#@time main(dir, 1, ntask)
@sync begin
    for cp=1:ntask
        worker_id=workers()[cp]
        @spawnat worker_id main(dir, worker_id, ntask);
    end
end

function read_data()
    coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours_square(Lx,Ly,"OBC");
    Eterms_set=Matrix{ComplexF64}(undef,0,sum(length.(NN_tuple_reduced)));
    grads_set=Vector{Matrix{TensorMap}}(undef,0);
    E_grads_set=Vector{Matrix{TensorMap}}(undef,0);


    for cp =1:ntask
       
 
        
        outputname = dir*"id_"*string(workers()[cp])*"_chi"*string(chi)*".jld2";
        # Read the list of numbers from the CSV file
        # data = open(outputname, "r") do file
        #     [parse(ComplexF64, line) for line in readlines(file)]
        # end
        data=load(outputname);
        Eterms_set=vcat(Eterms_set,data["E_terms"])  
        grads_set=vcat(grads_set,data["grads"])   
        E_grads_set=vcat(E_grads_set,data["E_grads"])        

        #rm(outputname)
        
    end
    return Eterms_set, grads_set, E_grads_set
end

Eterms_set, grads_set, E_grads_set=read_data();

function grad_analysis(Eterms_set, grads_set, E_grads_set)

    E_set=sum(Eterms_set,dims=2);

    E_mean=mean(E_set);
    grad_mean=mean(grads_set);
    E_grad_mean=mean(E_grads_set);
    Grad=E_grad_mean-E_mean*grad_mean;

   


    filenm="grad_"*string(Lx)*"x"*string(Ly)*"_D"*string(D)*"_chi"*string(chi)*".jld2"
    jldsave(filenm; Eterms_set, E_mean, grad_mean, E_grad_mean, Grad)    

    return Grad
end
    
Grad=grad_analysis(Eterms_set, grads_set, E_grads_set);

