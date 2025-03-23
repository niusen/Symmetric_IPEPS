using Distributed
@everywhere using LinearAlgebra:I,diagm,diag
@everywhere using TensorKit
@everywhere using Random
@everywhere using Printf
@everywhere using DelimitedFiles
@everywhere using CSV
@everywhere using DataFrames
@everywhere using JLD2

@everywhere cd(@__DIR__)

@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
dir=hostnm*"/";
isdir(dir) || mkdir(dir)


@everywhere include("../../../../state/iPEPS_ansatz.jl")

@everywhere include("../../../../setting/Settings.jl")
@everywhere include("../../../../setting/linearalgebra.jl")
@everywhere include("../../../../setting/tuple_methods.jl")

@everywhere include("../../../../environment/MC/contract_torus.jl")
@everywhere include("../../../../environment/MC/sampling.jl")
@everywhere include("../../../../environment/MC/sampling_eliminate_physical_leg.jl")
@everywhere include("../../../../environment/MC/build_degenerate_states.jl")

@everywhere begin
const Lx = 6      # number of sites along x / number of columns in the lattice
const Ly = 6      # number of sites along y / number of rows in the lattice
const D=3;#bond dimension of state
const chi=10;#bond dimension of environment
end

@everywhere include("sq_constants.jl")
@everywhere include("error_analysis.jl")

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

function get_psi_BCs(state_filenm, psi)

end






@everywhere function overlap_ratio(iconf_new::Vector,amplitude::Number, contract_history_otherstate_::Contract_History)
    # Compute the expectation value of the permutation operator
    
    global contract_fun,psi_decomposed_otherstate,Vp

    if contraction_path=="verify"
        amplitude_otherstate,_=contract_sample(psi_decomposed_otherstate,Lx,Ly,iconf_new,Vp,contract_fun);
        amplitude_otherstate_,_,contract_history_= partial_contract_sample(psi_decomposed_otherstate,iconf_new,Vp,contract_history_otherstate_);
        @assert abs(norm(amplitude_otherstate-amplitude_otherstate_)/amplitude_otherstate)<1e-10;
    elseif contraction_path=="full"
        amplitude_otherstate,_=contract_sample(psi_decomposed_otherstate,Lx,Ly,iconf_new,Vp,contract_fun);
    elseif contraction_path=="recycle"
        amplitude_otherstate,_,contract_history_otherstate_= partial_contract_sample(psi_decomposed_otherstate,iconf_new,Vp,contract_history_otherstate_);
    end


    elocal = amplitude_otherstate/amplitude;

        
    return elocal,contract_history_otherstate_
end



@everywhere function main(dir, worker_id, ntask, BC1,BC2)
    #load saved fPEPS data
    @show Nsteps_worker=Int(round(Nsteps/ntask));
    contraction_path="recycle";#"verify","full","recycle"
 
    filenm="CSL_D"*string(D)*"_U1";
    psi0,Vp,Vv=load_fPEPS_from_iPEPS(Lx,Ly,filenm);

    global contraction_path, contract_fun, psi_decomposed,psi_decomposed_otherstate, Vp, projector_method
    projector_method="1";#"1" or "2"
    contract_fun=contract_whole_torus_boundaryMPS;
    normalize_PEPS!(psi0,Vp,contract_whole_torus_boundaryMPS);#normalize psi0 such that the amplitude of a single config is close to 1
    #psi_00,psi_0pi,psi_pi0,psi_pipi =construct_4_states(psi0,Vv);#four states
    psi_BC_set =construct_4_states(psi0,Vv);#four states

    psi=psi_BC_set[BC1];
    psi_decomposed=decompose_physical_legs(psi,Vp);

    psi_otherstate=psi_BC_set[BC2];
    psi_decomposed_otherstate=decompose_physical_legs(psi_otherstate,Vp);

    
    ##########################################

    coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours(Lx,Ly,"PBC");

    initial_iconf =initial_Neel_config(Lx,Ly,1);
    #Recall that iconf here has elements 1 (up spin) and -1 (down spin), unlike in our C++ code where we have 1 and 2.

    #start from the config in test.csv


    #create empty contract_history
    contract_history=torus_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly));
    contract_history_otherstate=torus_contract_history(zeros(Int8,Lx*Ly),Matrix{TensorMap}(undef,Lx,Ly));


    # Initialize variables
    iconf_new = copy(initial_iconf)

    ebin1 = zeros(Complex{Float64}, binn)

    outputname = dir*"id_"*string(worker_id)*"_BC"*string(BC1)*"_"*string(BC2)*"_"*string(Lx)*"x"*string(Ly)*"_D"*string(D)*"_chi"*string(chi)*".csv"

    if isfile(outputname)
        rm(outputname)
    end

    open(outputname, "a") do file # "a" is for append

        # @inbounds for i in 1:Nsteps  # Number of Monte Carlo steps, usually 1 million
        #     @inbounds for j in 1:Nbra  # Inner loop to create uncorrelated samples
        for i in 1:Nsteps  # Number of Monte Carlo steps, usually 1 million
            # @show i
            # if mod(i,100)==0;@show i;flush(stdout);end


            if contraction_path=="verify"
                amplitude,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,Vp,contract_fun);
                amplitude_,_,contract_history= partial_contract_sample(psi_decomposed,iconf_new,Vp,contract_history);
                @assert abs(norm(amplitude-amplitude_)/amplitude)<1e-10;

                amplitude_otherstate,_=contract_sample(psi_decomposed_otherstate,Lx,Ly,iconf_new,Vp,contract_fun);
                amplitude_otherstate_,_,contract_history= partial_contract_sample(psi_decomposed_otherstate,iconf_new,Vp,contract_history_otherstate);
                @assert abs(norm(amplitude_otherstate-amplitude_otherstate_)/amplitude_otherstate)<1e-10;
            elseif contraction_path=="full"
                amplitude,_=contract_sample(psi_decomposed,Lx,Ly,iconf_new,Vp,contract_fun);
            elseif contraction_path=="recycle"
                amplitude,_,contract_history= partial_contract_sample(psi_decomposed,iconf_new,Vp,contract_history);
                amplitude_otherstate,_,contract_history_otherstate= partial_contract_sample(psi_decomposed_otherstate,iconf_new,Vp,contract_history_otherstate);
            end

            for j in 1:Nbra  # Inner loop to create uncorrelated samples
                randl = rand(1:L)  # Picking a site at random; "l"
                rand2 = rand(1:length(NN_tuple[randl]))  # Picking randomly one of the 4 neighbors
                randK = NN_tuple[randl][rand2]  # Picking a neighbor at random to which electron wants to hop; "K"

                

                if iconf_new[randl] != iconf_new[randK]
                    

                    iconf_new_flip=flip_config(iconf_new,randl,randK);
                    if contraction_path=="verify"
                        # println(iconf_new)
                        # pos1=findall(x->x.==randl,reshape(coord,Lx,Ly));
                        # pos2=findall(x->x.==randK,reshape(coord,Lx,Ly));
                        # @show pos1
                        # @show pos2
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
            
            energyl1,contract_history_otherstate= overlap_ratio(iconf_new,amplitude, contract_history_otherstate)

            rems = mod1(i, binn)  # Binning to store fewer numbers, usually binn is order of 1000
            ebin1[rems] = energyl1

            if rems == binn
                #println( mean(ebin1))
                #CSV.write(outputname, mean(ebin1); append=true) 
                println(file, mean(ebin1));flush(file);
            end

            # Optional: Uncomment to print configuration every 999 steps
            #save a good initial config for next time
            if mod(i + 1, 999) == 0
                # println(outfile, "\n\n", iconf_new, "\n\n\n")
                # println(file, "\n\n", iconf_new, "\n\n\n");flush(stdout);
            end

            # if mod(i + 1, 200) == 0
            println(mean(ebin1));flush(stdout);
            # end

        end

    end

end


BC1=1;
BC2=2;

ntask=nworkers();
main(dir, 1, ntask, BC1,BC2);
# @sync begin
#     for cp=1:ntask
#         worker_id=workers()[cp]
#         @spawnat worker_id main(dir, worker_id, ntask, BC1,BC2);
#     end
# end


data_analysis()
