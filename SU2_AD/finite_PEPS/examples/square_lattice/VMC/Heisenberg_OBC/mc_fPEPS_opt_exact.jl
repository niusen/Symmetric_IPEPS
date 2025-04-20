using Distributed
using Dates
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
@everywhere include("../../../../environment/MC/MC_stochastic_opt.jl")

@everywhere begin
@show const global_eltype=Float64;#Float64,ComplexF164
@show const Lattice="square";#"kagome", "square"
@show const Lx = 4      # number of sites along x / number of columns in the lattice
@show const Ly = 4      # number of sites along y / number of rows in the lattice
@show const D=2;#bond dimension of state
@show const chi=10;#bond dimension of environment
@show const use_mps_sweep=false;
@show const n_mps_sweep=0;

const L = Lx*Ly # total number of lattice sites
const Nbra = L             # Inner loop size, to generate uncorrelated samples, usually must be of size O(L).
const Ne = L            # Number of electrons on the lattice (for spin models this will always be equal to L)
@show const Nsteps = 200       # Total Monte Carlo steps
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








function exact_stochastic_opt(x0::Matrix{TensorMap}, ls) 

    println("stochastic optimization");
    
    global save_filenm
    global contract_fun, Vp 
    ntask=nworkers();
    x = deepcopy(x0);
    x_min=deepcopy(x);
    E_min=10000;
    E_set=Vector{global_eltype}(undef,0);
    delta=ls.delta0;

    

    gvec = similar(x);
    gnorm=10000;
    iter = 0
    while iter < ls.maxiter && gnorm > ls.gtol
        println("optim iteration "*string(iter));flush(stdout);


        #@time main(dir, 1, ntask)
         

        E_mean, gvec=exact_grad(x);
        gnorm = norm(gvec);
        @show E_mean;flush(stdout);
        println("norm of grad:"*string(norm(gvec)))

        # gvec=get_grad_conjugate(gvec);
        if real(E_mean)<E_min

            global starting_time
            Now=now();
            Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
            println("Time consumed: "*string(Time));flush(stdout);



            E_min=real(E_mean);
            x_min=x;
            push!(E_set,E_min);
            jldsave(save_filenm; E_set, E_mean, psi=x);


            x_norm=norm(x);
            x_updated=x-x_norm*get_random_grad(gvec,delta);#get random grad
            println("norm of random grad:"*string(norm(x_updated-x)));flush(stdout);
    
            x=to_Matrix_TensorMap(x_updated);
            iter += 1
    
        else
            @show delta=delta*(ls.alpha);
            x=x_min;



            x_norm=norm(x);
            x_updated=x-x_norm*get_random_grad(gvec,delta);#get random grad
            println("norm of random grad:"*string(norm(x_updated-x)));flush(stdout);
    
            x=to_Matrix_TensorMap(x_updated);

        end
        
        
    end
    return x
end





filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
psi,Vp=load_fPEPS(Lx,Ly,filenm);
@show Vp
@everywhere global contract_fun, Vp 
@everywhere contract_fun=contract_whole_disk;

Noise=0.3;
psi=add_noise(psi,Noise);

global save_filenm,starting_time
starting_time=now();
save_filenm="stochastic_"*string(Lx)*"x"*string(Ly)*"_D"*string(D)*"_chi"*string(chi)*".jld2"

ls=LineSearch();
ls.maxiter=100;
ls.gtol=1e-3;
ls.delta0=1e-3;
ls.alpha=3/4;
@show ls;




exact_stochastic_opt(psi, ls) 