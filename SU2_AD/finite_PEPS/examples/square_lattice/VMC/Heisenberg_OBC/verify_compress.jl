using Distributed
#number of workers to add and soft restrict of memory
#addprocs(50; exeflags=["--heap-size-hint=6G"])
#addprocs(2; exeflags=["--heap-size-hint=6G"])

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
@show const Lx = 8      # number of sites along x / number of columns in the lattice
@show const Ly = 8      # number of sites along y / number of rows in the lattice
@show const D=6;#bond dimension of state
@show const chi=6;#bond dimension of environment
@show const use_mps_sweep=true;
@show const n_mps_sweep=5;

const L = Lx*Ly # total number of lattice sites
const Nbra = L             # Inner loop size, to generate uncorrelated samples, usually must be of size O(L).
const Ne = L            # Number of electrons on the lattice (for spin models this will always be equal to L)
@show const Nsteps = 400       # Total Monte Carlo steps
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





contraction_path="recycle";#"verify","full","recycle"

filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
psi,Vp=load_fPEPS(Lx,Ly,filenm);
@show Vp
global contraction_path, psi_decomposed, Vp
contract_fun=contract_whole_disk;
#config_max=normalize_PEPS!(psi,Vp,contract_whole_disk);#normalize psi such that the amplitude of a single config is close to 1
config_max =initial_Neel_config_square(Lx,Ly,1);
psi_decomposed=decompose_physical_legs(psi,Vp);

sample=Matrix{TensorMap}(undef,Lx,Ly);
##########################################

coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours_square(Lx,Ly,"OBC");

initial_iconf =initial_Neel_config_square(Lx,Ly,1);
#Recall that iconf here has elements 1 (up spin) and -1 (down spin), unlike in our C++ code where we have 1 and 2.

#start from the config in test.csv


sample=pick_sample(psi_decomposed,config_max, sample);

@show Norm,trun_err=contract_whole_disk(sample,chi);






mpo_mps_fun=simple_truncate_to_middle;


mps_bot=sample[:,1];
cy=2;
mpo=sample[:,cy];
mps_approx,trun_errs,norm_coe=mpo_mps_fun(mpo, mps_bot,chi);
mps_approx[1]=mps_approx[1]*norm_coe;


mps_exact,_=apply_mpo(mpo,mps_bot);

@show sqrt(overlap_mps(mps_exact,mps_exact))
@show sqrt(overlap_mps(mps_approx,mps_approx))





@show mps_diff(mps_exact,mps_approx)



@show overlap_mps(mps_approx,mps_exact)/sqrt(overlap_mps(mps_approx,mps_approx)*overlap_mps(mps_exact,mps_exact))
