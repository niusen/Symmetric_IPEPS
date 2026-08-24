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
include("../../../../environment/MC/exact_contract.jl")

@everywhere begin
@show const global_eltype=Float64;#Float64,ComplexF164
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







filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
psi,Vp=load_fPEPS(Lx,Ly,filenm);

# Prefer the exact normalized PEPS saved together with the VMC gradient.  New
# gradient files therefore use one normalization for both VMC and finite
# difference.  Legacy files have no saved `psi`, so normalize their source
# state once as a backward-compatible fallback.
grad_filenm="grad_"*string(Lx)*"x"*string(Ly)*"_D"*string(D)*"_chi"*string(chi)*".jld2";
data=load(grad_filenm);
if haskey(data,"psi")
    psi_ref=deepcopy(data["psi"]);
else
    config_max=normalize_PEPS!(psi,Vp,contract_whole_disk);
    psi_ref=deepcopy(psi);
end
Grad=data["Grad"];

# psi=add_noise(psi,0,true);

E0,grad_FD=exact_grad(psi_ref;dt=1e-5);



#jldsave("grad_FD.jld2";grad_FD)




ov_set=compare_grad(grad_FD,Grad);

@show ov_set
