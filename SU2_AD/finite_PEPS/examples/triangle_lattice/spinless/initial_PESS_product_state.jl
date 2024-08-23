using LinearAlgebra:diag,I,diagm 
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\fermion\\peps_double_layer_methods_fermion.jl")
include("..\\..\\..\\environment\\AD\\fermion\\fermi_CTM_observables.jl")
include("..\\..\\..\\environment\\AD\\fermion\\fermi_contract.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\models\\Hubbard\\triangle_lattice\\Hofstadter_N2.jl")

include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_methods.jl")
include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_simple_update.jl")

Dmax=4;


####################
import LinearAlgebra.BLAS as BLAS
n_cpu=6;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);
Base.Sys.set_process_title("C"*string(n_cpu)*"_PESS_D"*string(Dmax))
pid=getpid();
println("pid="*string(pid));;flush(stdout);
####################

global use_AD;
use_AD=false;

t1=1;
t2=1;
ϕ=pi/2;
μ=0;
V=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("V",  V)]);
global parameters


svd_settings=Svd_settings();
svd_settings.svd_trun_method="chi";#chi" or "tol"
svd_settings.chi_max=500;
svd_settings.tol=1e-5;
dump(svd_settings);

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);
global svd_settings, backward_settings



#Hamiltonian
# H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=8;
Ly=8;

nu=1/2;
N=Int(Lx*Ly*nu);

initial_occu=vcat(ones(N),zeros(Lx*Ly-N));
initial_occu=reshape(initial_occu,Lx,Ly);


#create initial U(1) PESS according to particle number
Qn_set=Array{Int64}(undef,Lx,Ly,5);
for cx=1:Lx
    for cy=1:Ly
        Qn_set[cx,cy,5]=initial_occu[cx,cy];
    end
end

cx=1;
for cy=1:Ly
    Qn_set[cx,cy,1]=0;#left boundary zero charge
end

cy=1;
for cx=1:Lx
    Qn_set[cx,cy,2]=0;#bot boundary zero charge
end

cy=Ly;
for cx=1:Lx
    Qn_set[cx,cy,4]=0;#top boundary zero charge
end

for cx=1:Lx
    for cy=1:Ly-1
        Qn_set[cx,cy,4]=0;#virtical leg zero charge
        Qn_set[cx,cy+1,2]=0;#virtical leg zero charge
    end
end

for cy=1:Ly
    for cx=1:Lx
        Qn_set[cx,cy,3]=Qn_set[cx,cy,1]+Qn_set[cx,cy,4]+Qn_set[cx,cy,5]-Qn_set[cx,cy,2];
        if cx<Lx
            Qn_set[cx+1,cy,1]=Qn_set[cx,cy,3];
        end
    end
end

psi=Matrix{TensorMap}(undef,Lx,Ly);
B_set=Matrix{TensorMap}(undef,Lx,Ly);
T_set=Matrix{TensorMap}(undef,Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        Qs=Qn_set[cx,cy,:];
        A=TensorMap(randn,Rep[U₁](Qs[1]=>1)*Rep[U₁](Qs[4]=>1)*Rep[U₁](Qs[5]=>1),Rep[U₁](Qs[3]=>1)*Rep[U₁](Qs[2]=>1));#LUdRD

        P=TensorMap(randn,Rep[U₁](0=>1,1=>1),Rep[U₁](Qs[5]=>1));
        @tensor A[:]:=A[-1,-2,1,-4,-5]*P[-3,1];
        A=A/norm(A);
        
        u,s,v=tsvd(permute(A,(1,2,),(3,4,5,)));
        Tm=u*s;#|LU><M|
        Bm=v;#|Md><|RD
        B_set[cx,cy]=Tm;
        T_set[cx,cy]=Bm;


        A=permute(A,(1,5,4,2,3,));
        
        psi[cx,cy]=A;
    end
end


filenm="product_state_Lx"*string(Lx)*"_Ly"*string(Ly)*"_N"*string(N)*".jld2"
jldsave(filenm;B_set,T_set);
