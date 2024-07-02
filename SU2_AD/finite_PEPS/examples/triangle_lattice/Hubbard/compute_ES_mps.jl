using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
using LineSearches,OptimKit
cd(@__DIR__)

include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\..\\..\\src\\bosonic\\line_search_lib_cell.jl")

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\state\\FinitePEPS.jl")
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
include("..\\..\\..\\optimization\\stochastic_opt.jl")

include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_methods.jl")
include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_simple_update.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")

include("..\\..\\..\\environment\\simple_update\\fermionic\\gauge_fix.jl")
include("..\\..\\..\\environment\\full_update\\full_update_lib.jl")

Random.seed!(666)



global D,chi,multiplet_tol

D=4;
chi=256;
multiplet_tol=1e-5;
init_noise=0;

#filenm="SU_PESS_SU2_D4.jld2";
filenm="stochastic_4x4_D_4_chi_100.jld2";

println("D,chi="*string([D,chi]));
println("init_noise="*string(init_noise));



####################
import LinearAlgebra.BLAS as BLAS
n_cpu=6;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);
Base.Sys.set_process_title("C"*string(n_cpu)*"_stoc_D"*string(D))
pid=getpid();
println("pid="*string(pid));;flush(stdout);
####################

global use_AD;
use_AD=true;



t1=1;
t2=1;
ϕ=pi/2;
μ=0;
U=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);
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


psi=initial_SU2_PESS(filenm,init_noise,true);

Lx,Ly=size(psi);
println("Lx,Ly="*string([Lx,Ly]))


global psi,psi_double

if isa(psi[1,1],Triangle_iPESS)
    psi=PESS_to_PEPS_matrix(psi);
end
psi=normalize_tensor_group(psi);



use_canonical_form=true;

global use_canonical_form
if use_canonical_form
    println("convert to canonical form")
    psi,_=fermiPEPS_gauge_fix_simple(psi,100);
    psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap_new(psi,Lx,Ly);
end









global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;


E=cost_fun_global(psi);
println("E= "*string(E));
################################################


#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D
for cx=1:Lx
    for cy=1:Ly
        psi[cx,cy]=permute(psi[cx,cy],(1,4,5,3,2,));#LUPRD
    end
end


#############################
function combine_mps_2row(mps1,mps2)
    mps1=deepcopy(mps1);#L1 U1 P1 R1 D1
    mps2=deepcopy(mps2);#L2 U2 P2 R2 D2
    
    mps1=permute_neighbour_ind(mps1,3,4,5);#L1 U1 R1 P1 D1

    mps2=permute_neighbour_ind(mps2,1,2,5);#U2 L2 P2 R2 D2
    mps2=permute_neighbour_ind(mps2,2,3,5);#U2 P2 L2 R2 D2

    Up=unitary(fuse(space(mps1,4)*space(mps2,2)), space(mps1,4)*space(mps2,2));
    @tensor mps12[:]:=mps1[-1,-2,-3,1,2]*mps2[2,3,-5,-6,-7]*Up[-4,1,3];#L1 U1 R1 P L2 R2 D2
    mps12=permute_neighbour_ind(mps12,4,5,7);#L1 U1 R1 L2 P R2 D2
    mps12=permute_neighbour_ind(mps12,3,4,7);#L1 U1 L2 R1 P R2 D2
    mps12=permute_neighbour_ind(mps12,2,3,7);#L1 L2 U1 R1 P R2 D2
    mps12=permute_neighbour_ind(mps12,4,5,7);#L1 L2 U1 P R1 R2 D2
    gate=swap_gate(mps12,5,6);
    @tensor mps12[:]:=mps12[-1,-2,-3,-4,1,2,-7]*gate[-5,-6,1,2];#L1 L2 U1 P R2 R1 D2 #note that only apply swap gate, but no permutation
    UL=unitary(fuse(space(mps12,1)*space(mps12,2)),space(mps12,1)*space(mps12,2));
    UR=unitary(space(mps12,5)'*space(mps12,6)',fuse(space(mps12,5)*space(mps12,6)));
    @tensor mps12[:]:=mps12[1,2,-2,-3,3,4,-5]*UL[-1,1,2]*UR[3,4,-4];#L U1 P R D2 
    return mps12,UL,UR,Up
end

####################
function get_4row(A1,A2,A3,A4)
    A12,UL12a,UR12a,Up12a=combine_mps_2row(A1,A2);
    A123,UL123a,UR123a,Up123a=combine_mps_2row(A12,A3);
    A1234,UL1234a,UR1234a,Up1234a=combine_mps_2row(A123,A4);
    A1234=permute_neighbour_ind(A1234,2,3,5);#L P U R D 
    A1234=permute_neighbour_ind(A1234,3,4,5);#L P R U D 
    A1234=permute_neighbour_ind(A1234,4,5,5);#L P R D U 
    @tensor A1234[:]:=A1234[-1,-2,-3,1,1];#L P R
    return A1234
end

A1234=get_4row(psi[1,4],psi[1,3],psi[1,2],psi[1,1]);
B1234=get_4row(psi[2,4],psi[2,3],psi[2,2],psi[2,1]);
C1234=get_4row(psi[3,4],psi[3,3],psi[3,2],psi[3,1]);
D1234=get_4row(psi[4,4],psi[4,3],psi[4,2],psi[4,1]);

A1234=A1234/norm(A1234);
B1234=B1234/norm(B1234);
C1234=C1234/norm(C1234);
D1234=D1234/norm(D1234);
#############################


@tensor vl[:]:=A1234'[2,1,-1]*A1234[2,1,-2];
@tensor vl[:]:=vl[1,3]*B1234'[1,2,-1]*B1234[3,2,-2];

@tensor vr[:]:=D1234'[-1,2,1]*D1234[-2,2,1];
@tensor vr[:]:=vr[1,3]*C1234'[-1,2,1]*C1234[-2,2,3];

@tensor H[:]:=vl[1,-1]*vr[1,-2];

eu,ev=eig(H,(1,),(2,));


function get_ES(v0)
    eu0,ev0=eigen(v0);
    eu0_dense=convert(Array,eu0);
    order=sortperm(abs.(diag(eu0_dense)))


    Spin_set=Vector{Int64}(undef,0);
    eu_set=Vector{Int64}(undef,0);
    for cc=1:length(eu0.data.values)
      spa=eu0.data.keys[cc];

      Spin=spa.j;
      mm=diag(eu0.data.values[cc]);
      Spin_set=vcat(Spin_set,Spin*ones(length(mm)));
      eu_set=vcat(eu_set,mm);

    end

    eu_set=eu_set/sum(abs.(eu_set));
    pos=findall(x-> x.>1e-6, abs.(eu_set));
    Spin_set=Spin_set[pos];
    eu_set=eu_set[pos];

    order=sortperm(abs.(eu_set));
    Spin_set=Spin_set[order];
    eu_set=eu_set[order];
    return Spin_set,eu_set
  end

  Spin_set,eu_set=get_ES(eu);
