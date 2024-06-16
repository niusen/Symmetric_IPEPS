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

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\state\\FinitePEPS.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods_new.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk_excitation.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\environment\\Variational\\check_ob.jl")
include("..\\..\\..\\environment\\Variational\\H_environment.jl")
include("..\\..\\..\\environment\\Variational\\oneD_contractions.jl")
include("..\\..\\..\\environment\\Variational\\variational_methods.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\optimization\\PEPS_methods.jl")
include("..\\..\\..\\environment\\full_update\\full_update_lib.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\triangle_terms.jl")

global use_AD;
use_AD=true;

global chi,D
chi=100;
D=3;
filenm="optim_4x4_D_3_chi_100_13.89035.jld2";


J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
global parameters

svd_settings=Svd_settings();
svd_settings.svd_trun_method="chi";#chi" or "tol"
svd_settings.chi_max=500;
svd_settings.tol=1e-5;
dump(svd_settings);
global svd_settings

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);
global backward_settings



global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=0;


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""




init_noise=0;
psi=initial_SU2_state(filenm,init_noise,true);

Lx,Ly=size(psi);
global Lx,Ly


psi=disk_to_torus(psi);
# psi=remove_trivial_boundary_leg(psi);


include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator_dense.jl")
#convert to dense tensor 
psi_dense=Matrix{Any}(undef,Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        psi_dense[ca,cb]=convert_to_dense(psi[ca,cb]);
    end
end
psi=psi_dense;
psi=remove_trivial_boundary_leg(psi);

global U_phy
U_phy=space(psi[2,2],5);



psi_double_open, U_s_s=construct_double_layer_open(psi);
global U_s_s
AA1,_=build_double_layer_bulk_open(psi[2,2], psi[2,2], false);
AA2,_=build_double_layer_bulk_open(psi[2,2],false);


multiplet_tol=1e-5;



global chi,multiplet_tol

global px,py

psi_double=contract_physical_all(psi_double_open, U_s_s);

psi=disk_to_torus(psi);
# psi=remove_trivial_boundary_leg(psi);

#spin operator
sp=[0 1;0 0];
sm=[0 0;1 0];
sz=[1 0;0 -1]/2;
sp=TensorMap(sp,ComplexSpace(2)',ComplexSpace(2)');
sm=TensorMap(sm,ComplexSpace(2)',ComplexSpace(2)');
sz=TensorMap(sz,ComplexSpace(2)',ComplexSpace(2)');

op=sz;

H_eff=zeros(Lx*Ly,Lx*Ly)*im;
N_eff=zeros(Lx*Ly,Lx*Ly)*im;

real_imag="full";
N_operator_sites=1;
for ca=1:(Lx*Ly)
    for cb=1:(Lx*Ly)
        pos_bra=coordinate_1d_to_2d(Lx,Ly,ca);
        pos_ket=coordinate_1d_to_2d(Lx,Ly,cb);
        A_bra=psi[pos_bra[1],pos_bra[2]];
        A_ket=psi[pos_ket[1],pos_ket[2]];
        @tensor A_bra[:]:=A_bra[-1,-2,-3,-4,1]*op[-5,1];
        @tensor A_ket[:]:=A_ket[-1,-2,-3,-4,1]*op[-5,1];

        E0=cost_fun_bra_ket(N_operator_sites,A_bra,A_ket,pos_bra,pos_ket,psi,psi,psi_double_open,psi_double,U_s_s,"energy",real_imag);
        N0=cost_fun_bra_ket(N_operator_sites,A_bra,A_ket,pos_bra,pos_ket,psi,psi,psi_double_open,psi_double,U_s_s,"norm",real_imag);
        N0=N0/((Lx-1)*(Ly-1));

        H_eff[ca,cb]=E0;
        N_eff[ca,cb]=N0;
        # E_opt=E0/N0;
    end
end
R=compute_rotation_matrix_1site(psi,op);

jldsave("excitation_1site.jld2";H_eff,N_eff,R);


Norm=norm(N_eff);
N_eff=N_eff/Norm;
H_eff=H_eff/Norm;

eu,ev=eigen(N_eff/2+N_eff'/2);
norm(ev*diagm(eu)*ev'-N_eff)
# eu,ev=eigen(pinv(N_eff)*H_eff);
sqrt(pinv(diagm(eu)))*ev'*N_eff*ev*sqrt(pinv(diagm(eu)));
Es,Vs=eigen(sqrt(pinv(diagm(eu)))*ev'*(H_eff/2+H_eff'/2)*ev*sqrt(pinv(diagm(eu))));


R=R/Norm;
R_otho=sqrt(pinv(diagm(eu)))*ev'*(R)*ev*sqrt(pinv(diagm(eu)));#otho normal basis
ks=imag(log.(diag(Vs'*R_otho*Vs)))/2/pi;

