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
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\environment\\Variational\\check_ob.jl")
include("..\\..\\..\\environment\\Variational\\H_environment.jl")
include("..\\..\\..\\environment\\Variational\\oneD_contractions.jl")
include("..\\..\\..\\environment\\Variational\\variational_methods.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\optimization\\PEPS_methods.jl")

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
global backward_settings

global n_mps_sweep
n_mps_sweep=0;


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""




init_noise=0;
psi=initial_SU2_state(filenm,init_noise,true);

Lx=size(psi,1);
Ly=size(psi,2);
global U_phy
U_phy=space(psi[2,2],5);



psi_double_open, U_s_s=construct_double_layer_open(psi);



multiplet_tol=1e-5;



global chi,multiplet_tol

global px,py
px=1;py=1;
psi_double=contract_physical_all(psi_double_open, U_s_s);
x=psi_double_open[px,py];
E0=cost_fun_double_layer(x,px,py,psi_double_open,psi_double,U_s_s,"energy");
N0=cost_fun_double_layer(x,px,py,psi_double_open,psi_double,U_s_s,"norm");

E_opt=real(E0/(N0/(Lx-1)/(Ly-1)));

#########################################






px=2;
py=2;
println("coordinate: "*string([px,py]));
A0=psi[px,py];

x=psi_double_open[px,py];

E_tem1,∂E1=get_grad_double_layer(x,px,py,psi_double_open,U_s_s,"energy");
E_tem2,∂E2=get_grad_double_layer(x,px,py,psi_double_open,U_s_s,"norm");

∂E1=∂E1';
∂E2=∂E2';
Norm=norm(∂E2);
AA_open,U_L,U_D,U_R,U_U=build_double_layer_open_position(psi[px,py],px,py,Lx,Ly,true);
∂E1=∂E1/Norm;
∂E2=∂E2/Norm;
H_env=∂E1;
Norm_env=∂E2;

U_s_s=U_s_s';

U_U_d=unitary(fuse(space(U_U,2)*space(U_s_s,3)), space(U_U,2)*space(U_s_s,3));
@tensor H_env[:]:=H_env[-1,-2,-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
U_L_D=unitary(fuse(space(U_L,3)*space(U_D,3)), space(U_L,3)*space(U_D,3));
@tensor H_env[:]:=H_env[5,6,-3,-4,-5]*U_L[5,1,2]*U_D[6,3,4]*U_L_D'[1,3,-1]*U_L_D[-2,2,4];
@tensor H_env[:]:=H_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
H_env=permute(H_env,(1,3,5,),(2,4,6,));

@tensor Norm_env[:]:=Norm_env[-1,-2,-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
@tensor Norm_env[:]:=Norm_env[5,6,-3,-4,-5]*U_L[5,1,2]*U_D[6,3,4]*U_L_D'[1,3,-1]*U_L_D[-2,2,4];
@tensor Norm_env[:]:=Norm_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
Norm_env=permute(Norm_env,(1,3,5,),(2,4,6,));



function apply_H(x,H_env)
    println("Mx")
    @tensor x_new[:]:=H_env[-1,-2,-3,1,2,3]*x[1,2,3];
    return x_new
end
function apply_N(x,Norm_env)
    println("Mx")
    @tensor x_new[:]:=Norm_env[-1,-2,-3,1,2,3]*x[1,2,3];
    return x_new
end

eu,ev=eigen(Norm_env/2+Norm_env'/2);
Es,Vs=eigen(sqrt(pinv(eu))*ev'*(H_env/2+H_env'/2)*ev*sqrt(pinv(eu)));


x0=TensorMap(randn,space(H_env,4)'*space(H_env,5)',space(H_env,6));
x0=permute(x0,(1,2,3,))

f1(x)=apply_H(x,H_env);
f2(x)=apply_H(x,Norm_env);



alg = GolubYe(; orth=KrylovDefaults.orth, krylovdim=10, maxiter=10, tol=1e-10, verbosity=1)
D1, V1, info =geneigsolve((f1, f2), x0, 2, :SR; orth=KrylovDefaults.orth, krylovdim=10, maxiter=10, tol=1e-10, ishermitian=true, isposdef=true, verbosity=2)
D2, V2, info =geneigsolve((f1, f2), x0, 2, :SR, alg)



alg = GolubYe(; orth=KrylovDefaults.orth, krylovdim=2, maxiter=1, tol=1e-10)
D3, V3, info1 = geneigsolve((f1, f2), x0, 2, :SR, alg)


        # if py==Ly
        #     if px==1
        #         ∂E1=permute(∂E1,(2,1,3,));
        #         ∂E2=permute(∂E2,(2,1,3,));
        #         AA_open=permute(AA_open,(2,1,3,));
        #     elseif 1<px<Lx
        #         ∂E1=permute(∂E1,(1,3,2,4,));
        #         ∂E2=permute(∂E2,(1,3,2,4,));
        #         AA_open=permute(AA_open,(1,3,2,4,));
        #     end
        # end

        
        # E_opt_new,T=H_eig_solve(∂E1,∂E2/((Lx-1)*(Ly-1)),AA_open,px,py,U_L,U_D,U_R,U_U);
        # E_opt_new=real(E_opt_new)
        # if E_opt_new<E_opt
        #     psi[px,py]=T;
        #     psi_double_open, U_s_s=construct_double_layer_open(psi);
        #     E_opt=E_opt_new;
        #     println("Energy of updated state: "*string(E_opt));flush(stdout);
        # else
        #     println("Energy not improved, change to next site")
        # end







