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
include("..\\..\\..\\state\\excitation_ansatz.jl")



#########################################
L=20;
Hm=randn(L,L)+randn(L,L)*im;
Hm=Hm+Hm';
Nm=randn(L,L);
Nm=Nm*Nm';


psi_ex_init=initial_excitation_artificial(L);



f1(x)=apply_excitation_H_or_N(Hm, x);
f2(x)=apply_excitation_H_or_N(Nm, x);
alg = GolubYe(; orth=KrylovDefaults.orth, krylovdim=20, maxiter=1, tol=1e-10)
x0=psi_ex_init;
D3, V3, info1 = geneigsolve((f1, f2), x0, 2, :SR, alg)
println(D3)
println(info1)

eu,ev=eigen(pinv(Nm)*Hm);
println(eu[1:5])







# for cx=1:Lx
#     for cy=1:Ly
#         px=cx;
#         py=cy;
#         println("coordinate: "*string([px,py]));
#         A0=psi[px,py];

#         x=psi_double_open[px,py];

#         E_tem1,∂E1=get_grad_double_layer(x,px,py,psi_double_open,U_s_s,"energy");
#         E_tem2,∂E2=get_grad_double_layer(x,px,py,psi_double_open,U_s_s,"norm");

#         ∂E1=∂E1';
#         ∂E2=∂E2';
#         AA_open,U_L,U_D,U_R,U_U=build_double_layer_open_position(psi[px,py],px,py,Lx,Ly,true);
#         if py==Ly
#             if px==1
#                 ∂E1=permute(∂E1,(2,1,3,));
#                 ∂E2=permute(∂E2,(2,1,3,));
#                 AA_open=permute(AA_open,(2,1,3,));
#             elseif 1<px<Lx
#                 ∂E1=permute(∂E1,(1,3,2,4,));
#                 ∂E2=permute(∂E2,(1,3,2,4,));
#                 AA_open=permute(AA_open,(1,3,2,4,));
#             end
#         end

        
#         E_opt_new,T=H_eig_solve(∂E1,∂E2/((Lx-1)*(Ly-1)),AA_open,px,py,U_L,U_D,U_R,U_U);
#         E_opt_new=real(E_opt_new)
#         if E_opt_new<E_opt
#             psi[px,py]=T;
#             psi_double_open, U_s_s=construct_double_layer_open(psi);
#             E_opt=E_opt_new;
#             println("Energy of updated state: "*string(E_opt));flush(stdout);
#         else
#             println("Energy not improved, change to next site")
#         end
#     end
# end






