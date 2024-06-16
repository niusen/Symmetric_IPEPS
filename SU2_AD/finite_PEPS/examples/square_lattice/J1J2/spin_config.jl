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
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
include("..\\..\\..\\environment\\extend_bond\\extend_bond.jl")
include("..\\..\\..\\environment\\extend_bond\\environment_2site.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator_dense.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\optimization\\PEPS_methods.jl")
include("..\\..\\..\\optimization\\LineSearches\\My_Backtracking.jl")


Random.seed!(888)
global use_AD;
use_AD=false;

global chi
chi=100;

filenm="WYLiu_D2.jld2";


J1=1;
J2=0;
Jchi=0;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
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




"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""





data=load(filenm);
psi=data["psi"];
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

global Lx,Ly
Lx=size(psi,1);
Ly=size(psi,2);

psi_double=construct_double_layer(psi,psi);


global E_history
E_history=[10000];





multiplet_tol=1e-5;



global chi,multiplet_tol



global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;


E_opt=real(cost_fun_global(psi));
println(E_opt)



E_total,E_set=energy_disk_(psi)



#########################################
@tensor mps_top[:]:=psi[1,4][-5,1,-1]*psi[2,4][1,-6,2,-2]*psi[3,4][2,-7,3,-3]*psi[4,4][3,-8,-4];
@tensor mps_top[:]:=mps_top[-1,-2,-3,-4,1,3,5,7]*psi[1,3][-9,2,1,-5]*psi[2,3][2,-10,4,3,-6]*psi[3,3][4,-11,6,5,-7]*psi[4,3][6,-12,7,-8];


@tensor mps_bot[:]:=psi[1,1][1,-1,-5]*psi[2,1][1,2,-2,-6]*psi[3,1][2,3,-3,-7]*psi[4,1][3,-4,-8];
@tensor mps_bot[:]:=mps_bot[1,3,5,7,-9,-10,-11,-12]*psi[1,2][1,2,-1,-5]*psi[2,2][2,3,4,-2,-6]*psi[3,2][4,5,6,-3,-7]*psi[4,2][6,7,-4,-8];

@tensor psi_total[:]:=mps_top[-1,-2,-3,-4,-5,-6,-7,-8,1,2,3,4]*mps_bot[1,2,3,4,-9,-10,-11,-12,-13,-14,-15,-16];
##############################################

psi_projected=deepcopy(psi_total);
for c1=1:2
    for c2=1:2
        for c3=1:2
            for c4=1:2
                for c5=1:2
                    for c6=1:2
                        for c7=1:2
                            for c8=1:2
                                for c9=1:2
                                    for c10=1:2
                                        for c11=1:2
                                            for c12=1:2
                                                for c13=1:2
                                                    for c14=1:2
                                                        for c15=1:2
                                                            for c16=1:2
                                                                if c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11+c12+c13+c14+c15+c16==(1+2)*8
                                                                else
                                                                    psi_projected[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16]=0
                                                                end

                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


@tensor rho_12[:]:=psi_total'[-1,-2,1,2,3,4,5,6,7,8,9,10,11,12,13,14]*psi_total[-3,-4,1,2,3,4,5,6,7,8,9,10,11,12,13,14];
Norm=@tensor rho_12[1,2,1,2];
e0=@tensor rho_12[1,2,3,4]*H_Heisenberg[1,2,3,4];
e0/Norm

@tensor rho_23[:]:=psi_total'[1,-1,-2,2,3,4,5,6,7,8,9,10,11,12,13,14]*psi_total[1,-3,-4,2,3,4,5,6,7,8,9,10,11,12,13,14];
Norm=@tensor rho_23[1,2,1,2];
e0=@tensor rho_23[1,2,3,4]*H_Heisenberg[1,2,3,4];
e0/Norm

@tensor rho_67[:]:=psi_total'[1,2,3,4,5,-1,-2,6,7,8,9,10,11,12,13,14]*psi_total[1,2,3,4,5,-3,-4,6,7,8,9,10,11,12,13,14];
Norm=@tensor rho_67[1,2,1,2];
e0=@tensor rho_67[1,2,3,4]*H_Heisenberg[1,2,3,4];
e0/Norm

##############

@tensor rho_12[:]:=psi_projected'[-1,-2,1,2,3,4,5,6,7,8,9,10,11,12,13,14]*psi_projected[-3,-4,1,2,3,4,5,6,7,8,9,10,11,12,13,14];
Norm=@tensor rho_12[1,2,1,2];
e0=@tensor rho_12[1,2,3,4]*H_Heisenberg[1,2,3,4];
e0/Norm

@tensor rho_23[:]:=psi_projected'[1,-1,-2,2,3,4,5,6,7,8,9,10,11,12,13,14]*psi_projected[1,-3,-4,2,3,4,5,6,7,8,9,10,11,12,13,14];
Norm=@tensor rho_23[1,2,1,2];
e0=@tensor rho_23[1,2,3,4]*H_Heisenberg[1,2,3,4];
e0/Norm

@tensor rho_67[:]:=psi_projected'[1,2,3,4,5,-1,-2,6,7,8,9,10,11,12,13,14]*psi_projected[1,2,3,4,5,-3,-4,6,7,8,9,10,11,12,13,14];
Norm=@tensor rho_67[1,2,1,2];
e0=@tensor rho_67[1,2,3,4]*H_Heisenberg[1,2,3,4];
e0/Norm
