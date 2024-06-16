using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\environment\\Variational\\check_ob.jl")
include("..\\..\\..\\environment\\Variational\\H_environment.jl")
include("..\\..\\..\\environment\\Variational\\oneD_contractions.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")



global use_AD;
use_AD=false;

D=3;

J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
global J1,J2,Jchi

#Hamiltonian
H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=6;
Ly=6;

data=load("CSL_D3.jld2");
A=data["A"];
A=A/norm(A);


global U_phy
U_phy=space(A,5);


P=zeros(1,3);P[1,1]=1;
P_L=TensorMap(P,Rep[SU₂](0=>1),space(A,1));
P_D=TensorMap(P,Rep[SU₂](0=>1),space(A,2));

psi=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
for cx=2:Lx-1
    for cy=2:Ly-1
        psi[cx,cy]=A;
    end
end

cx=1;
for cy=2:Ly-1
    @tensor T[:]:=A[1,-2,-3,-4,-5]*P_L[-1,1];
    psi[cx,cy]=T;
end

cx=Lx;
for cy=2:Ly-1
    @tensor T[:]:=A[-1,-2,1,-4,-5]*P_L'[1,-3];
    psi[cx,cy]=T;
end

cy=1;
for cx=2:Lx-1
    @tensor T[:]:=A[-1,1,-3,-4,-5]*P_D[-2,1];
    psi[cx,cy]=T;
end

cy=Ly;
for cx=2:Lx-1
    @tensor T[:]:=A[-1,-2,-3,1,-5]*P_D'[1,-4];
    psi[cx,cy]=T;
end

cx=1;
cy=1;
@tensor T[:]:=A[1,2,-3,-4,-5]*P_L[-1,1]*P_D[-2,2];
psi[cx,cy]=T;

cx=Lx;
cy=1;
@tensor T[:]:=A[-1,2,1,-4,-5]*P_L'[1,-3]*P_D[-2,2];
psi[cx,cy]=T;

cx=1;
cy=Ly;
@tensor T[:]:=A[1,-2,-3,2,-5]*P_L[-1,1]*P_D'[2,-4];
psi[cx,cy]=T;

cx=Lx;
cy=Ly;
@tensor T[:]:=A[-1,-2,1,2,-5]*P_L'[1,-3]*P_D'[2,-4];
psi[cx,cy]=T;





psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

filenm="CSL_D3_L"*string(Lx)*".jld2";
jldsave(filenm;psi);


chi=100;
global multiplet_tol;
multiplet_tol=1e-5;


psi_double=construct_double_layer(psi,psi);

# x_range=[3,4];
# y_range=[1,2];
# psi_double_plaquatte=construct_double_layer_open_plaquatte(psi,x_range,y_range);



mps_set_down_move, trun_errs, norm_coe_down_move, UR_set_down_move, UL_set_down_move, unitarys_R_set_down_move, unitarys_L_set_down_move, projectors_R_set_down_move, projectors_L_set_down_move=get_projector_down_move(psi_double);
data_down_move=Dict("mps_set"=>mps_set_down_move, " trun_errs"=> trun_errs, "norm_coe"=>norm_coe_down_move,"UR_set"=>UR_set_down_move, "UL_set"=>UL_set_down_move,  "unitarys_R_set"=>unitarys_R_set_down_move, "unitarys_L_set"=>unitarys_L_set_down_move, "projectors_R_set"=>projectors_R_set_down_move, "projectors_L_set"=>projectors_L_set_down_move);    

mps_set_up_move, trun_errs, norm_coe_up_move, UR_set_up_move, UL_set_up_move, unitarys_R_set_up_move, unitarys_L_set_up_move, projectors_R_set_up_move, projectors_L_set_up_move=get_projector_up_move(psi_double);
data_up_move=Dict("mps_set"=>mps_set_up_move, " trun_errs"=> trun_errs, "norm_coe"=>norm_coe_up_move,"UR_set"=>UR_set_up_move, "UL_set"=>UL_set_up_move,  "unitarys_R_set"=>unitarys_R_set_up_move, "unitarys_L_set"=>unitarys_L_set_up_move, "projectors_R_set"=>projectors_R_set_up_move, "projectors_L_set"=>projectors_L_set_up_move);    


norm_set=Matrix{TensorMap}(undef,Lx,Ly);
norm_coe_set=Matrix{Number}(undef,Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        term,log_coe=norm_env(psi,psi_double,cx,cy, mps_set_down_move, norm_coe_down_move, mps_set_up_move, norm_coe_up_move);
        norm_set[cx,cy]=term;
        norm_coe_set[cx,cy]=log_coe;
    end
end


datanm="E_set_L"*string(Lx)*"_D_"*string(D)*".jld2";
E_set=load(datanm)["E_set"];


for px=1:Lx
    for py=1:Ly
        println([px,py])
        ee_set=Matrix{Number}(undef,Lx-1,Ly-1);

        for cx=1:Lx-1
            for cy=1:Ly-1
                x_range=[cx,cx+1];
                y_range=[cy,cy+1];
                #println(string(x_range)*", "*string(y_range))
                
                term,term_log_coe=Ham_env_term(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move);
                
                # println(space(term))
                A_double_open=build_double_layer_open_position(psi[px,py],px,py,Lx,Ly);
                if py==Ly
                    if px==1
                        A_double_open=permute(A_double_open,(2,1,3,));
                    elseif 1<px<Lx
                        A_double_open=permute(A_double_open,(1,3,2,4,));
                    end
                end

                Norm=norm_set[px,py];
                Norm_log_coe=norm_coe_set[px,py];
                if Rank(term)==3
                    e=@tensor A_double_open[1,2,3]*term[1,2,3];
                    Norm=@tensor A_double_open[1,2,3]*Norm[1,2,3];
                elseif Rank(term)==4
                    e=@tensor A_double_open[1,2,3,4]*term[1,2,3,4];
                    Norm=@tensor A_double_open[1,2,3,4]*Norm[1,2,3,4];
                elseif Rank(term)==5
                    e=@tensor A_double_open[1,2,3,4,5]*term[1,2,3,4,5];
                    Norm=@tensor A_double_open[1,2,3,4,5]*Norm[1,2,3,4,5];
                end
                ee_set[cx,cy]=e/Norm;
                #println([e/Norm,term_log_coe-Norm_log_coe])

            end
        end



        dE_set=ee_set-E_set;
        println(maximum(abs.(dE_set)))
        println(real.(dE_set))

    end
end



