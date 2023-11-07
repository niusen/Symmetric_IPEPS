using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using LinearAlgebra, OptimKit
#using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using LineSearches
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)

include("src\\iPEPS_ansatz.jl")
include("src\\kagome_CTMRG.jl")
include("src\\kagome_CTMRG_unitcell.jl")
include("src\\kagome_model.jl")
include("src\\kagome_model_cell.jl")
include("src\\kagome_AD_SU2_cell.jl")
include("src\\Settings.jl")
include("src\\Settings_cell.jl")
include("src\\AD_lib.jl")
include("src\\line_search_lib.jl")
include("src\\line_search_lib_cell.jl")
include("src\\optimkit_lib.jl")




Random.seed!(555)
global Lx,Ly #unitcell of ansatz
Lx=2;
Ly=2

D=6;
chi=40;


theta=0*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);


grad_ctm_setting=grad_CTMRG_settings();
grad_ctm_setting.CTM_conv_tol=1e-6;
grad_ctm_setting.CTM_ite_nums=50;
grad_ctm_setting.CTM_trun_tol=1e-8;
grad_ctm_setting.svd_lanczos_tol=1e-8;
grad_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
grad_ctm_setting.conv_check="singular_value";
grad_ctm_setting.CTM_ite_info=true;
grad_ctm_setting.CTM_conv_info=true;
grad_ctm_setting.CTM_trun_svd=false;
grad_ctm_setting.construct_double_layer=true;
grad_ctm_setting.grad_checkpoint=true;
dump(grad_ctm_setting);

LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=50;
LS_ctm_setting.CTM_trun_tol=1e-8;
LS_ctm_setting.svd_lanczos_tol=1e-8;
LS_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
LS_ctm_setting.conv_check="singular_value";
LS_ctm_setting.CTM_ite_info=false;
LS_ctm_setting.CTM_conv_info=true;
LS_ctm_setting.CTM_trun_svd=false;
LS_ctm_setting.construct_double_layer=true;
LS_ctm_setting.grad_checkpoint=true;
dump(LS_ctm_setting);

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);

optim_setting=Optim_settings();
optim_setting.init_statenm="LS_D_6_chi_40.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0.1;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Energy_settings()
energy_setting.kagome_method ="E_triangle";#"E_single_triangle", "E_triangle"
energy_setting.E_up_method = "2x2";#"1x1", "2x2"
energy_setting.E_dn_method = "simplified";#"open_leg", "simplfied"
energy_setting.cal_chiral_order = false;
dump(energy_setting);





global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings






global Vv

if D==6
    Vv=SU2Space(0=>1,1/2=>1,1=>1);
elseif D==8
    Vv=SU2Space(0=>1,1/2=>2,1=>1);
elseif D==12
    Vv=SU2Space(0=>1,1/2=>2,1=>1,3/2=>1);
elseif D==13
    Vv=SU2Space(0=>2,1/2=>2,1=>1,3/2=>1);    
elseif D==16
    Vv=SU2Space(0=>2,1/2=>2,1=>2,3/2=>1);
elseif D==18
    Vv=SU2Space(0=>2,1/2=>3,1=>2,3/2=>1);
elseif Vv==23
    Vv=SU2Space(0=>2,1/2=>3,1=>2,3/2=>1,2=>1);
end
@assert dim(Vv)==D;

global starting_time
starting_time=now();


#E_tem,∂E,CTM_tem=get_grad((triangle_tensor,triangle_tensor,bond_tensor,bond_tensor,bond_tensor));
#run_FiniteDiff(parameters, Vv, chi, LS_ctm_setting, optim_setting, energy_setting)

#fun(state_vec)
# global E_tem, CTM_tem
# E,∂E,CTM_tem=get_grad(state_vec);
# println(E,∂E)


# E0, grad=FD(state_vec)
# println(grad)
# println(∂E./grad)


state_vec=initial_SU2_state(Vv, optim_setting.init_statenm, optim_setting.init_noise)
state_vec=normalize_tensor_group(state_vec);

# E0_, grad,CTM_tem=get_grad(state_vec);
# include("src\\kagome_AD_SU2.jl")
# E0,grad_=FD(state_vec)

global E_history
E_history=[10000];




# ls = BackTracking(order=3)
# println(ls)
# fx_bt3, x_bt3, iter_bt3 = gdoptimize(f, g!, fg!, state_vec, ls)

# ls = StrongWolfe()
# println(ls)
# fx_sw, x_sw, iter_sw = gdoptimize(f, g!, fg!, state_vec, ls)

# ls = LineSearches.HagerZhang()
# println(ls)
# fx_hz, x_hz, iter_hz = gdoptimize(f, g!, fg!, state_vec, ls)

# ls = MoreThuente()
# println(ls)
# fx_mt, x_mt, iter_mt = gdoptimize(f, g!, fg!, state_vec, ls)


# #optimize with OptimKit
# optimkit_op(state_vec)


x=initial_SU2_state(Vv,optim_setting.init_statenm,0);

# modification of bonds should be between B and T tensors.
Vp=SU2Space(0=>1,1/2=>1,1=>1);
U=TensorMap(randn,Vp,Vv);
U,S,V=tsvd(U);
#coordinate of triangle tensor
# cx=1;
# cy=1;
for cx=1:2
    for cy=1:2
        Triangle="down";#"up","down"
        B_ind=1;#1,2,3

        


        if Triangle=="up"
            if B_ind==1
                x_tem=deepcopy(x);
                iPESS_a=x_tem[cx,cy];
                Tup=iPESS_a.Tup;
                B1=iPESS_a.B1;
                @tensor B1[:]:=B1[-1,1,-3]*U[-2,1];
                @tensor Tup[:]:=Tup[1,-2,-3]*U'[1,-1];
                iPESS_a=Kagome_iPESS(B1,iPESS_a.B2,iPESS_a.B3,Tup,iPESS_a.Tdn);
                x_tem[cx,cy]=iPESS_a;

                cost_fun(x_tem)
            elseif B_ind==2
                x_tem=deepcopy(x);
                iPESS_a=x_tem[cx,cy];
                Tup=iPESS_a.Tup;
                B2=iPESS_a.B2;
                @tensor B2[:]:=B2[-1,1,-3]*U[-2,1];
                @tensor Tup[:]:=Tup[-1,1,-3]*U'[1,-2];
                iPESS_a=Kagome_iPESS(iPESS_a.B1,B2,iPESS_a.B3,Tup,iPESS_a.Tdn);
                x_tem[cx,cy]=iPESS_a;

                cost_fun(x_tem)

            elseif B_ind==3
                x_tem=deepcopy(x);
                iPESS_a=x_tem[cx,cy];
                Tup=iPESS_a.Tup;
                B3=iPESS_a.B3;
                @tensor B3[:]:=B3[-1,1,-3]*U[-2,1];
                @tensor Tup[:]:=Tup[-1,-2,1]*U'[1,-3];
                iPESS_a=Kagome_iPESS(iPESS_a.B1,iPESS_a.B2,B3,Tup,iPESS_a.Tdn);
                x_tem[cx,cy]=iPESS_a;

                cost_fun(x_tem)

            end
        elseif Triangle=="down"
            if B_ind==1
                x_tem=deepcopy(x);
                iPESS_a=x_tem[cx,cy];
                iPESS_b=x_tem[mod1(cx+1,2),mod1(cy,2)];
                Tdn=iPESS_a.Tdn;
                B1=iPESS_b.B1;
                @tensor B1[:]:=B1[1,-2,-3]*U[-1,1];
                @tensor Tdn[:]:=Tdn[1,-2,-3]*U'[1,-1];
                iPESS_a=Kagome_iPESS(iPESS_a.B1,iPESS_a.B2,iPESS_a.B3,iPESS_a.Tup,Tdn);
                iPESS_b=Kagome_iPESS(B1,iPESS_b.B2,iPESS_b.B3,iPESS_b.Tup,iPESS_b.Tdn);
                x_tem[cx,cy]=iPESS_a;
                x_tem[mod1(cx+1,2),mod1(cy,2)]=iPESS_b;

                cost_fun(x_tem)
            elseif B_ind==2
                x_tem=deepcopy(x);
                iPESS_a=x_tem[cx,cy];
                Tdn=iPESS_a.Tdn;
                B2=iPESS_a.B2;
                @tensor B2[:]:=B2[1,-2,-3]*U[-1,1];
                @tensor Tdn[:]:=Tdn[-1,1,-3]*U'[1,-2];
                iPESS_a=Kagome_iPESS(iPESS_a.B1,B2,iPESS_a.B3,iPESS_a.Tup,Tdn);
                x_tem[cx,cy]=iPESS_a;


                cost_fun(x_tem)

            elseif B_ind==3
                x_tem=deepcopy(x);
                iPESS_a=x_tem[cx,cy];
                iPESS_b=x_tem[mod1(cx,2),mod1(cy+1,2)];
                Tdn=iPESS_a.Tdn;
                B3=iPESS_b.B3;
                @tensor B3[:]:=B3[1,-2,-3]*U[-1,1];
                @tensor Tdn[:]:=Tdn[-1,-2,1]*U'[1,-3];
                iPESS_a=Kagome_iPESS(iPESS_a.B1,iPESS_a.B2,iPESS_a.B3,iPESS_a.Tup,Tdn);
                iPESS_b=Kagome_iPESS(iPESS_b.B1,iPESS_b.B2,B3,iPESS_b.Tup,iPESS_b.Tdn);
                x_tem[cx,cy]=iPESS_a;
                x_tem[mod1(cx,2),mod1(cy+1,2)]=iPESS_b;

                cost_fun(x_tem)

            end
        end
    end
end





