using Revise, TensorKit
using LinearAlgebra:diag,I,diagm 
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
using Random
using LineSearches,OptimKit
using Dates
cd(@__DIR__)

include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_correl.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD.jl")

Random.seed!(777)

M=1;
chi=40

t=1;
ϕ=pi/2;
μ=0;
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ)]);



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
optim_setting.init_statenm="nothing";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "spinless_triangle_lattice";
dump(energy_setting);



global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings






################################################
if M==1
    data=load("swap_gate_Tensor_M1.jld2")

    A=data["A"];   #P1,P2,L,R,D,U

    A_new=zeros(1,2,2,2,2,2,2)*im;
    A_new[1,:,:,:,:,:,:]=A;
    Vdummy=ℂ[U1Irrep](-3=>1);
    V=ℂ[U1Irrep](0=>1,1=>1);
    # Vdummy=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((-3,1)=>1);
    # V=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((0,0)=>1,(1,1)=>1);
    A_new = TensorMap(A_new, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V');

    @assert norm(convert(Array,A_new)[1,:,:,:,:,:,:]-A)/norm(A)<1e-14
    A=A_new; # dummy,P1,P2,L,R,D,U


    U_phy1=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,3)), space(A,1)⊗space(A,2)⊗space(A,3));
    @tensor A[:]:=A[1,2,3,-2,-3,-4,-5]*U_phy1[-1,1,2,3]; # P,L,R,D,U

    #Add bond:both parity gate and bond operator
    bond=zeros(1,2,2); bond[1,1,2]=1;bond[1,2,1]=1; bond=TensorMap(bond, ℂ[U1Irrep](1=>1) ← V ⊗ V);
    gate=parity_gate(A,3); @tensor A[:]:=A[-1,-2,1,-4,-5]*gate[-3,1];
    @tensor A[:]:=A[-1,-2,1,2,-5]*bond[-6,-3,1]*bond[-7,-4,2];
    U_phy2=unitary(fuse(space(A,1)⊗space(A,6)⊗space(A,7)), space(A,1)⊗space(A,6)⊗space(A,7));
    @tensor A[:]:=A[1,-2,-3,-4,-5,2,3]*U_phy2[-1,1,2,3];
    #P,L,R,D,U





    gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2];           
    A=permute(A,(1,2,3,5,4,));#P,L,R,U,D

    gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; 
    A=permute(A,(1,2,4,3,5,));#P,L,U,R,D

    gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; 
    A=permute(A,(2,1,3,4,5,));#L,P,U,R,D

    gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; 
    A=permute(A,(1,3,2,4,5,));#L,U,P,R,D

    #convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


    #convert to the order of PEPS code
    A=permute(A,(1,5,4,2,3,));



elseif M==2
    data=load("swap_gate_Tensor_M2.jld2")

    A=data["A"];   #P1,P2,L,R,D,U

    A_new=zeros(1,2,2,2,2,2,2,2,2,2,2)*im;
    A_new[1,:,:,:,:,:,:,:,:,:,:]=A;
    Vdummy=ℂ[U1Irrep](-5=>1);
    V=ℂ[U1Irrep](0=>1,1=>1);
    # Vdummy=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((-3,1)=>1);
    # V=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((0,0)=>1,(1,1)=>1);
    #A_new = TensorMap(A_new, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V'⊗ V'⊗ V'⊗ V');
    A_new = TensorMap(A_new, Vdummy' ⊗ V' ⊗ V' ⊗ V' ⊗ V' ⊗ V' ⊗ V' ← V⊗ V⊗ V⊗ V);


    @assert norm(convert(Array,A_new)[1,:,:,:,:,:,:,:,:,:,:]-A)/norm(A)<1e-14
    A=A_new; # dummy,P1,P2,L,R,D,U


    U_phy1=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,3)), space(A,1)⊗space(A,2)⊗space(A,3));
    @tensor A[:]:=A[1,2,3,-2,-3,-4,-5,-6,-7,-8,-9]*U_phy1[-1,1,2,3]; # P,L,R,D,U


    #Add bond:both parity gate and bond operator
    bond=zeros(1,2,2); bond[1,1,2]=1;bond[1,2,1]=1; bond=TensorMap(bond, ℂ[U1Irrep](1=>1)' ← V' ⊗ V');
    gate=parity_gate(A,4); @tensor A[:]:=A[-1,-2,-3,1,-5,-6,-7,-8,-9]*gate[-4,1];
    gate=parity_gate(A,6); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,-7,-8,-9]*gate[-6,1];
    @tensor total_bond[:]:=bond[-1,-5,-9]*bond[-2,-6,-10]*bond[-3,-7,-11]*bond[-4,-8,-12];
    @tensor A[:]:=A[-1,-2,-3,1,2,3,4,-8,-9]*total_bond[-10,-11,-12,-13,-4,-5,-6,-7,1,2,3,4];
    U_phy2=unitary(fuse(space(A,1)⊗space(A,10)⊗space(A,11)⊗space(A,12)⊗space(A,13)), space(A,1)⊗space(A,10)⊗space(A,11)⊗space(A,12)⊗space(A,13));
    @tensor A[:]:=A[1,-2,-3,-4,-5,-6,-7,-8,-9,2,3,4,5]*U_phy2[-1,1,2,3,4,5];
    #P,L,R,D,U


    ###################
    #|><R1R2|=|><|R2R1
    gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2,-6,-7,-8,-9]*gate[-4,-5,1,2];  
    gate=swap_gate(A,6,7); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,2,-8,-9]*gate[-6,-7,1,2];  
    ###################


    #group virtual legs on the same legs
    U1=unitary(fuse(space(A,2)⊗space(A,3)),space(A,2)⊗space(A,3)); 
    U2=unitary(fuse(space(A,8)⊗space(A,9)),space(A,8)⊗space(A,9));

    @tensor A[:]:=A[-1,1,2,-3,-4,-5,-6,-7,-8]*U1[-2,1,2];
    @tensor A[:]:=A[-1,-2,1,2,-4,-5,-6,-7]*U2'[1,2,-3];
    @tensor A[:]:=A[-1,-2,-3,1,2,-5,-6]*U1'[1,2,-4];
    @tensor A[:]:=A[-1,-2,-3,-4,1,2]*U2[-5,1,2];


    gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2];           
    A=permute(A,(1,2,3,5,4,));#P,L,R,U,D

    gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; 
    A=permute(A,(1,2,4,3,5,));#P,L,U,R,D

    gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; 
    A=permute(A,(2,1,3,4,5,));#L,P,U,R,D

    gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; 
    A=permute(A,(1,3,2,4,5,));#L,U,P,R,D

    #convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


    #convert to the order of PEPS code
    A=permute(A,(1,5,4,2,3,));


end

################################################



init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM, AA, U_L,U_D,U_R,U_U,ite_num,ite_err=fermi_CTMRG(A,chi,init,[],grad_ctm_setting);


Ident, NA, NB, NANB,cAdag_cB, CAdag, CA, CBdag, CB = @ignore_derivatives Hamiltonians_spinless_U1_2site(M);
ex1=hopping_x(CTM,CBdag,CA,A,AA,grad_ctm_setting);
ex2=ob_onsite(CTM,cAdag_cB,A,AA,grad_ctm_setting);
ey1=hopping_y(CTM,CAdag,CA,A,AA,grad_ctm_setting);
ey2=hopping_y(CTM,CBdag,CB,A,AA,grad_ctm_setting);

e_right_top1=hopping_right_top(CTM,CBdag,CA,A,AA,grad_ctm_setting);
e_right_top2=hopping_y(CTM,CAdag,CB,A,AA,grad_ctm_setting);


E=im*ex1+im*ex2+ey1-ey2+e_right_top1-e_right_top2;
E=E+E';
dE=E+2.4020;
println("dE= "*string(dE));


direction="x";
distance=20;
CAdag_CA_ob,CAdag_CB_ob,CBdag_CA_ob,CBdag_CB_ob=cal_correl(direction,M,A, AA, chi,CTM, distance,grad_ctm_setting);

direction="y";
distance=20;
CAdag_CA_ob,CAdag_CB_ob,CBdag_CA_ob,CBdag_CB_ob=cal_correl(direction,M,A, AA, chi,CTM, distance,grad_ctm_setting);