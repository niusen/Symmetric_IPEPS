using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\M1")

include("parton_CTMRG.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\swap_gate_ctmrg\\projected_energy.jl")
include("swap_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\parton_ctmrg\\mpo_mps_funs.jl")


M=1;#number of virtual mode
sublattice_order="RL";
chi=30
tol=1e-6
Guztwiller=true;#add projector



CTM_ite_nums=500;
CTM_trun_tol=1e-10;


data=load("swap_gate_Tensor_M"*string(M)*".jld2")

P_G=data["P_G"];

psi_G=data["psi_G"];   #P1,P2,L,R,D,U
M1=psi_G[1];
M2=psi_G[2];
M3=psi_G[3];
M4=psi_G[4];
M5=psi_G[5];
M6=psi_G[6];

if Guztwiller
    @tensor M1[:]:=M1[-1,-2,1]*P_G[-3,1];
    @tensor M2[:]:=M2[-1,-2,1]*P_G[-3,1];
    SS_op=data["SS_op_S"];
    Schiral_op=data["Schiral_op_S"];
else
    SS_op=data["SS_op_F"];
    Schiral_op=data["Schiral_op_F"];
end


U_phy1=unitary(fuse(space(M1,1)⊗space(M1,3)⊗space(M2,3)), space(M1,1)⊗space(M1,3)⊗space(M2,3));

@tensor A[:]:=M1[4,1,2]*M2[1,-2,3]*U_phy1[-1,4,2,3];
@tensor A[:]:=A[-1,1]*M3[1,-3,-2];
@tensor A[:]:=A[-1,-2,1]*M4[1,-4,-3];
@tensor A[:]:=A[-1,-2,-3,1]*M5[1,-5,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1]*M6[1,-6,-5];

U_phy2=unitary(fuse(space(A,1)⊗space(A,6)), space(A,1)⊗space(A,6));
@tensor A[:]:=A[1,-2,-3,-4,-5,2]*U_phy2[-1,1,2];
# P,L,R,D,U


bond=data["bond_gate"];#dummy, D1, D2 

#Add bond:both parity gate and bond operator
@tensor A[:]:=A[-1,-2,1,2,-5]*bond[-6,-3,1]*bond[-7,-4,2];
U_phy2=unitary(fuse(space(A,1)⊗space(A,6)⊗space(A,7)), space(A,1)⊗space(A,6)⊗space(A,7));
@tensor A[:]:=A[1,-2,-3,-4,-5,2,3]*U_phy2[-1,1,2,3];
#P,L,R,D,U





#swap between spin up and spin down modes, since |L,U,P><D,R|====L,U,P|><|R,D
special_gate=special_parity_gate(A,3);
@tensor A[:]:=A[-1,-2,1,-4,-5]*special_gate[-3,1];
special_gate=special_parity_gate(A,4);
@tensor A[:]:=A[-1,-2,-3,1,-5]*special_gate[-4,1];



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









A_fused=A;


conv_check="singular_value";
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true);
@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(AA_fused,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol);

display(space(CTM["Cset"][1]))
display(space(CTM["Cset"][2]))
display(space(CTM["Cset"][3]))
display(space(CTM["Cset"][4]))


println("construct physical operators");flush(stdout);
#spin-spin operator act on a single site
um,sm,vm=tsvd(permute(SS_op,(1,3,),(2,4,)));
vm=sm*vm;vm=permute(vm,(2,3,),(1,));

@tensor SS_cell[:]:=SS_op[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];#spin-spin operator inside a unitcell
if sublattice_order=="LR"
    @tensor S_L_a[:]:=um[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_R_a[:]:=um[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor S_L_b[:]:=vm[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_R_b[:]:=vm[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
elseif sublattice_order=="RL"
    @tensor S_R_a[:]:=um[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_L_a[:]:=um[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor S_R_b[:]:=vm[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_L_b[:]:=vm[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
end

if M==1        
    @tensor SS_cell[:]:=SS_cell[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_L_a[:]:=S_L_a[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_R_a[:]:=S_R_a[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_L_b[:]:=S_L_b[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_R_b[:]:=S_R_b[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
elseif M==2
    @tensor SS_cell[:]:=SS_cell[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor S_L_a[:]:=S_L_a[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor S_R_a[:]:=S_R_a[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor S_L_b[:]:=S_L_b[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor S_R_b[:]:=S_R_b[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
end



####################################
#chiral operator act on a single site: Si Sj Sk
um,sm,vm=tsvd(Schiral_op,(1,4,),(2,3,5,6,));
vm=sm*vm;
Si=permute(um,(1,2,3,));#P,P',D1
um,sm,vm=tsvd(vm,(1,2,4,),(3,5,));
vm=sm*vm;
Sj=permute(um,(2,3,1,4,));#P,P', D1,D2
Sk=permute(vm,(2,3,1,))#P,P',D2
@tensor SiSj[:]:=Si[-1,-3,1]*Sj[-2,-4,1,-5]; 
@tensor SjSk[:]:=Sj[-1,-3,-5,1]*Sk[-2,-4,1]; 
#@tensor aa[:]:=Si[-1,-4,1]*Sj[-2,-5,1,2]*Sk[-3,-6,2];
U_Schiral=unitary(fuse(space(Sj,3)⊗space(Sj,4)), space(Sj,3)⊗space(Sj,4));
@tensor Sj[:]:=Sj[-1,-2,1,2]*U_Schiral[-3,1,2];#combine two extra indices of Sj


if sublattice_order=="LR"
    @tensor Si_left[:]:=Si[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Si_right[:]:=Si[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sj_left[:]:=Sj[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sj_right[:]:=Sj[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sk_left[:]:=Sk[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sk_right[:]:=Sk[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];

    @tensor SiSj_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SjSi_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];
    @tensor SjSk_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SkSj_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];
elseif sublattice_order=="RL"
    @tensor Si_right[:]:=Si[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Si_left[:]:=Si[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sj_right[:]:=Sj[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sj_left[:]:=Sj[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sk_right[:]:=Sk[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sk_left[:]:=Sk[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];

    @tensor SjSi_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SiSj_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];
    @tensor SkSj_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SjSk_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];
end

if M==1        
    @tensor Si_left[:]:=Si_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Si_right[:]:=Si_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sj_left[:]:=Sj_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sj_right[:]:=Sj_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sk_left[:]:=Sk_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sk_right[:]:=Sk_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SiSj_op[:]:=SiSj_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SjSi_op[:]:=SjSi_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SjSk_op[:]:=SjSk_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SkSj_op[:]:=SkSj_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
elseif M==2
    @tensor Si_left[:]:=Si_left[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Si_right[:]:=Si_right[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Sj_left[:]:=Sj_left[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Sj_right[:]:=Sj_right[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Sk_left[:]:=Sk_left[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor Sk_right[:]:=Sk_right[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SiSj_op[:]:=SiSj_op[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SjSi_op[:]:=SjSi_op[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SjSk_op[:]:=SjSk_op[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    @tensor SkSj_op[:]:=SkSj_op[3,4,-3]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,-2,5,6];
end
################################################


println("construct double layer tensor with operator");flush(stdout);
AA_SS=build_double_layer_swap_op(A_fused,SS_cell,false);
AA_SLa=build_double_layer_swap_op(A_fused,S_L_a,true);
AA_SRa=build_double_layer_swap_op(A_fused,S_R_a,true);
AA_SLb=build_double_layer_swap_op(A_fused,S_L_b,true);
AA_SRb=build_double_layer_swap_op(A_fused,S_R_b,true);

AA_SiL=build_double_layer_swap_op(A_fused,Si_left,true);
AA_SiR=build_double_layer_swap_op(A_fused,Si_right,true);
AA_SjL=build_double_layer_swap_op(A_fused,Sj_left,true);
AA_SjR=build_double_layer_swap_op(A_fused,Sj_right,true);
AA_SkL=build_double_layer_swap_op(A_fused,Sk_left,true);
AA_SkR=build_double_layer_swap_op(A_fused,Sk_right,true);
AA_SiSj=build_double_layer_swap_op(A_fused,SiSj_op,true);
AA_SjSi=build_double_layer_swap_op(A_fused,SjSi_op,true);
AA_SjSk=build_double_layer_swap_op(A_fused,SjSk_op,true);
AA_SkSj=build_double_layer_swap_op(A_fused,SkSj_op,true);


println("Calculate energy terms:");flush(stdout);


Norm_1=ob_1site_closed(CTM,AA_fused)
Norm_2x=norm_2sites_x(CTM,AA_fused)
Norm_2y=norm_2sites_y(CTM,AA_fused)
Norm_4=norm_2x2(CTM,AA_fused);

#J1 term
E_1_a=ob_1site_closed(CTM,AA_SS)/Norm_1
E_1_b=ob_2sites_x(CTM,AA_SRa,AA_SLb)/Norm_2x
E_1_c=ob_2sites_y(CTM,AA_SLa,AA_SLb)/Norm_2y
E_1_d=ob_2sites_y(CTM,AA_SRa,AA_SRb)/Norm_2y

#J2 term
E_2_a=ob_2sites_y(CTM,AA_SLa,AA_SRb)/Norm_2y
E_2_b=ob_2sites_y(CTM,AA_SRa,AA_SLb)/Norm_2y
E_2_c=ob_LU_RD(CTM,AA_fused,AA_SRa,AA_SLb)/Norm_4
E_2_d=ob_RU_LD(CTM,AA_fused,AA_SLa,AA_SRb)/Norm_4



#chiral term
E_C_a=ob_2sites_y(CTM,AA_SiSj,AA_SkR)/Norm_2y
E_C_b=ob_2sites_y(CTM,AA_SiR,AA_SkSj)/Norm_2y
E_C_c=ob_2sites_y(CTM,AA_SjSk,AA_SiL)/Norm_2y
E_C_d=ob_2sites_y(CTM,AA_SiSj,AA_SkL)/Norm_2y

E_C_e=ob_LD_LU_RU(CTM,AA_fused,AA_SiR,AA_SjR,AA_SkL,U_Schiral)/Norm_4
E_C_f=ob_LU_RU_RD(CTM,AA_fused,AA_SiR,AA_SjL,AA_SkL,U_Schiral)/Norm_4
E_C_g=ob_RU_RD_LD(CTM,AA_fused,AA_SiL,AA_SjL,AA_SkR,U_Schiral)/Norm_4
E_C_h=ob_RD_LD_LU(CTM,AA_fused,AA_SiL,AA_SjR,AA_SkR,U_Schiral)/Norm_4

println("J1 terms:")
println(E_1_a)
println(E_1_b)
println(E_1_c)
println(E_1_d)
println("J2 terms:")
println(E_2_a)
println(E_2_b)
println(E_2_c)
println(E_2_d)
println("Jchi terms:")
println(E_C_a)
println(E_C_b)
println(E_C_c)
println(E_C_d)
println(E_C_e)
println(E_C_f)
println(E_C_g)
println(E_C_h)


E1=E_1_a+E_1_b+E_1_c+E_1_d;
E1=E1/4;
E2=E_2_a+E_2_b+E_2_c+E_2_d;
E2=E2/4;
EC=E_C_a+E_C_b+E_C_c+E_C_d+E_C_e+E_C_f+E_C_g+E_C_h;  
EC=EC/8*4;#plaquette

J1=1.78;
J2=0.84;
lambda_c=0.375;
E=J1*E1*2+J2*E2*2+lambda_c*(2*EC)
E/J1



# results of M=2:


# #J1 terms:
# E_1_a=-0.206662109380237 + 2.1944589331039503e-10im
# E_1_b=-0.1995063328089548 - 6.419132509471073e-11im
# E_1_c=-0.20098446013688723 - 3.149842262611985e-12im
# E_1_d=-0.2001869482868713 + 7.856203060286609e-11im
# #J2 terms:
# E_2_a=-0.01741366967684622 - 7.263772430085575e-11im
# E_2_b=-0.017304990298305425 - 1.685037589410443e-10im
# E_2_c=-0.018697361812563307 - 1.8283772283110652e-11im
# E_2_d=-0.018739997881830048 - 1.8199556269525957e-11im
# #Jchi terms:
# E_C_a=-0.07794920559487392 + 3.429098604197869e-11im
# E_C_b=-0.07784222315733416 - 5.20529775364734e-11im
# E_C_c=-0.07811633435378108 - 8.357065250050356e-11im
# E_C_d=-0.07811633435378101 - 8.357062829133927e-11im
# E_C_e=-0.07811616290215918 - 1.515394370737235e-11im
# E_C_f=-0.07814400860604917 - 2.430509729692064e-11im
# E_C_g=-0.07837942735119692 - 3.2236411889810624e-11im
# E_C_h=-0.07826143417909616 - 3.116130014884925e-11im

# E1=E_1_a+E_1_b+E_1_c+E_1_d;
# E1=E1/4;
# E2=E_2_a+E_2_b+E_2_c+E_2_d;
# E2=E2/4;
# EC=E_C_a+E_C_b+E_C_c+E_C_d+E_C_e+E_C_f+E_C_g+E_C_h;  
# EC=EC/8*4;#plaquette

# J1=1.78;
# J2=0.84;
# lambda_c=0.375;
# E=J1*E1*2+J2*E2*2+lambda_c*(2*EC)
# E/J1
