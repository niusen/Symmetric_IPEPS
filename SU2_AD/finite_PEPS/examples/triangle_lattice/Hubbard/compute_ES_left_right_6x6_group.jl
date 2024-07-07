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
chi=100;
multiplet_tol=1e-5;
init_noise=0;

#filenm="SU_PESS_SU2_D4.jld2";
filenm="SU_PESS_SU2_D8.jld2";

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
@assert Lx==6;#for larger size, need to change code for ES
@assert Ly==6;


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

psi_double=rotate_doublelayer(psi_double);
#(1,Ly) -> (1,1)
#(1,1) -> (Lx,1)
#################################################
function get_boundary_mps(psi_double,U_set)
    Lx,Ly=size(psi_double);
    global chi, multiplet_tol

    #     global chi, multiplet_tol

    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="exact"
            mpo_mps_fun=truncate_mpo_mps_exact;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end

    ########################################
    #construct top and bot environment

    trun_history=[];
    mps_bot_set=initial_tuple(Ly);
    mps_top_set=initial_tuple(Ly);

    mps_bot=(psi_double[:,1]...,);
    mps_bot_set=vector_update(mps_bot_set,mps_bot,1);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:Ly-2
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs,_=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
        end
        return mps_top
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),Ly);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:3
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs,_=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
        mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    ###################################
    mps_R=mps_top_set[Int(Ly/2+1)];
    mps_L=mps_bot_set[Int(Ly/2)];
    mps_R=collect(mps_R);
    mps_L=collect(mps_L);
    for cx=1:Lx
        cy=Int(Ly/2);
        # A=psi[cx,cy];


        V_trivial=Rep[SU₂](0=>1);
        if cx==1
            T=mps_L[cx];
            uni=unitary(V_trivial*space(T,1), space(T,1));
            @tensor T[:]:=T[1,-3]*uni[-1,-2,1];            
            mps_L[cx]=T;

            T=mps_R[cx];
            uni=unitary(V_trivial*space(T,1), space(T,1));
            @tensor T[:]:=T[1,-3]*uni[-1,-2,1];   
            mps_R[cx]=T;        
     
        elseif cx==Lx
            T=mps_L[cx];
            uni=unitary(space(T,1)*V_trivial', space(T,1));
            @tensor T[:]:=T[1,-3]*uni[-1,-2,1];            
            mps_L[cx]=T;

            T=mps_R[cx];
            uni=unitary(space(T,1)*V_trivial', space(T,1));
            @tensor T[:]:=T[1,-3]*uni[-1,-2,1];   
            mps_R[cx]=T; 
        end

        U_U=U_set[cx];
        println(space(U_U))
        gate=swap_gate(U_U',2,3);


        T=mps_L[cx];
        @tensor T[:]:=T[-1,-2,1]*U_U'[1,3,4]*gate[-3,-4,3,4];
        T=permute(T,(4,2,3,1,));
        mps_L[cx]=T;

        T=mps_R[cx];
        @tensor T[:]:=T[-1,-2,1]*U_U[3,4,1]*gate[3,4,-3,-4];
        T=permute(T,(3,2,4,1,));
        mps_R[cx]=T;   
    end
    return mps_L,mps_R
end


U_set=UR_set[Int(Lx/2),:];
mps_L,mps_R=get_boundary_mps(psi_double,U_set);
mps_L=mps_L*10;
mps_R=mps_R*10;
###################################
TL1,TL2,TL3,TL4,TL5,TL6=mps_L;
TR1,TR2,TR3,TR4,TR5,TR6=mps_R;
###################################

#apply on-site swap gate on T tensors
TL_set=[TL1,TL2,TL3,TL4,TL5,TL6];
TR_set=[TR1,TR2,TR3,TR4,TR5,TR6];
for cc=1:length(TL_set)-1
    tl=TL_set[cc];
    gate=swap_gate(tl,2,3);
    @tensor tl[:]:=tl[-1,1,2,-4]*gate[-2,-3,1,2];
    TL_set[cc]=tl;

    tr=TR_set[cc];
    gate=swap_gate(tr,1,2);
    @tensor tr[:]:=tr[1,2,-3,-4]*gate[-1,-2,1,2];
    TR_set[cc]=tr;
end

TL1,TL2,TL3,TL4,TL5,TL6=TL_set;
TR1,TR2,TR3,TR4,TR5,TR6=TR_set;

#############################
U12a=unitary(fuse(space(TL1,1)*space(TL2,1)),space(TL1,1)*space(TL2,1));
U34a=unitary(fuse(space(TL3,1)*space(TL4,1)),space(TL3,1)*space(TL4,1));
U56a=unitary(fuse(space(TL5,1)*space(TL6,1)),space(TL5,1)*space(TL6,1));

U12b=unitary(fuse(space(TL1,3)*space(TL2,3)),space(TL1,3)*space(TL2,3));
U34b=unitary(fuse(space(TL3,3)*space(TL4,3)),space(TL3,3)*space(TL4,3));
U56b=unitary(fuse(space(TL5,3)*space(TL6,3)),space(TL5,3)*space(TL6,3));

@tensor TL12[:]:=TL1[2,1,4,-4]*TL2[3,-2,5,1]*U12a[-1,2,3]*U12b[-3,4,5];
@tensor TL34[:]:=TL3[2,1,4,-4]*TL4[3,-2,5,1]*U34a[-1,2,3]*U34b[-3,4,5];
@tensor TL56[:]:=TL5[2,1,4,-4]*TL6[3,-2,5,1]*U56a[-1,2,3]*U56b[-3,4,5];


@tensor TR12[:]:=TR1[2,1,4,-4]*TR2[3,-2,5,1]*U12b'[2,3,-1]*U12a'[4,5,-3];
@tensor TR34[:]:=TR3[2,1,4,-4]*TR4[3,-2,5,1]*U34b'[2,3,-1]*U34a'[4,5,-3];
@tensor TR56[:]:=TR5[2,1,4,-4]*TR6[3,-2,5,1]*U56b'[2,3,-1]*U56a'[4,5,-3];




function vr_ML_MR(vr0,  TL12,TL34,TL56,TR12,TR34,TR56)
    println("apply Mr");flush(stdout);

    ################

    # vr0=permute_neighbour_ind(vr0,1,2,5);
    # vr0=permute_neighbour_ind(vr0,2,3,5);
    # vr0=permute_neighbour_ind(vr0,3,4,5);
    ################


    vr=deepcopy(vr0)*0;
    for ca=1:1#parity of index U in ML


        for cb=1:1#parity of index U in MR


            
            @tensor vr_temp[:]:=TR12[-1,2,1,6]*TR34[-2,4,3,2]*TR56[-3,6,5,4]*vr0[1,3,5,-4];
            @tensor vr_temp[:]:=TL12[-1,2,1,6]*TL34[-2,4,3,2]*TL56[-3,6,5,4]*vr_temp[1,3,5,-4];
            vr=vr+vr_temp;
        end
    end



    # ################
    # if y_anti_pbc
    #     op=parity_gate(vr,1);
    #     @tensor vr[:]:=op[-4,1]*vr[-1,-2,-3,1,-5];
    # end
    # vr=permute_neighbour_ind(vr,3,4,5);
    # vr=permute_neighbour_ind(vr,2,3,5);
    # vr=permute_neighbour_ind(vr,1,2,5);
    # ################
    return vr
end


if isa(space(TL1,1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})#Z2 symmetry
    Spin_set=[0,1];
elseif isa(space(TL1,1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
    Spin_set=[0,1/2,1,3/2,2,5/2];
end
eu=Vector{ComplexF64}(undef,0);
k_phase=Vector{ComplexF64}(undef,0);
Spin=Vector{Float64}(undef,0);
for ss=1:length(Spin_set)
    if isa(space(TL1,1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})#Z2 symmetry
        v_init=TensorMap(randn, space(TL12,1)*space(TL34,1)*space(TL56,1),Rep[ℤ₂](Spin_set[ss]=>1));
    elseif isa(space(TL1,1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        v_init=TensorMap(randn, space(TL12,1)*space(TL34,1)*space(TL56,1),Rep[SU₂](Spin_set[ss]=>1));
    end
    
    v_init=permute(v_init,(1,2,3,4,),());#L1,L2,L3,L4,dummy
    
    if norm(v_init)==0
        continue;
    end
    contraction_fun_R(x)=vr_ML_MR(x, TL12,TL34,TL56,TR12,TR34,TR56);
    contraction_fun_R(v_init)
    # println(norm(v_init))
    # println(norm(contraction_fun_R(v_init)))
    @time eu0,ev=eigsolve(contraction_fun_R, v_init, 30,:LM,Arnoldi(krylovdim=40));
    ks=Vector{ComplexF64}(undef,length(eu0));
    spins=Vector{ComplexF64}(undef,length(eu0));
    for cc=1:length(eu0)
        ks[cc]=eu0[cc]/abs(eu0[cc]);
        spins[cc]=Spin_set[ss];
    end
    eu=vcat(eu,eu0);
    k_phase=vcat(k_phase,ks);
    Spin=vcat(Spin,spins);
end

##############################


println(sort(abs.(eu)))

order=sortperm(abs.(eu));
if length(order)>500
    order=order[end-500:end];
end
eu=eu[order];
# k_phase=k_phase[order];
Spin=Spin[order]





##########################

Dmax=get_max_dim(psi);


matnm="ES_6x6_D"*string(Dmax)*"_chi"*string(chi)*".mat";

matwrite(matnm, Dict(
    "eu" => eu,
    "Spin"=>Spin
); compress = false)









#########################################



