using Distributed
if length(ARGS)>0
  @assert nworkers()==1;
  @show addprocs(parse(Int, ARGS[1]); exeflags=["--heap-size-hint=40G"])
end
using LinearAlgebra:diag,I,diagm 
using JLD2,ChainRulesCore,MAT, Zygote
using Zygote:@ignore_derivatives
@everywhere using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT, Random
cd(@__DIR__)


include("../../../../../src/fermionic/swap_funs.jl")
include("../../../../../src/fermionic/fermi_permute.jl")
include("../../../../../src/mps_algorithms/Projector_funs.jl")
include("../../../../../src/bosonic/Settings.jl")
include("../../../../../src/bosonic/Settings_cell.jl")
include("../../../../../src/bosonic/AD_lib.jl")
include("../../../../../src/bosonic/iPEPS_ansatz.jl")
include("../../../../../src/bosonic/CTMRG.jl")
include("../../../../../src/fermionic/Fermionic_CTMRG.jl")
include("../../../../../src/fermionic/Fermionic_CTMRG_unitcell.jl")
include("../../../../../src/fermionic/triangle_fiPESS_method.jl")

####################
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
@everywhere import LinearAlgebra.BLAS as BLAS

if nworkers()==1
  n_cpu=10;
  BLAS.set_num_threads(n_cpu);
  println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);
else
    @everywhere BLAS.set_num_threads(5)
    BLAS.set_num_threads(5);#main processer
    @show n_cpu=nworkers()#*(BLAS.get_num_threads());
end
Base.Sys.set_process_title("ES")
main_pid=getpid();
println("pid="*string(main_pid));;flush(stdout);
#change name process for each worker
if nworkers()>1
for (i, pid) in enumerate(workers())
    remotecall_wait(() -> begin
        title = string(main_pid)*"-$i"
        Base.Sys.set_process_title(title)
        println("Worker $(myid()): set process title to '$title'")
    end, pid)
end
end

##########################
function get_CTM(y_translation,force_redo_CTMRG)
    chi=40;
    global chi
    @show filenm="newnewversion_stochastic_iPESS_LS_D_8_chi_120.jld2";
    #@show filenm="newnewversion_FU_iPESS_LS_D_12_chi_80.jld2";
    data=load(filenm);


    #convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


    algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
    algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
    dump(algrithm_CTMRG_settings);
    global algrithm_CTMRG_settings

    LS_ctm_setting=LS_CTMRG_settings();
    LS_ctm_setting.CTM_conv_tol=1e-6;
    LS_ctm_setting.CTM_ite_nums=30;
    LS_ctm_setting.CTM_trun_tol=1e-8;
    LS_ctm_setting.svd_lanczos_tol=1e-8;
    LS_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
    LS_ctm_setting.conv_check="singular_value";
    LS_ctm_setting.CTM_ite_info=true;
    LS_ctm_setting.CTM_conv_info=true;
    LS_ctm_setting.CTM_trun_svd=false;
    LS_ctm_setting.construct_double_layer=true;
    LS_ctm_setting.grad_checkpoint=true;
    dump(LS_ctm_setting);

    global Lx,Ly
    if haskey(data,"x")
        global Lx,Ly
        Lx,Ly=size(state);
        A_cell=initial_tuple_cell(Lx,Ly);
        for ca=1:Lx
            for cb=1:Ly
                if isa(state[ca,cb],Square_iPEPS)
                    A_cell=fill_tuple(A_cell, state[ca,cb].T, ca,cb);
                elseif isa(state[ca,cb],Triangle_iPESS)
                    A0=iPESS_to_iPEPS(state[ca,cb]).T;
                    A0=A0/norm(A0)*10;
                    A_cell=fill_tuple(A_cell, A0, ca,cb);
                else
                    error("unknown type ansatz")
                end
            end
        end
    else
        global Lx,Ly
        B_set=data["B_set"];
        T_set=data["T_set"];
        Lx,Ly=size(B_set);
        A_cell=convert_iPESS_to_iPEPS(B_set,T_set);
    end
    global D
    D=dim(space(A_cell[1][1],1));

    ##################################
    global chi,multiplet_tol,projector_trun_tol
    multiplet_tol=1e-5;
    projector_trun_tol=LS_ctm_setting.CTM_trun_tol
    ###################################


    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    
    if y_translation==false
        if (force_redo_CTMRG==true)|(isfile(filenm*"_CTM.jld2")==false)
            CTM_cell, _, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);
        else
            println("load CTM directly")
            data=load(filenm*"_CTM.jld2");
            CTM_cell=data["CTM_cell"];
            U_L_cell=data["U_L_cell"];
            U_D_cell=data["U_D_cell"];
            U_R_cell=data["U_R_cell"];
            U_U_cell=data["U_U_cell"];
        end
        return filenm, CTM_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell
    elseif y_translation==true
        if (force_redo_CTMRG==true)|(isfile(filenm*"_CTM_tran.jld2")==false)
            CTM_cell_tran, _, U_L_cell_tran,U_D_cell_tran,U_R_cell_tran,U_U_cell_tran,ite_num,ite_err=Fermionic_CTMRG_cell_y_translate(A_cell,chi,init, init_CTM,LS_ctm_setting);
        else
            println("load CTM directly")
            data=load(filenm*"_CTM_tran.jld2");
            CTM_cell_tran=data["CTM_cell_tran"];
            U_L_cell_tran=data["U_L_cell_tran"];
            U_D_cell_tran=data["U_D_cell_tran"];
            U_R_cell_tran=data["U_R_cell_tran"];
            U_U_cell_tran=data["U_U_cell_tran"];
        end
        return filenm, CTM_cell_tran, U_L_cell_tran,U_D_cell_tran,U_R_cell_tran,U_U_cell_tran
    end
end

function main()
complete_divide=true;
force_redo_CTMRG=false;
global y_anti_pbc,D,chi
y_anti_pbc=true;

#############################

y_translation=false;
filenm, CTM_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell=get_CTM(y_translation,force_redo_CTMRG);
jldsave(filenm*"_CTM.jld2";filenm, CTM_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell);
y_translation=true;
filenm, CTM_cell_tran, U_L_cell_tran,U_D_cell_tran,U_R_cell_tran,U_U_cell_tran=get_CTM(y_translation,force_redo_CTMRG);
jldsave(filenm*"_CTM_tran.jld2";filenm, CTM_cell_tran, U_L_cell_tran,U_D_cell_tran,U_R_cell_tran,U_U_cell_tran);
#############################
#get T tensors
ca=1;
cb=1;
TL01=CTM_cell.Tset[ca][cb].T4;
TR01=CTM_cell_tran.Tset[mod1(ca+1,Lx)][cb].T2;
U_L=U_L_cell[ca][cb];
U_R=U_R_cell_tran[ca+1][cb];

cb=2;
TL02=CTM_cell.Tset[ca][cb].T4;
TR02=CTM_cell_tran.Tset[mod1(ca+1,Lx)][cb].T2;
U_L=U_L_cell[ca][cb];
U_R=U_R_cell_tran[ca+1][cb];
#############################
#extra swap gate that was not included when construct double layer tensor
gate=swap_gate(U_L,2,3);
@tensor U_L_new[:]:=U_L'[1,2,-1]*gate[1,2,3,4]*U_L[-2,3,4];
gate=swap_gate(U_R,1,2);
@tensor U_R_new[:]:=U_R'[-1,1,2]*gate[1,2,3,4]*U_R[3,4,-2];

@tensor TL01[:]:=TL01[-1,1,-3]*U_R_new'[-2,1]; 
@tensor TR01[:]:=TR01[-1,1,-3]*U_L_new'[-2,1];
@tensor TL02[:]:=TL02[-1,1,-3]*U_R_new'[-2,1]; 
@tensor TR02[:]:=TR02[-1,1,-3]*U_L_new'[-2,1]; 

TL01=TL01*10;
TR01=TR01*10;
TL02=TL02*10;
TR02=TR02*10;
#############################
TL1=deepcopy(TL01);
TL2=deepcopy(TL02);
TL3=deepcopy(TL01);
TL4=deepcopy(TL02);
TL5=deepcopy(TL01);
TL6=deepcopy(TL02);
TL7=deepcopy(TL01);
TL8=deepcopy(TL02);

TR1=deepcopy(TR01);
TR2=deepcopy(TR02);
TR3=deepcopy(TR01);
TR4=deepcopy(TR02);
TR5=deepcopy(TR01);
TR6=deepcopy(TR02);
TR7=deepcopy(TR01);
TR8=deepcopy(TR02);

#############################
#extra parity gate from crossing
gate=parity_gate(TL1,3);
@tensor TL1[:]:=TL1[-1,-2,1]*gate[-3,1];
gate=parity_gate(TR1,1);
@tensor TR1[:]:=TR1[1,-2,-3]*gate[-1,1];

#############################
if y_anti_pbc
    gate=parity_gate(TL1,3);
    @tensor TL1[:]:=TL1[-1,-2,1]*gate[-3,1];
    gate=parity_gate(TR1,1);
    @tensor TR1[:]:=TR1[1,-2,-3]*gate[-1,1];
end
#############################
#expand T tensors
@tensor TL1[:]:=TL1[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL2[:]:=TL2[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL3[:]:=TL3[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL4[:]:=TL4[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL5[:]:=TL5[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL6[:]:=TL6[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL7[:]:=TL7[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL8[:]:=TL8[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U

@tensor TR1[:]:=TR1[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR2[:]:=TR2[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR3[:]:=TR3[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR4[:]:=TR4[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR5[:]:=TR5[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR6[:]:=TR6[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR7[:]:=TR7[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR8[:]:=TR8[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U

#############################
#apply on-site swap gate on T tensors
TL_set=[TL1,TL2,TL3,TL4,TL5,TL6,TL7,TL8];
TR_set=[TR1,TR2,TR3,TR4,TR5,TR6,TR7,TR8];
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

TL1,TL2,TL3,TL4,TL5,TL6,TL7,TL8=TL_set;
TR1,TR2,TR3,TR4,TR5,TR6,TR7,TR8=TR_set;

#############################

# P_odda_set,P_evena_set=projector_split(space(TL1,4));
# P_oddb_set,P_evenb_set=projector_split(space(TR1,4));

P_seta=[];
P_setb=[];
if complete_divide==true
    P_seta=projector_parity_divide(space(TL1,4));
    P_setb=projector_parity_divide(space(TR1,4));

    P_odda_set,P_evena_set=P_seta;
    P_oddb_set,P_evenb_set=P_setb;
    @show length(P_odda_set)
    @show length(P_evena_set)
elseif complete_divide==false
    P_odda,P_evena=projector_parity(space(TL1,4));
    P_oddb,P_evenb=projector_parity(space(TR1,4));

    P_odda_set=(P_odda,);
    P_evena_set=(P_evena,);
    P_oddb_set=(P_oddb,);
    P_evenb_set=(P_evenb,);

    P_seta=[P_odda_set,P_evena_set];
    P_setb=[P_oddb_set,P_evenb_set];
end




TL1_odd_set=Vector{TensorMap}(undef,length(P_odda_set));
TL1_even_set=Vector{TensorMap}(undef,length(P_evena_set));
TL8_odd_set=Vector{TensorMap}(undef,length(P_odda_set));
TL8_even_set=Vector{TensorMap}(undef,length(P_evena_set));
TR1_odd_set=Vector{TensorMap}(undef,length(P_oddb_set));
TR1_even_set=Vector{TensorMap}(undef,length(P_evenb_set));
TR8_odd_set=Vector{TensorMap}(undef,length(P_oddb_set));
TR8_even_set=Vector{TensorMap}(undef,length(P_evenb_set));
for cc=1:length(P_odda_set)
    @tensor TL1_[:]:=TL1[-1,-2,-3,1]*P_odda_set[cc][-4,1];
    TL1_odd_set[cc]=TL1_;
    @tensor TL8_[:]:=TL8[-1,1,-3,-4]*P_odda_set[cc]'[1,-2];
    TL8_odd_set[cc]=TL8_;
end
for cc=1:length(P_evena_set)
    @tensor TL1_[:]:=TL1[-1,-2,-3,1]*P_evena_set[cc][-4,1];
    TL1_even_set[cc]=TL1_;
    @tensor TL8_[:]:=TL8[-1,1,-3,-4]*P_evena_set[cc]'[1,-2];
    TL8_even_set[cc]=TL8_;
end
for cc=1:length(P_oddb_set)
    @tensor TR1_[:]:=TR1[-1,-2,-3,1]*P_oddb_set[cc][-4,1];
    TR1_odd_set[cc]=TR1_;
    @tensor TR8_[:]:=TR8[-1,1,-3,-4]*P_oddb_set[cc]'[1,-2];
    TR8_odd_set[cc]=TR8_;
end
for cc=1:length(P_evenb_set)
    @tensor TR1_[:]:=TR1[-1,-2,-3,1]*P_evenb_set[cc][-4,1];
    TR1_even_set[cc]=TR1_;
    @tensor TR8_[:]:=TR8[-1,1,-3,-4]*P_evenb_set[cc]'[1,-2];
    TR8_even_set[cc]=TR8_;
end


U_D_D1a=unitary(fuse(space(TL2,1)*space(TL3,1)), space(TL2,1)*space(TL3,1));
U_D_D2a=unitary(fuse(space(TL4,1)*space(TL5,1)), space(TL4,1)*space(TL5,1));
U_D_D3a=unitary(fuse(space(TL6,1)*space(TL7,1)), space(TL6,1)*space(TL7,1));
U_D_D_seta=(U_D_D1a,U_D_D2a,U_D_D3a,);
U_D_D1m=unitary(fuse(space(TL2,3)*space(TL3,3)), space(TL2,3)*space(TL3,3));
U_D_D2m=unitary(fuse(space(TL4,3)*space(TL5,3)), space(TL4,3)*space(TL5,3));
U_D_D3m=unitary(fuse(space(TL6,3)*space(TL7,3)), space(TL6,3)*space(TL7,3));
U_D_D_setm=(U_D_D1m,U_D_D2m,U_D_D3m,);

@tensor TL23[:]:=TL2[2,1,4,-4]*TL3[3,-2,5,1]*U_D_D1a[-1,2,3]*U_D_D1m[-3,4,5];
@tensor TL45[:]:=TL4[2,1,4,-4]*TL5[3,-2,5,1]*U_D_D2a[-1,2,3]*U_D_D2m[-3,4,5];
@tensor TL67[:]:=TL6[2,1,4,-4]*TL7[3,-2,5,1]*U_D_D3a[-1,2,3]*U_D_D3m[-3,4,5];

@tensor TR23[:]:=TR2[2,1,4,-4]*TR3[3,-2,5,1]*U_D_D1a'[4,5,-3]*U_D_D1m'[2,3,-1];
@tensor TR45[:]:=TR4[2,1,4,-4]*TR5[3,-2,5,1]*U_D_D2a'[4,5,-3]*U_D_D2m'[2,3,-1];
@tensor TR67[:]:=TR6[2,1,4,-4]*TR7[3,-2,5,1]*U_D_D3a'[4,5,-3]*U_D_D3m'[2,3,-1];

TL_set=((TL1_odd_set,TL1_even_set,), TL23,TL45,TL67, (TL8_odd_set,TL8_even_set,),);
TR_set=((TR1_odd_set,TR1_even_set,), TR23,TR45,TR67, (TR8_odd_set,TR8_even_set,),);

gate_middle1=parity_gate(TL1,3);
gate_middle23=parity_gate(TL23,3);
gate_middle45=parity_gate(TL45,3);
gate_middle67=parity_gate(TL67,3);
gate_middle_set=(gate_middle1,gate_middle23,gate_middle45,gate_middle67,);

global U_D_D1a,U_D_D2a,U_D_D3a
@assert norm(U_D_D1a-U_D_D2a)<1e-10;#ensure the same virtual space, otherwise need to be very careful
@assert norm(U_D_D2a-U_D_D3a)<1e-10;#ensure the same virtual space, otherwise need to be very careful

@everywhere function apply_TR(vr0, cb, TR1_te, TR23_tem, TR45_tem, TR67_tem, TR8_te, gate_middle_set, v0_sample)
    vr1=v0_sample*0;
    for cbb=1:length(TR1_te)
        TR8_tem=TR8_te[cbb];
        if mod(cb,2)==1
            @tensor TR1_tem[:]:=TR1_te[cbb][1,-2,-3,-4]*gate_middle_set[1]'[1,-1];
        else
            TR1_tem=TR1_te[cbb];
        end

        #@show cbb
        @tensor vr_temp[:]:=TR1_tem[-1,2,1,10]*TR23_tem[-2,4,3,2]*TR45_tem[-3,6,5,4]*TR67_tem[-4,8,7,6]*TR8_tem[-5,10,9,8]*vr0[1,3,5,7,9,-6];
        vr1=vr1+vr_temp;
    end
    return vr1
end

@everywhere function apply_TL(vr1, ca, TL1_te, TL23_tem, TL45_tem, TL67_tem, TL8_te, gate_middle_set, v0_sample)
    vr2=v0_sample*0;
    for caa=1:length(TL1_te)
        TL8_tem=TL8_te[caa];
        if mod(ca,2)==1
            @tensor TL1_tem[:]:=TL1_te[caa][-1,-2,1,-4]*gate_middle_set[1][-3,1];
        else
            TL1_tem=TL1_te[caa];
        end
        #@show caa
        @tensor vr_temp[:]:=TL1_tem[-1,2,1,10]*TL23_tem[-2,4,3,2]*TL45_tem[-3,6,5,4]*TL67_tem[-4,8,7,6]*TL8_tem[-5,10,9,8]*vr1[1,3,5,7,9,-6];
        vr2=vr2+vr_temp;
    end
    return vr2
end

function vr_ML_MR(vr0,  TL_set,TR_set, gate_middle_set)
    println("apply Mr");flush(stdout);
    ################
    global U_D_D1a,y_anti_pbc
    if y_anti_pbc
        op=parity_gate(vr0,1);
        @tensor vr0[:]:=op[-1,1]*vr0[1,-2,-3,-4,-5,-6];
    end
    vr0=permute_neighbour_ind(vr0,1,2,6);
    vr0=permute_neighbour_ind(vr0,2,3,6);
    vr0=permute_neighbour_ind(vr0,3,4,6);
    vr0=permute_neighbour_ind(vr0,4,5,6);
    @tensor vr0[:]:=vr0[7,4,1,3,-5,-6]*U_D_D1a'[-1,8,7]*U_D_D1a'[9,5,4]*U_D_D1a'[6,2,1]*U_D_D1a[-2,8,9]*U_D_D1a[-3,5,6]*U_D_D1a[-4,2,3];
    ################

    TL1_set,TL23,TL45,TL67,TL8_set=TL_set;
    TR1_set,TR23,TR45,TR67,TR8_set=TR_set;
    # vr1=deepcopy(vr0)*0;

    vr1=TensorMap(randn, space(TR1_set[1][1],1)*space(TR23,1)*space(TR45,1)*space(TR67,1)*space(TR8_set[1][1],1),space(vr0,6)');
    vr1=permute(vr1,(1,2,3,4,5,6,))*0;

    for cb=1:2#parity of index U in MR
        TR1_te=deepcopy(TR1_set[cb]);
        TR23_tem=deepcopy(TR23);
        TR45_tem=deepcopy(TR45);
        TR67_tem=deepcopy(TR67);
        TR8_te=deepcopy(TR8_set[cb]);

        if mod(cb,2)==1
            @tensor TR23_tem[:]:=TR23_tem[1,-2,-3,-4]*gate_middle_set[2]'[1,-1];
            @tensor TR45_tem[:]:=TR45_tem[1,-2,-3,-4]*gate_middle_set[3]'[1,-1];
            @tensor TR67_tem[:]:=TR67_tem[1,-2,-3,-4]*gate_middle_set[4]'[1,-1];
        end

        N_terms=length(TR1_te);
        ind_set=distribute_workers(N_terms,nworkers());

        vr1_comps=Vector{Any}(undef,size(ind_set,1));
        @sync begin
            for cp=1:size(ind_set,1)
                vr1_comps[cp]= @spawnat workers()[cp] apply_TR(vr0, cb, TR1_te[ind_set[cp,1]:ind_set[cp,2]], TR23_tem, TR45_tem, TR67_tem, TR8_te[ind_set[cp,1]:ind_set[cp,2]], gate_middle_set, vr1*0);
            end
        end
        vr1_cb=sum(fetch.(vr1_comps));
        vr1=vr1+vr1_cb;
    end

    vr2=vr0*0;
    for ca=1:2#parity of index U in ML
        TL1_te=deepcopy(TL1_set[ca]);
        TL23_tem=deepcopy(TL23);
        TL45_tem=deepcopy(TL45);
        TL67_tem=deepcopy(TL67);
        TL8_te=deepcopy(TL8_set[ca]);

        if mod(ca,2)==1
            @tensor TL23_tem[:]:=TL23_tem[-1,-2,1,-4]*gate_middle_set[2][-3,1];
            @tensor TL45_tem[:]:=TL45_tem[-1,-2,1,-4]*gate_middle_set[3][-3,1];
            @tensor TL67_tem[:]:=TL67_tem[-1,-2,1,-4]*gate_middle_set[4][-3,1];
        end

        N_terms=length(TL1_te);
        ind_set=distribute_workers(N_terms,nworkers());

        vr2_comps=Vector{Any}(undef,size(ind_set,1));
        @sync begin
            for cp=1:size(ind_set,1)
                vr2_comps[cp]= @spawnat workers()[cp] apply_TL(vr1, ca, TL1_te[ind_set[cp,1]:ind_set[cp,2]], TL23_tem, TL45_tem, TL67_tem, TL8_te[ind_set[cp,1]:ind_set[cp,2]], gate_middle_set, vr0*0);
            end
        end
        vr2_ca=sum(fetch.(vr2_comps));
        vr2=vr2+vr2_ca;


    end
    flush(stdout);
    return vr2
end


Spin_set=[0,1/2,1,3/2,2,5/2];
n_Es=[50,40,40,20,20,10];
#n_Es=[10,10,10,10,10,10];
eu=Vector{ComplexF64}(undef,0);
k_phase=Vector{ComplexF64}(undef,0);
Spin=Vector{Float64}(undef,0);

@show S_ind=3;

for ss=S_ind:S_ind#1:length(Spin_set)
    Random.seed!(1234)
    v_init=TensorMap(randn, space(TL1,1)*fuse(space(TL2,1)*space(TL3,1))*fuse(space(TL4,1)*space(TL5,1))*fuse(space(TL6,1)*space(TL7,1))*space(TL8,1),Rep[SU₂](Spin_set[ss]=>1));
    v_init=permute(v_init,(1,2,3,4,5,6,),());#L1,L2,L3,L4,dummy

    memory=Base.summarysize(v_init)/1024/1024/1024;
    println("Memory cost for a single vector: "*string(memory)*" GB.")
    if norm(v_init)==0
        continue;
    end
    contraction_fun_R(x)=vr_ML_MR(x, TL_set,TR_set, gate_middle_set);
    @time v_1step=contraction_fun_R(v_init);flush(stdout);

    # jldsave("test2.jld2";v_init,v_1step,TL_set,TR_set,gate_middle_set);
    # error("stop")

    @time contraction_fun_R(v_init);flush(stdout);
    @time eu0,ev,info=eigsolve(contraction_fun_R, v_init, n_Es[ss],:LM,Arnoldi(krylovdim=n_Es[ss]*2,tol=norm(v_1step)/norm(v_init)*1e-6,eager=true));
    @show info
    ks=Vector{ComplexF64}(undef,length(eu0));
    spins=Vector{ComplexF64}(undef,length(eu0));
    for cc=1:length(eu0)
        ks[cc]=eu0[cc]/abs(eu0[cc]);
        spins[cc]=Spin_set[ss];
    end
    @show eu0
    # eu=vcat(eu,eu0);
    # k_phase=vcat(k_phase,ks);
    # Spin=vcat(Spin,spins);


    #############################
    if y_anti_pbc
        matnm="ES_CTM_D"*string(D)*"_chi"*string(chi)*"_Nv8_APBC_S"*string(Spin_set[ss])*".mat";
    else
        matnm="ES_CTM_D"*string(D)*"_chi"*string(chi)*"_Nv8_PBC_S"*string(Spin_set[ss])*".mat";
    end
    matwrite(matnm, Dict(
        "ks" => ks,
        "eu0" => eu0,
        "spins"=>spins
    ); compress = false)
end

##############################


# println(sort(abs.(eu)))

# order=sortperm(abs.(eu));
# if length(order)>500
#     order=order[end-500:end];
# end
# eu=eu[order];
# k_phase=k_phase[order];
# Spin=Spin[order]


##########################






end

main()