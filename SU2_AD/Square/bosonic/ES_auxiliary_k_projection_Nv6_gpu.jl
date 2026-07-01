using TensorKit, Zygote
using LinearAlgebra: I, diagm, norm
import LinearAlgebra.BLAS as BLAS
using JLD2, ChainRulesCore, MAT
using KrylovKit
using Random
using Zygote: @ignore_derivatives

cd(@__DIR__)

@show run_device = "cuda:1"  # choose from "cpu", "cuda:0", "cuda:1"

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT_DIR, "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(ROOT_DIR, "src", "bosonic", "AD_lib.jl"))
include(joinpath(ROOT_DIR, "src", "bosonic", "Settings.jl"))
include(joinpath(ROOT_DIR, "src", "tensorkit_compat.jl"))
include("ES_gpu_utils.jl")
include(joinpath(ROOT_DIR, "src", "bosonic", "square", "square_chiral_pair_model.jl"))
include("ES_auxiliary_Ty_shift_builders.jl");

# ES_algorithms.jl provides shared helpers such as vison_op, k_projection,
# calculate_k, and CTM_T_action. The explicit version avoids global U_L/U_R.
include(joinpath(ROOT_DIR, "src", "mps_algorithms", "ES_algorithms.jl"))
include(joinpath(ROOT_DIR, "src", "mps_algorithms", "ES_algorithms_explicit.jl"))
include(joinpath(ROOT_DIR, "src", "mps_algorithms", "Projector_funs.jl"))

Random.seed!(555)

###########################
n_cpu = 10
BLAS.set_num_threads(n_cpu)
println("number of cpus: " * string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C" * string(n_cpu) * "_ES_pair_custom_left")
println("pid=" * string(getpid()))
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm = gethostname()
###########################

init_statenm = "OptimKit_SU2_chiral_pair_D6_chi_54_-1.9594.jld2"
T_tensor_scale = 100

A=data=load(init_statenm)["A"];
A=A*T_tensor_scale;


@show Nv = 6
@show kind=0  #momentum projector for A subsystem

EH_n = 30
run_projector_benchmark = false


# AA, U_L, U_D, U_R, U_U = build_double_layer(A);

AA_0, U_L, U_D, U_R, U_U=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',A);

Ty1=chiral_pair_Ty1_mpo();
Tym1=chiral_pair_Tym1_mpo();
Ty2=compose_chiral_pair_shift_mpo(Ty1, Ty1);
Ty3=compose_chiral_pair_shift_mpo(Ty2, Ty1);
Tym2=compose_chiral_pair_shift_mpo(Tym1, Tym1);

AA_1, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Ty1));
AA_m1, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Tym1));
AA_2, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Ty2));
AA_m2, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Tym2));
AA_3, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Ty3));


Ve=space(AA_0,1)';
vl0=randn(ComplexF64,Ve*Ve*Ve*Ve*Ve*Ve);

vr0=randn(ComplexF64,Ve'*Ve'*Ve'*Ve'*Ve'*Ve');

AA_0 = to_es_device(AA_0)
AA_1 = to_es_device(AA_1)
AA_2 = to_es_device(AA_2)
AA_3 = to_es_device(AA_3)
AA_m2 = to_es_device(AA_m2)
AA_m1 = to_es_device(AA_m1)
U_L = to_es_device(U_L)
U_R = to_es_device(U_R)

es_synchronize()



#############
# if run_projector_benchmark
#     projectors = projector_general_SU2(space(AA_3, 4); check=true)
#     projectors = to_es_device(projectors)
#     ind1 = 1
#     ind2 = 1
#     @show length(projectors)
#     @assert 1 <= ind1 <= length(projectors)
#     @assert 1 <= ind2 <= length(projectors)
#     @show TensorKit.storagetype(AA_3)
#     @show TensorKit.storagetype(vl0)
#     @show TensorKit.storagetype(projectors[ind1])
#     @time begin
#         @tensor AA6a[:] := projectors[ind1][-8, 1] * AA_3[-1, 3, -5, 1] *
#             AA_3[-2, 4, -6, 3] * AA_3[-3, 2, -7, 4] * projectors[ind2]'[2, -4]
#         @tensor vl_[:] := vl0[1, 2, 3, -6, -7, -8] * AA6a[1, 2, 3, -2, -3, -4, -5, -1]
#         AA6a = nothing
#         @tensor AA6b[:] := projectors[ind2][-8, 1] * AA_3[-1, 3, -5, 1] *
#             AA_3[-2, 4, -6, 3] * AA_3[-3, 2, -7, 4] * projectors[ind1]'[2, -4]
#         @tensor vl_[:] := vl_[1, 2, -1, -2, -3, 3, 4, 5] * AA6b[3, 4, 5, 1, -4, -5, -6, 2]
#         AA6b = nothing
#         es_synchronize()
#         ipeps_reclaim_device_memory!()
#         nothing
#     end
# end
##############

#Ty3

spins=[0,1/2,1,3/2,2,5/2];#content of irreps in of small projectors
max_eff_dim_set=[15,10,6,10,9,12]
projectors_Ty3 = projector_general_SU2(space(AA_3, 4); check=true);
projectors_Ty3_larger=group_SU2_projectors(projectors_Ty3; max_eff_dim=15, alpha=0.8, check=true);
projectors_Ty3 = to_es_device(projectors_Ty3);
projectors_Ty3_larger = to_es_device(projectors_Ty3_larger);
# @show length(projectors_Ty3)*length(projectors_Ty3_larger)
@show length(projectors_Ty3)
@show TensorKit.storagetype(AA_3)
@show TensorKit.storagetype(vl0)
#CUDA.pool_status()
ipeps_reclaim_device_memory!()




function  apply_l_Ty3(vl, AA_3, projectors_Ty3, projectors_Ty3_larger)
    function apply_comp(vl_in, AA_3, projectors_Ty3, projectors_Ty3_larger, ind1, ind2)
        @tensor AA6a[:] := projectors_Ty3[ind1][-8, 1] * AA_3[-1, 3, -5, 1] *
            AA_3[-2, 4, -6, 3] * AA_3[-3, 2, -7, 4] * projectors_Ty3_larger[ind2]'[2, -4]
        @tensor vl_tmp[:] := vl_in[1, 2, 3, -6, -7, -8] * AA6a[1, 2, 3, -2, -3, -4, -5, -1]
        AA6a = nothing
        GC.gc(true)
        ipeps_reclaim_device_memory!(aggressive=true)
        @tensor AA6b[:] := projectors_Ty3_larger[ind2][-8, 1] * AA_3[-1, 3, -5, 1] *
            AA_3[-2, 4, -6, 3] * AA_3[-3, 2, -7, 4] * projectors_Ty3[ind1]'[2, -4]
        @tensor vl_out[:] := vl_tmp[1, 2, -1, -2, -3, 3, 4, 5] * AA6b[3, 4, 5, 1, -4, -5, -6, 2]
        AA6b = nothing
        vl_tmp = nothing

        es_synchronize()
        vl_cpu=to_es_cpu(vl_out);
        vl_out=nothing
        
        return vl_cpu
    end
    GC.gc(true)
    ipeps_reclaim_device_memory!(aggressive=true)
    vl_gpu=to_es_device(deepcopy(vl));
    vl_total=vl*0;
    for ind1 =length(projectors_Ty3):-1:1
        @show ind1
        t=@elapsed begin
            for ind2 =length(projectors_Ty3_larger):-1:1
                vl_comp=apply_comp(vl_gpu, AA_3, projectors_Ty3, projectors_Ty3_larger, ind1, ind2);
                vl_total=vl_total+vl_comp
                vl_comp=nothing
                GC.gc(true)
                ipeps_reclaim_device_memory!(aggressive=true)
            end
            CUDA.pool_status();flush(stdout)
        end
        println("time = ", t)
    end
    return vl_total
end
apply_l_Ty3(vl0, AA_3, projectors_Ty3, projectors_Ty3_larger)
##############

#Ty2
projectors_Ty2 = projector_general_SU2(space(AA_2, 4); check=true);
projectors_Ty2_larger=group_SU2_projectors(projectors_Ty2; max_eff_dim=10, alpha=0.8, check=true);
projectors_Ty2 = to_es_device(projectors_Ty2);
projectors_Ty2_larger = to_es_device(projectors_Ty2_larger);
# @show length(projectors_Ty2)*length(projectors_Ty2_larger)
@show length(projectors_Ty2)
@show TensorKit.storagetype(AA_2)
@show TensorKit.storagetype(vl0)
#CUDA.pool_status()
ipeps_reclaim_device_memory!()




function  apply_l_Ty2(vl, AA_2, projectors_Ty2, projectors_Ty2_larger)
    function apply_comp(vl_in, AA_2, projectors_Ty2, projectors_Ty2_larger, ind1, ind2)
        @tensor AA6a[:] := projectors_Ty2[ind1][-8, 1] * AA_2[-1, 3, -5, 1] *
            AA_2[-2, 4, -6, 3] * AA_2[-3, 2, -7, 4] * projectors_Ty2_larger[ind2]'[2, -4]
        @tensor vl_tmp[:] := vl_in[1, 2, 3, -6, -7, -8] * AA6a[1, 2, 3, -2, -3, -4, -5, -1]
        AA6a = nothing
        GC.gc(true)
        ipeps_reclaim_device_memory!(aggressive=true)
        @tensor AA6b[:] := projectors_Ty2_larger[ind2][-8, 1] * AA_2[-1, 3, -5, 1] *
            AA_2[-2, 4, -6, 3] * AA_2[-3, 2, -7, 4] * projectors_Ty2[ind1]'[2, -4]
        @tensor vl_out[:] := vl_tmp[1, 2, -1, -2, -3, 3, 4, 5] * AA6b[3, 4, 5, 1, -4, -5, -6, 2]
        AA6b = nothing
        vl_tmp = nothing

        es_synchronize()
        vl_cpu=to_es_cpu(vl_out);
        vl_out=nothing
        
        return vl_cpu
    end
    GC.gc(true)
    ipeps_reclaim_device_memory!(aggressive=true)
    vl_gpu=to_es_device(deepcopy(vl));
    vl_total=vl*0;
    for ind1 =length(projectors_Ty2):-1:1
        @show ind1
        t=@elapsed begin
            for ind2 =length(projectors_Ty2_larger):-1:1
                vl_comp=apply_comp(vl_gpu, AA_2, projectors_Ty2, projectors_Ty2_larger, ind1, ind2);
                vl_total=vl_total+vl_comp
                vl_comp=nothing
                GC.gc(true)
                ipeps_reclaim_device_memory!(aggressive=true)
            end
            CUDA.pool_status();flush(stdout)
        end
        println("time = ", t)
    end
    return vl_total
end
#apply_l_Ty2(vl0, AA_2, projectors_Ty2, projectors_Ty2_larger)
######


#Ty1
projectors_Ty1 = projector_general_SU2(space(AA_1, 4); check=true);
projectors_Ty1_larger=group_SU2_projectors(projectors_Ty1; max_eff_dim=10, alpha=0.8, check=true);
projectors_Ty1 = to_es_device(projectors_Ty1);
projectors_Ty1_larger = to_es_device(projectors_Ty1_larger);
# @show length(projectors_Ty1)*length(projectors_Ty1_larger)
@show length(projectors_Ty1)
@show TensorKit.storagetype(AA_1)
@show TensorKit.storagetype(vl0)
#CUDA.pool_status()
ipeps_reclaim_device_memory!()




function  apply_l_Ty1(vl, AA_1, projectors_Ty1, projectors_Ty1_larger)
    function apply_comp(vl_in, AA_1, projectors_Ty1, projectors_Ty1_larger, ind1, ind2)
        @tensor AA6a[:] := projectors_Ty1[ind1][-8, 1] * AA_1[-1, 3, -5, 1] *
            AA_1[-2, 4, -6, 3] * AA_1[-3, 2, -7, 4] * projectors_Ty1_larger[ind2]'[2, -4]
        @tensor vl_tmp[:] := vl_in[1, 2, 3, -6, -7, -8] * AA6a[1, 2, 3, -2, -3, -4, -5, -1]
        AA6a = nothing
        GC.gc(true)
        ipeps_reclaim_device_memory!(aggressive=true)
        @tensor AA6b[:] := projectors_Ty1_larger[ind2][-8, 1] * AA_1[-1, 3, -5, 1] *
            AA_1[-2, 4, -6, 3] * AA_1[-3, 2, -7, 4] * projectors_Ty1[ind1]'[2, -4]
        @tensor vl_out[:] := vl_tmp[1, 2, -1, -2, -3, 3, 4, 5] * AA6b[3, 4, 5, 1, -4, -5, -6, 2]
        AA6b = nothing
        vl_tmp = nothing

        es_synchronize()
        vl_cpu=to_es_cpu(vl_out);
        vl_out=nothing
        
        return vl_cpu
    end
    GC.gc(true)
    ipeps_reclaim_device_memory!(aggressive=true)
    vl_gpu=to_es_device(deepcopy(vl));
    vl_total=vl*0;
    for ind1 =length(projectors_Ty1):-1:1
        @show ind1
        t=@elapsed begin
            for ind2 =length(projectors_Ty1_larger):-1:1
                vl_comp=apply_comp(vl_gpu, AA_1, projectors_Ty1, projectors_Ty1_larger, ind1, ind2);
                vl_total=vl_total+vl_comp
                vl_comp=nothing
                GC.gc(true)
                ipeps_reclaim_device_memory!(aggressive=true)
            end
            CUDA.pool_status();flush(stdout)
        end
        println("time = ", t)
    end
    return vl_total
end
#apply_l_Ty1(vl0, AA_1, projectors_Ty1, projectors_Ty1_larger)
#######



#Ty0
projectors_Ty0 = projector_general_SU2(space(AA_0, 4); check=true);
projectors_Ty0_larger=group_SU2_projectors(projectors_Ty0; max_eff_dim=10, alpha=0.8, check=true);
projectors_Ty0 = to_es_device(projectors_Ty0);
projectors_Ty0_larger = to_es_device(projectors_Ty0_larger);
# @show length(projectors_Ty0)*length(projectors_Ty0_larger)
@show length(projectors_Ty0)
@show TensorKit.storagetype(AA_0)
@show TensorKit.storagetype(vl0)
#CUDA.pool_status()
ipeps_reclaim_device_memory!()




function  apply_l_Ty0(vl, AA_0, projectors_Ty0, projectors_Ty0_larger)
    function apply_comp(vl_in, AA_0, projectors_Ty0, projectors_Ty0_larger, ind1, ind2)
        @tensor AA6a[:] := projectors_Ty0[ind1][-8, 1] * AA_0[-1, 3, -5, 1] *
            AA_0[-2, 4, -6, 3] * AA_0[-3, 2, -7, 4] * projectors_Ty0_larger[ind2]'[2, -4]
        @tensor vl_tmp[:] := vl_in[1, 2, 3, -6, -7, -8] * AA6a[1, 2, 3, -2, -3, -4, -5, -1]
        AA6a = nothing
        GC.gc(true)
        ipeps_reclaim_device_memory!(aggressive=true)
        @tensor AA6b[:] := projectors_Ty0_larger[ind2][-8, 1] * AA_0[-1, 3, -5, 1] *
            AA_0[-2, 4, -6, 3] * AA_0[-3, 2, -7, 4] * projectors_Ty0[ind1]'[2, -4]
        @tensor vl_out[:] := vl_tmp[1, 2, -1, -2, -3, 3, 4, 5] * AA6b[3, 4, 5, 1, -4, -5, -6, 2]
        AA6b = nothing
        vl_tmp = nothing

        es_synchronize()
        vl_cpu=to_es_cpu(vl_out);
        vl_out=nothing
        
        return vl_cpu
    end
    GC.gc(true)
    ipeps_reclaim_device_memory!(aggressive=true)
    vl_gpu=to_es_device(deepcopy(vl));
    vl_total=vl*0;
    for ind1 =length(projectors_Ty0):-1:1
        @show ind1
        t=@elapsed begin
            for ind2 =length(projectors_Ty0_larger):-1:1
                vl_comp=apply_comp(vl_gpu, AA_0, projectors_Ty0, projectors_Ty0_larger, ind1, ind2);
                vl_total=vl_total+vl_comp
                vl_comp=nothing
                GC.gc(true)
                ipeps_reclaim_device_memory!(aggressive=true)
            end
            CUDA.pool_status();flush(stdout)
        end
        println("time = ", t)
    end
    return vl_total
end
#apply_l_Ty0(vl0, AA_0, projectors_Ty0, projectors_Ty0_larger)
#######

function apply_M_vl(AA_0,AA_1,AA_2,AA_3,AA_m2,AA_m1, vl)
    function apply_l(MM,ll)
        # @tensor ll[:]:=MM[1,2,-1,8]*MM[3,4,-2,2]*MM[5,6,-3,4]*MM[7,8,-4,6]*ll[1,3,5,7];
        ll0 = ll

        @tensor tmp1[-1, 2, 3, 5, 7, 8] :=MM[1, 2, -1, 8] * ll0[1, 3, 5, 7]

        @tensor tmp2[-1, -2, 4, 5, 7, 8] :=MM[3, 4, -2, 2] * tmp1[-1, 2, 3, 5, 7, 8]

        @tensor tmp3[-1, -2, -3, 6, 7, 8] :=MM[5, 6, -3, 4] * tmp2[-1, -2, 4, 5, 7, 8]

        @tensor ll_new[-1, -2, -3, -4] :=MM[7, 8, -4, 6] * tmp3[-1, -2, -3, 6, 7, 8]
        return ll_new
    end
            

    vl_out=apply_l(AA_1,vl);
    es_synchronize()
    return vl_out
end


function apply_M_vl_kA_projection(AA_0,AA_1,AA_2,AA_3,AA_m2,AA_m1, vl, kind,Nv)
    function apply_l(MM,ll)
        # @tensor ll[:]:=MM[1,2,-1,8]*MM[3,4,-2,2]*MM[5,6,-3,4]*MM[7,8,-4,6]*ll[1,3,5,7];
        ll0 = ll

        @tensor tmp1[-1, 2, 3, 5, 7, 8] :=MM[1, 2, -1, 8] * ll0[1, 3, 5, 7]

        @tensor tmp2[-1, -2, 4, 5, 7, 8] :=MM[3, 4, -2, 2] * tmp1[-1, 2, 3, 5, 7, 8]

        @tensor tmp3[-1, -2, -3, 6, 7, 8] :=MM[5, 6, -3, 4] * tmp2[-1, -2, 4, 5, 7, 8]

        @tensor ll_new[-1, -2, -3, -4] :=MM[7, 8, -4, 6] * tmp3[-1, -2, -3, 6, 7, 8]
        return ll_new
    end
            
    AA_set=(AA_0,AA_1,AA_2,AA_3,AA_m2,AA_m1);
    vl_out=apply_l(AA_0,vl);
    for cc=2:length(AA_set)
        vl_out=vl_out+apply_l(AA_set[cc],vl)*exp(im*kind*(cc-1)*2*pi/Nv);
    end
    vl_out=vl_out/Nv;

    es_synchronize()
    return vl_out
end

function apply_M_vr(AA_0,AA_1,AA_2,AA_3,AA_m2,AA_m1, vr)
    function apply_r(MM,rr)
        # @tensor rr[:]:=MM[-1,2,1,8]*MM[-2,4,3,2]*MM[-3,6,5,4]*MM[-4,8,7,6]*rr[1,3,5,7];

        rr0 = rr

        @tensor tmp1[-1, -2, -3, -4, -5, -6] :=
            MM[-1, -2, 1, -6] * rr0[1, -3, -4, -5]

        @tensor tmp2[-1, -2, -3, -4, -5, -6] :=
            MM[-2, -3, 1, 2] * tmp1[-1, 2, 1, -4, -5, -6]

        @tensor tmp3[-1, -2, -3, -4, -5, -6] :=
            MM[-3, -4, 1, 2] * tmp2[-1, -2, 2, 1, -5, -6]

        @tensor rr_new[-1, -2, -3, -4] :=
            MM[-4, 1, 2, 3] * tmp3[-1, -2, -3, 3, 2, 1]

        rr = rr_new
        return rr
    end

    vr_out=apply_r(AA_0,vr); #right half system should have no translation operation
    es_synchronize()
    return vr_out
end







#test1: only_AA1
# vl=deepcopy(vl0);
# for cc=1:20
#     vl=apply_M_vl(AA_0,AA_1,AA_2,AA_m1, vl);
# end
# vr=deepcopy(vr0);
# for cc=1:20
#     vr=apply_M_vr(AA_0,AA_1,AA_2,AA_m1, vr);
# end

# contraction_l_fun(x)=apply_M_vl(AA_0,AA_1,AA_2,AA_m1,x);
contraction_l_fun(x)=apply_M_vl_kA_projection(AA_0,AA_1,AA_2,AA_3,AA_m2,AA_m1,x,kind,Nv);
contraction_r_fun(x)=apply_M_vr(AA_0,AA_1,AA_2,AA_3,AA_m2,AA_m1,x);

@time contraction_l_fun(vl0);
@time contraction_r_fun(vr0);

#vals1, vecs1,info1 = eigsolve(hfun, AB, 1, :LM; tol=eigsolve_tol, krylovdim=eigsolve_krylovdim, maxiter=eigsolve_maxiter,eager=true)
@time eul,evl=eigsolve(contraction_l_fun, vl0, 1,:LM; tol=1e-5, krylovdim=10,eager=true);
es_synchronize()
@show eul
evl=evl[1];
jldsave("evl_kind"*string(kind)*"_Nv"*string(Nv);evl=to_es_cpu(evl),U_L=to_es_cpu(U_L));

@time eur,evr=eigsolve(contraction_r_fun, vr0, 1,:LM,Arnoldi(krylovdim=20));
es_synchronize()
@show eur
evr=evr[1];
jldsave("evr_Nv"*string(Nv);evr=to_es_cpu(evr),U_R=to_es_cpu(U_R));

@tensor vl_expand[:]:=evl[1,2,3,4]*U_R'[1,-1,-5]*U_R'[2,-2,-6]*U_R'[3,-3,-7]*U_R'[4,-4,-8];

@tensor vr_expand[:]:=evr[1,2,3,4]*U_L'[-1,-5,1]*U_L'[-2,-6,2]*U_L'[-3,-7,3]*U_L'[-4,-8,4];


@tensor rho[:]:=vl_expand[-1,-2,-3,-4,1,2,3,4]*vr_expand[-5,-6,-7,-8,1,2,3,4];
es_synchronize()
rho = to_es_cpu(rho)
rho=permute(rho,(1,2,3,4,),(5,6,7,8,));

jldsave("rho_kind"*string(kind)*"_Nv"*string(Nv);rho=rho);

eu,ev=eigen(rho);

@tensor km[:]:=ev'[-1,1,2,3,4]*ev[2,3,4,1,-2];
km=permute(km,(1,),(2,));
Sectors=ComplexF64[];
eu_set=ComplexF64[];
km_set=ComplexF64[];

global Sectors,eu_set,km_set
for (sec,dat) in blocks(eu)
    global Sectors,eu_set,km_set
    Sectors=vcat(Sectors, ones(length(diag(dat)))*Float64(sec.j));
    eu_set=vcat(eu_set, diag(dat));
    km_set=vcat(km_set,diag(block(km,sec)));

end

# filenm="ES_exact_Ty1_Nv"*string(Nv)*".mat";
#     matwrite(filenm, Dict(
#         "eu_set" => eu_set,
#         "Sectors" => Sectors
#     ); compress = false)


filenm="ES_exact_kA_"*string(kind)*"_Nv"*string(Nv)*".mat";
    matwrite(filenm, Dict(
        "eu_set" => eu_set,
        "km_set" => km_set,
        "Sectors" => Sectors
    ); compress = false)
