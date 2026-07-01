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


@show Nv = 5
@show kind=0  #momentum projector for A subsystem




# AA, U_L, U_D, U_R, U_U = build_double_layer(A);

AA_0, U_L, U_D, U_R, U_U=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',A);

Ty1=chiral_pair_Ty1_mpo();
Tym1=chiral_pair_Tym1_mpo();
Ty2=compose_chiral_pair_shift_mpo(Ty1, Ty1);
# Ty3=compose_chiral_pair_shift_mpo(Ty2, Ty1);
Tym2=compose_chiral_pair_shift_mpo(Tym1, Tym1);

AA_1, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Ty1));
AA_m1, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Tym1));
AA_2, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Ty2));
AA_m2, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Tym2));
# AA_3, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Ty3));


Ve=space(AA_0,1)';
vl0=randn(ComplexF64,Ve*Ve*Ve*Ve*Ve);

vr0=randn(ComplexF64,Ve'*Ve'*Ve'*Ve'*Ve');

AA_0 = to_es_device(AA_0)
AA_1 = to_es_device(AA_1)
AA_2 = to_es_device(AA_2)
# AA_3 = to_es_device(AA_3)
AA_m2 = to_es_device(AA_m2)
AA_m1 = to_es_device(AA_m1)
U_L = to_es_device(U_L)
U_R = to_es_device(U_R)

es_synchronize()





# projectors = projector_general_SU2(space(AA_3, 4); check=true);
# projectors = to_es_device(projectors);
##############


function apply_l_comp(vl,AA__,projectors,ind1)
    # @tensor vl_comp[:]:=vl[3,5,7,9,11]*projectors[ind1][12,1]*AA__[3,4,-1,1]*AA__[5,6,-2,4]*AA__[7,8,-3,6]*AA__[9,10,-4,8]*AA__[11,2,-5,10]*projectors[ind1]'[2,12];
    @tensor tmp1[-1, 4, 5, 7, 9, 11, 12] := vl[3, 5, 7, 9, 11] *
        projectors[ind1][12, 1] * AA__[3, 4, -1, 1]

    @tensor tmp2[-1, -2, 6, 7, 9, 11, 12] := tmp1[-1, 4, 5, 7, 9, 11, 12] *
        AA__[5, 6, -2, 4]
    tmp1 = nothing
    # GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)
    @tensor tmp3[-1, -2, -3, 8, 9, 11, 12] := tmp2[-1, -2, 6, 7, 9, 11, 12] *
        AA__[7, 8, -3, 6]
    tmp2 = nothing
    GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)
    @tensor tmp4[-1, -2, -3, -4, 10, 11, 12] := tmp3[-1, -2, -3, 8, 9, 11, 12] *
        AA__[9, 10, -4, 8]
    tmp3 = nothing
    # GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)
    @tensor tmp5[-1, -2, -3, -4, -5, 2, 12] := tmp4[-1, -2, -3, -4, 10, 11, 12] *
        AA__[11, 2, -5, 10]
    tmp4 = nothing
    # GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)
    @tensor vl_comp[-1, -2, -3, -4, -5] := tmp5[-1, -2, -3, -4, -5, 2, 12] *
        projectors[ind1]'[2, 12]
    tmp5 = nothing
    # GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)

    es_synchronize()
    vl_cpu=to_es_cpu(vl_comp);
    vl_comp=nothing
    
    return vl_cpu
end

function apply_r_comp(vr, AA__, projectors, ind1)
    # @tensor vr_comp[:]:=projectors[ind1][12,1] *
    #     AA__[-1,4,3,1] * AA__[-2,6,5,4] *
    #     AA__[-3,8,7,6] * AA__[-4,10,9,8] *
    #     AA__[-5,2,11,10] * projectors[ind1]'[2,12] *
    #     vr_in[3,5,7,9,11];

    @tensor tmp1[-1, 4, 5, 7, 9, 11, 12] := projectors[ind1][12, 1] *
        AA__[-1, 4, 3, 1] * vr[3, 5, 7, 9, 11]

    @tensor tmp2[-1, -2, 6, 7, 9, 11, 12] := tmp1[-1, 4, 5, 7, 9, 11, 12] *
        AA__[-2, 6, 5, 4]
    tmp1 = nothing
    # GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)

    @tensor tmp3[-1, -2, -3, 8, 9, 11, 12] := tmp2[-1, -2, 6, 7, 9, 11, 12] *
        AA__[-3, 8, 7, 6]
    tmp2 = nothing
    GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)

    @tensor tmp4[-1, -2, -3, -4, 10, 11, 12] := tmp3[-1, -2, -3, 8, 9, 11, 12] *
        AA__[-4, 10, 9, 8]
    tmp3 = nothing
    # GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)

    @tensor tmp5[-1, -2, -3, -4, -5, 2, 12] := tmp4[-1, -2, -3, -4, 10, 11, 12] *
        AA__[-5, 2, 11, 10]
    tmp4 = nothing
    # GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)

    @tensor vr_comp[-1, -2, -3, -4, -5] := tmp5[-1, -2, -3, -4, -5, 2, 12] *
        projectors[ind1]'[2, 12]
    tmp5 = nothing
    # GC.gc(true)
    # ipeps_reclaim_device_memory!(aggressive=true)

    es_synchronize()
    vr_cpu=to_es_cpu(vr_comp);
    vr_comp=nothing
    
    return vr_cpu
end

# ind1=20;
# AA__=AA_3;

# vl=to_es_device(vl0);
# GC.gc(true)
# ipeps_reclaim_device_memory!(aggressive=true)
# apply_l_comp(vl,AA__,projectors,ind1)

# vr=to_es_device(vr0);
# GC.gc(true)
# ipeps_reclaim_device_memory!(aggressive=true)
# apply_r_comp(vr,AA__,projectors,ind1)
##############

function  apply_l_Tyn(vl, AA_n, projectors)
    GC.gc(true)
    ipeps_reclaim_device_memory!(aggressive=true)
    vl_gpu=to_es_device(deepcopy(vl));
    vl_total=vl*0;
    for ind1 =1:length(projectors)
        @show ind1
        t=@elapsed begin
            vl_comp=apply_l_comp(vl_gpu, AA_n, projectors, ind1);
            vl_total=vl_total+vl_comp
            vl_comp=nothing
            GC.gc(true)
            ipeps_reclaim_device_memory!(aggressive=true)
            CUDA.pool_status();flush(stdout)
        end
        println("time = ", t)
    end
    return vl_total
end

function  apply_r_Tyn(vr, AA_n, projectors)
    GC.gc(true)
    ipeps_reclaim_device_memory!(aggressive=true)
    vr_gpu=to_es_device(deepcopy(vr));
    vr_total=vr*0;
    for ind1 =1:length(projectors)
        @show ind1
        t=@elapsed begin
            vr_comp=apply_r_comp(vr_gpu, AA_n, projectors, ind1);
            vr_total=vr_total+vr_comp
            vr_comp=nothing
            GC.gc(true)
            ipeps_reclaim_device_memory!(aggressive=true)
            CUDA.pool_status();flush(stdout)
        end
        println("time = ", t)
    end
    return vr_total
end
##################




#########
# #Ty3
# projectors_Ty3 = projector_general_SU2(space(AA_3, 4); check=true);
# projectors_Ty3 = to_es_device(projectors_Ty3);
# # @show length(projectors_Ty3)*length(projectors_Ty3_larger)
# @show length(projectors_Ty3)
# #CUDA.pool_status()
# ipeps_reclaim_device_memory!()

# println("apply Ty3");
# apply_l_Tyn(vl0, AA_3, projectors_Ty3)
##############

#Ty2
projectors_Ty2 = projector_general_SU2(space(AA_2, 4); check=true);
projectors_Ty2 = to_es_device(projectors_Ty2);
# @show length(projectors_Ty2)*length(projectors_Ty2_larger)
@show length(projectors_Ty2)
#CUDA.pool_status()
ipeps_reclaim_device_memory!()

# println("apply Ty2");
# apply_l_Tyn(vl0, AA_2, projectors_Ty2)
######


#Ty1
projectors_Ty1 = projector_general_SU2(space(AA_1, 4); check=true);
projectors_Ty1 = to_es_device(projectors_Ty1);
# @show length(projectors_Ty1)*length(projectors_Ty1_larger)
@show length(projectors_Ty1)
#CUDA.pool_status()
ipeps_reclaim_device_memory!()

# println("apply Ty1");
# apply_l_Tyn(vl0, AA_1, projectors_Ty1)
#######



#Ty0
projectors_Ty0 = projector_general_SU2(space(AA_0, 4); check=true);
projectors_Ty0 = to_es_device(projectors_Ty0);
# @show length(projectors_Ty0)*length(projectors_Ty0_larger)
@show length(projectors_Ty0)
#CUDA.pool_status()
ipeps_reclaim_device_memory!()

# println("apply Ty0");
# apply_l_Tyn(vl0, AA_0, projectors_Ty0)
#######

##############

#Tym1
projectors_Tym1 = projector_general_SU2(space(AA_m1, 4); check=true);
projectors_Tym1 = to_es_device(projectors_Tym1);
# @show length(projectors_Tym1)*length(projectors_Tym1_larger)
@show length(projectors_Tym1)
#CUDA.pool_status()
ipeps_reclaim_device_memory!()

# println("apply Tym1");
# apply_l_Tyn(vl0, AA_m1, projectors_Tym1)

##############
#Tym2
projectors_Tym2 = projector_general_SU2(space(AA_m2, 4); check=true);
projectors_Tym2 = to_es_device(projectors_Tym2);
# @show length(projectors_Tym2)*length(projectors_Tym2_larger)
@show length(projectors_Tym2)
#CUDA.pool_status()
ipeps_reclaim_device_memory!()

# println("apply Tym2");
# apply_l_Tyn(vl0, AA_m2, projectors_Tym2)
#############

projectors_set=(projectors_Ty0,projectors_Ty1,projectors_Ty2,projectors_Tym2,projectors_Tym1);
AA_set=(AA_0,AA_1,AA_2,AA_m2,AA_m1);

function apply_M_vl_kA_projection(AA_set, projectors_set, vl, kind,Nv) 
    println("apply Ty0");     
    vl_out=apply_l_Tyn(vl, AA_set[1], projectors_set[1]);
    for cc=2:length(AA_set)
        println("apply Ty"*string(cc-1)); 
        vl_out=vl_out+apply_l_Tyn(vl, AA_set[cc], projectors_set[cc])*exp(im*kind*(cc-1)*2*pi/Nv);
    end
    vl_out=vl_out/Nv;

    es_synchronize()
    println("finished one Mvl")
    return vl_out
end

function apply_M_vr(AA_0, projectors_Ty0, vr)
    vr_out=apply_r_Tyn(vr, AA_0, projectors_Ty0);#right half system should have no translation operation
    es_synchronize()
    println("finished one Mvr")
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
contraction_l_fun(x)=apply_M_vl_kA_projection(AA_set, projectors_set, x, kind,Nv);
contraction_r_fun(x)=apply_M_vr(AA_0, projectors_Ty0, x);

# @time contraction_l_fun(vl0);
# @time contraction_r_fun(vr0);

#vals1, vecs1,info1 = eigsolve(hfun, AB, 1, :LM; tol=eigsolve_tol, krylovdim=eigsolve_krylovdim, maxiter=eigsolve_maxiter,eager=true)
println("left fixed point")
@time eul,evl=eigsolve(contraction_l_fun, vl0, 1,:LM; tol=1e-5, krylovdim=10,eager=true);
es_synchronize()
@show eul
evl=evl[1];
jldsave("evl_kind"*string(kind)*"_Nv"*string(Nv);evl=to_es_cpu(evl),U_L=to_es_cpu(U_L));

println("right fixed point")
@time eur,evr=eigsolve(contraction_r_fun, vr0, 1,:LM,Arnoldi(krylovdim=20));
es_synchronize()
@show eur
evr=evr[1];
jldsave("evr_Nv"*string(Nv);evr=to_es_cpu(evr),U_R=to_es_cpu(U_R));

@tensor vl_expand[:]:=evl[1,2,3,4,5]*U_R'[1,-1,-6]*U_R'[2,-2,-7]*U_R'[3,-3,-8]*U_R'[4,-4,-9]*U_R'[5,-5,-10];

@tensor vr_expand[:]:=evr[1,2,3,4,5]*U_L'[-1,-6,1]*U_L'[-2,-7,2]*U_L'[-3,-8,3]*U_L'[-4,-9,4]*U_L'[-5,-10,5];


@tensor rho[:]:=vl_expand[-1,-2,-3,-4,-5,1,2,3,4,5]*vr_expand[-6,-7,-8,-9,-10,1,2,3,4,5];
es_synchronize()
rho = to_es_cpu(rho)
rho=permute(rho,(1,2,3,4,5,),(6,7,8,9,10,));

jldsave("rho_kind"*string(kind)*"_Nv"*string(Nv);rho=rho);

eu,ev=eigen(rho);

@tensor km[:]:=ev'[-1,1,2,3,4,5]*ev[2,3,4,5,1,-2];
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
