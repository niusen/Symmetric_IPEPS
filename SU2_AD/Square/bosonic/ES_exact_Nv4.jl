using TensorKit, Zygote
using LinearAlgebra: I, diagm, norm
import LinearAlgebra.BLAS as BLAS
using JLD2, ChainRulesCore, MAT
using KrylovKit
using Random
using Zygote: @ignore_derivatives

cd(@__DIR__)

include("ES_auxiliary_Ty_shift_builders.jl");
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_chiral_pair_model.jl"))

# ES_algorithms.jl provides shared helpers such as vison_op, k_projection,
# calculate_k, and CTM_T_action. The explicit version avoids global U_L/U_R.
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "mps_algorithms", "ES_algorithms.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "mps_algorithms", "ES_algorithms_explicit.jl"))

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


@show Nv = 4
@show kind=0  #momentum projector for A subsystem

EH_n = 30


# AA, U_L, U_D, U_R, U_U = build_double_layer(A);

AA_0, U_L, U_D, U_R, U_U=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',A);

Ty1=chiral_pair_Ty1_mpo();
Tym1=chiral_pair_Tym1_mpo();
Ty2=compose_chiral_pair_shift_mpo(Ty1, Ty1);

AA_1, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Ty1));

AA_m1, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Tym1));

AA_2, _=build_double_layer_NoSwap(permute(A,(1,2,3,4,5,))',apply_mpo_to_peps(A,Ty2));




Ve=space(AA_0,1)';
vl0=randn(ComplexF64,Ve*Ve*Ve*Ve);

vr0=randn(ComplexF64,Ve'*Ve'*Ve'*Ve');

function apply_M_vl(AA_0,AA_1,AA_2,AA_m1, vl)
    function apply_l(MM,ll)
        # @tensor ll[:]:=MM[1,2,-1,8]*MM[3,4,-2,2]*MM[5,6,-3,4]*MM[7,8,-4,6]*ll[1,3,5,7];
        ll0 = ll

        @tensor tmp1[-1, 2, 3, 5, 7, 8] :=MM[1, 2, -1, 8] * ll0[1, 3, 5, 7]

        @tensor tmp2[-1, -2, 4, 5, 7, 8] :=MM[3, 4, -2, 2] * tmp1[-1, 2, 3, 5, 7, 8]

        @tensor tmp3[-1, -2, -3, 6, 7, 8] :=MM[5, 6, -3, 4] * tmp2[-1, -2, 4, 5, 7, 8]

        @tensor ll_new[-1, -2, -3, -4] :=MM[7, 8, -4, 6] * tmp3[-1, -2, -3, 6, 7, 8]
        return ll_new
    end
            

    vl_out=apply_l(AA_0,vl);
    println("M*vl")
    flush(stdout);
    return vl_out
end



function apply_M_vr(AA_0,AA_1,AA_2,AA_m1, vr)
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
    println("M*vr")
    flush(stdout);
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

contraction_l_fun(x)=apply_M_vl(AA_0,AA_1,AA_2,AA_m1,x);
contraction_r_fun(x)=apply_M_vr(AA_0,AA_1,AA_2,AA_m1,x);

@time contraction_l_fun(vl0);
@time contraction_r_fun(vr0);

#vals1, vecs1,info1 = eigsolve(hfun, AB, 1, :LM; tol=eigsolve_tol, krylovdim=eigsolve_krylovdim, maxiter=eigsolve_maxiter,eager=true)
@time eul,evl=eigsolve(contraction_l_fun, vl0, 1,:LM,Arnoldi(krylovdim=20));
@show eul
evl=evl[1];


@time eur,evr=eigsolve(contraction_r_fun, vr0, 1,:LM,Arnoldi(krylovdim=20));
@show eur
evr=evr[1];


@tensor vl_expand[:]:=evl[1,2,3,4]*U_R'[1,-1,-5]*U_R'[2,-2,-6]*U_R'[3,-3,-7]*U_R'[4,-4,-8];

@tensor vr_expand[:]:=evr[1,2,3,4]*U_L'[-1,-5,1]*U_L'[-2,-6,2]*U_L'[-3,-7,3]*U_L'[-4,-8,4];


@tensor rho[:]:=vl_expand[-1,-2,-3,-4,1,2,3,4]*vr_expand[-5,-6,-7,-8,1,2,3,4];
rho=permute(rho,(1,2,3,4,),(5,6,7,8,));



eu,ev=eigen(rho);

@tensor km[:]:=ev'[-1,1,2,3,4]*ev[2,3,4,1,-2];
km=permute(km,(1,),(2,));
Sectors=ComplexF64[];
eu_set=ComplexF64[];
km_set=ComplexF64[];

for (sec,dat) in blocks(eu)
    Sectors=vcat(Sectors, ones(length(diag(dat)))*Float64(sec.j));
    eu_set=vcat(eu_set, diag(dat));
    km_set=vcat(km_set,diag(block(km,sec)));

end

# filenm="ES_exact_Ty1_Nv"*string(Nv)*".mat";
#     matwrite(filenm, Dict(
#         "eu_set" => eu_set,
#         "Sectors" => Sectors
#     ); compress = false)


filenm="ES_exact_"*"_Nv"*string(Nv)*".mat";
    matwrite(filenm, Dict(
        "eu_set" => eu_set,
        "km_set" => km_set,
        "Sectors" => Sectors
    ); compress = false)