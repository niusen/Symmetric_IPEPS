using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using LinearAlgebra, OptimKit
using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives


cd(@__DIR__)
include("resource_codes\\AD_lib.jl")


Random.seed!(12345)

a=load("hard_tensor.jld2");
M1=a["M"];
M2=TensorMap(randn,space(M1,1)*space(M1,2),space(M1,1)*space(M1,2));
M3=TensorMap(randn,space(M1,1)*space(M1,2),space(M1,1)*space(M1,2));


# M1=TensorMap(randn,ℂ^10*ℂ^10,ℂ^10*ℂ^10);
# M2=TensorMap(randn,ℂ^10*ℂ^10,ℂ^10*ℂ^10);
# M1=M1*M1*M1*M1;
# M2=M2*M2*M2*M2;
M1=M1/norm(M1);
M2=M2/norm(M2);

global chi=20;


function fun(x)
    global chi,M1,M2,M3
    M=M1*x[1]*M2*x[2];
    #M=@ignore_derivatives M/norm(M);
    uM,sM,vM = my_tsvd(M; trunc=truncdim(chi));
    
    #uM,sM,vM = tsvd(M);
    println(diag(convert(Array,sM)))

    #E_=uM*sM*sM*vM;
    E=@tensor uM[6,7,3]*sM[3,4]*sM[4,5]*vM[5,1,2]*M3[1,2,6,7];
    E=real(E);
    println(E)
    return E
end


function cfun(state_vec)
    
    ∂E = fun'(state_vec)
    #E=fun(state_vec)
    
    @assert !isnan(norm(∂E))
    
    return ∂E
end


state_vec=[1,1.1];
∂E=cfun(state_vec);
println(∂E)



dt=0.00001

E0=fun(state_vec);

grad=Vector{Float64}(undef,0);

for cc=1:length(state_vec)
    state_vec_tem=deepcopy(state_vec);
    state_vec_tem[cc]=state_vec_tem[cc]+dt;
    grad=vcat(grad,(fun(state_vec_tem)-E0)/dt);

end
println(grad)
println(∂E./grad)



# init=initial_condition(init_type="PBC", reconstruct=true, has_AA_fused=false);

# CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,[],ctm_setting,optim_setting)
# E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting1.kagome_method);

# function cfun(x)
#     (ψ,env) = x

#     function fun(peps)
#         env = leading_boundary(peps, alg_ctm, env)
#         x = H_expectation_value(peps, env, H)   
#         return x
#     end

#     ∂E = fun'(ψ)
#     env = leading_boundary(ψ, alg_ctm, env)
#     E = H_expectation_value(ψ, env, H)

#     @assert !isnan(norm(∂E))
#     return E,∂E
# end

# # my_retract is not an in place function which should not change x
# function my_retract(x,dx,α::Number)
#     (ϕ,env0) = x
#     ψ = deepcopy(ϕ)
#     env = deepcopy(env0)
#     ψ.A .+= dx.A .* α
#     #env = leading_boundary(ψ, alg_ctm,env)
#     return (ψ,env),dx
# end

# my_inner(x,dx1,dx2) = real(dot(dx1,dx2))

# function my_add!(Y, X, a)
#     Y.A .+= X.A .* a
#     return Y
# end

# function my_scale!(η, β)
#     rmul!(η.A, β)
#     return η
# end


# function init_psi(d::Int, D::Int, Lx::Int, Ly::Int)
#     Pspaces = fill(ℂ^d,Lx,Ly)
#     Nspaces = fill(ℂ^D,Lx,Ly)
#     Espaces = fill(ℂ^D,Lx,Ly)

#     Sspaces = adjoint.(circshift(Nspaces, (1, 0)))
#     Wspaces = adjoint.(circshift(Espaces, (0, -1)))

#     A = map(Pspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
#         return TensorMap(rand, ComplexF64, P ← N * E * S * W)
#     end

#     return InfinitePEPS(A)
# end


# alg_ctm = CTMRG(
#             verbose=10000,
#             tol=1e-10,
#             trscheme=truncdim(10),
#             miniter=4,
#             maxiter=200
#         )

# function main(;d=2,D=2,Lx=1,Ly=1)
#     ψ = init_psi(d,D,Lx,Ly)   
#     env = leading_boundary(ψ, alg_ctm) 
#     optimize(
#         cfun, 
#         (ψ,env),
#         ConjugateGradient(verbosity=3); 
#         inner=my_inner,
#         retract=my_retract,
#         scale! = my_scale!,
#         add! = my_add!
#     )
#     return ψ
# end

# main()
