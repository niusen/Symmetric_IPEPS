function normalize_no_grad(T)
    Norm=norm(T);
    return T/Norm;
end
function ChainRulesCore.rrule(::typeof(normalize_no_grad), T::AbstractTensorMap)
    function normalize_no_grad_pushback(f̄wd)
        return NoTangent(), f̄wd
    end 
    return normalize_no_grad(T), normalize_no_grad_pushback
end


function show_grad(T)
    
    return T;
end
function ChainRulesCore.rrule(::typeof(show_grad), T::AbstractTensorMap)
    function show_grad_pushback(f̄wd)
        println("Grad of variable: "*string(norm(f̄wd)))
        return NoTangent(), f̄wd
    end 
    return show_grad(T), show_grad_pushback
end


function my_pinv(T::DiagonalTensorMap)
    epsilon0 = 1e-12
    epsilon=epsilon0*maximum(abs.(diag(convert(Array,T))))^2
    T_new=deepcopy(T);
    
    mm=T_new.data;
    mm = mm./(mm.^2 .+epsilon)
    T_new.data.=mm;

    return T_new
end

function my_pinv2(T::DiagonalTensorMap)
    epsilon0 = 1e-8
    epsilon=epsilon0*maximum(abs.(diag(convert(Array,T))))
    T_new=deepcopy(T);

    mm=T_new.data;
    mm = (1 ./mm).*(abs(mm).>epsilon)
    T_new.data.=mm;

    # if sectortype(space(T_new,1)) == Trivial
    #     mm=T_new.data;
    #     @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
    #     for i=1:size(mm,1)
    #         if abs(mm[i,i])>epsilon
    #             mm[i,i] = 1/mm[i,i]
    #         end
    #     end
    #     T_new=TensorMap(mm,codomain(T),domain(T));
    # else
    #     for cc=1:length(T_new.data.values)
    #         mm=T_new.data.values[cc];
    #         @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
    #         for i = 1:size(mm,1)
    #             if abs(mm[i,i])>epsilon
    #                 mm[i,i] = 1/mm[i,i]
    #             end
    #         end
    #         T_new.data.values[cc]=mm;
    #     end
    # end
    return T_new
end


# function my_tsvd(T::AbstractTensorMap; kwargs...)
#     (U,S,V) = tsvd(T;kwargs...);
#     return (U,S,V)
# end
# function ChainRulesCore.rrule(::typeof(my_tsvd), t::AbstractTensorMap;kwargs...)  #this is not good compared to TensorKit's new internal function
#     global multiplet_tol, backward_settings
#     #grad_inverse_tol=1e-8
#     #grad_regulation_epsilon=1e-12;
#     (U,S,V) = my_tsvd(t;kwargs...);#println(S.data.values)
#     epsilon1=maximum(diag(convert(Array,S)))*backward_settings.grad_inverse_tol;
#     epsilon2=maximum(diag(convert(Array,S)))*backward_settings.grad_regulation_epsilon;
#     Fp = TensorMap(randn,space(S,1),space(S,1));
#     for (k,dst) in blocks(Fp)
#         src = block(S,k)
#         @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
#             ff=0;
#             if abs(src[j,j]-src[i,i])>epsilon1
#                 if abs(abs(src[j,j])/abs(src[i,i])-1)>multiplet_tol #relative difference is big
#                     ff=ff + (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon2)
#                 end
#             end
#             if src[j,j] + src[i,i]>epsilon1
#                 ff=ff + (src[j,j] + src[i,i])/((src[j,j] + src[i,i])^2+epsilon2)
#             end
#             dst[i,j] = (i == j) ? zero(eltype(S)) : ff
#         end
#     end

#     Fm = TensorMap(randn,space(S,1),space(S,1));;
#     for (k,dst) in blocks(Fm)
#         src = block(S,k)
#         @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
#             ff=0;
#             if abs(src[j,j]-src[i,i])>epsilon1
#                 if abs(abs(src[j,j])/abs(src[i,i])-1)>multiplet_tol #relative difference is big
#                     ff=ff + (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon2)
#                 end
#             end
#             if src[j,j] + src[i,i]>epsilon1
#                 ff=ff - (src[j,j] + src[i,i])/((src[j,j] + src[i,i])^2+epsilon2)
#             end
#             dst[i,j] = (i == j) ? zero(eltype(S)) : ff
#         end
#     end
#     #jldsave("svd.jld2"; U,S,V)

#     function pullback(v)
#         dU,dS,dV = v
#         #jldsave("svd_backward.jld2"; dU,dS,dV)
#         #println("Norm of dU, dS, dV: "*string([norm(dU),norm(dS),norm(dV)]))

#         dA = zero(t);
#         #A_s bar term
#         if dS != ChainRulesCore.ZeroTangent()
#             if isa(dS, Thunk)
#                 dS=unthunk(dS)
#             end
#             dA += U*_elementwise_mult(dS,one(dS))*V
#         end
#         #A_uo bar term
#         if dU != ChainRulesCore.ZeroTangent()
#             Jp = _elementwise_mult((U'*dU),Fp) - _elementwise_mult(dU'*U,Fp)
            
#             dA += U*(Jp)*V/2
#         end
#         #A_vo bar term
#         if dV != ChainRulesCore.ZeroTangent()
#             VpdV = V*dV';
#             Km = _elementwise_mult(VpdV,Fm) - _elementwise_mult(dV*(V'),Fm)
#             dA += U*(Km)*V/2
#         end

#         # ####!!!  In my test without such term is more accurate.
#         # #A_d bar term, only relevant if matrix is complex
#         # if dV != ChainRulesCore.ZeroTangent() && eltype(V) <: Complex
#         #     L = _elementwise_mult(V*dV',one(Fm))
#         #     dA += 1/2*U*my_pinv(S)*(L' - L)*V
#         # end

#         if codomain(t)!=domain(t)
#             pru = U*U';
#             prv = V'*V;
#             dA += (one(pru)-pru)*dU*my_pinv(S)*V
#             dA += U*my_pinv(S)*dV*(one(prv)-prv)
#         end

#         if backward_settings.show_ite_grad_norm
#             println("Norm of dA: "*string(norm(dA)))
#         end
#         global grad_norm
#         grad_norm=deepcopy(norm(dA));
#         return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
#     end
#     return (U,S,V), pullback
# end



# #try new functions:


# function ChainRulesCore.rrule(::typeof(TensorKit.tsvd!), t::AbstractTensorMap;
#                               trunc::TensorKit.TruncationScheme=TensorKit.NoTruncation(),
#                               p::Real=2,
#                               alg::Union{TensorKit.SVD,TensorKit.SDD}=TensorKit.SDD())
#     U, Σ, V⁺, truncerr = tsvd(t; trunc=TensorKit.NoTruncation(), p=p, alg=alg)

#     if !(trunc isa TensorKit.NoTruncation) && !isempty(blocksectors(t))
#         Σdata = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(Σ))

#         truncdim = TensorKit._compute_truncdim(Σdata, trunc, p)
#         truncerr = TensorKit._compute_truncerr(Σdata, truncdim, p)

#         SVDdata = TensorKit.SectorDict(c => (block(U, c), Σc, block(V⁺, c))
#                                        for (c, Σc) in Σdata)

#         Ũ, Σ̃, Ṽ⁺ = TensorKit._create_svdtensors(t, SVDdata, truncdim)
#     else
#         Ũ, Σ̃, Ṽ⁺ = U, Σ, V⁺
#     end

#     tol0=1e-12;
#     tol_global=maximum(real.(Σ.data))^2*tol0;

#     function tsvd!_pullback(ΔUSVϵ)
#         ΔU, ΔΣ, ΔV⁺, = unthunk.(ΔUSVϵ)
#         Δt = similar(t)
#         for (c, b) in blocks(Δt)
#             Uc, Σc, V⁺c = block(U, c), block(Σ, c), block(V⁺, c)
#             ΔUc, ΔΣc, ΔV⁺c = block(ΔU, c), block(ΔΣ, c), block(ΔV⁺, c)
#             Σdc = view(Σc, diagind(Σc))
#             ΔΣdc = (ΔΣc isa AbstractZero) ? ΔΣc : view(ΔΣc, diagind(ΔΣc))
#             svd_pullback!(b, Uc, Σdc, V⁺c, ΔUc, ΔΣdc, ΔV⁺c, tol_global)
#         end
#         return NoTangent(), Δt
#     end
#     function tsvd!_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
#         return NoTangent(), ZeroTangent()
#     end

#     return (Ũ, Σ̃, Ṽ⁺, truncerr), tsvd!_pullback
# end

# function svd_pullback!(ΔA::AbstractMatrix, U::AbstractMatrix, S::AbstractVector,
#                        Vd::AbstractMatrix, ΔU, ΔS, ΔVd,
#                        tol::Real)

#     # Basic size checks and determination
#     m, n = size(U, 1), size(Vd, 2)
#     size(U, 2) == size(Vd, 1) == length(S) == min(m, n) || throw(DimensionMismatch())
#     p = -1
#     if !(ΔU isa AbstractZero)
#         m == size(ΔU, 1) || throw(DimensionMismatch())
#         p = size(ΔU, 2)
#     end
#     if !(ΔVd isa AbstractZero)
#         n == size(ΔVd, 2) || throw(DimensionMismatch())
#         if p == -1
#             p = size(ΔVd, 1)
#         else
#             p == size(ΔVd, 1) || throw(DimensionMismatch())
#         end
#     end
#     if !(ΔS isa AbstractZero)
#         if p == -1
#             p = length(ΔS)
#         else
#             p == length(ΔS) || throw(DimensionMismatch())
#         end
#     end
#     Up = view(U, :, 1:p)
#     Vp = view(Vd, 1:p, :)'
#     Sp = view(S, 1:p)

#     # rank
#     r = searchsortedlast(S, tol; rev=true)

#     # compute antihermitian part of projection of ΔU and ΔV onto U and V
#     # also already subtract this projection from ΔU and ΔV
#     if !(ΔU isa AbstractZero)
#         UΔU = Up' * ΔU
#         aUΔU = rmul!(UΔU - UΔU', 1 / 2)
#         if m > p
#             ΔU -= Up * UΔU
#         end
#     else
#         aUΔU = fill!(similar(U, (p, p)), 0)
#     end
#     if !(ΔVd isa AbstractZero)
#         VΔV = Vp' * ΔVd'
#         aVΔV = rmul!(VΔV - VΔV', 1 / 2)
#         if n > p
#             ΔVd -= VΔV' * Vp'
#         end
#     else
#         aVΔV = fill!(similar(Vd, (p, p)), 0)
#     end

#     # check whether cotangents arise from gauge-invariance objective function
#     mask = abs.(Sp' .- Sp) .< tol
#     Δgauge = norm(view(aUΔU, mask) + view(aVΔV, mask), Inf)
#     if p > r
#         rprange = (r + 1):p
#         Δgauge = max(Δgauge, norm(view(aUΔU, rprange, rprange), Inf))
#         Δgauge = max(Δgauge, norm(view(aVΔV, rprange, rprange), Inf))
#     end
#     Δgauge < tol ||
#         @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

#     UdΔAV = (aUΔU .+ aVΔV) .* safe_inv.(Sp' .- Sp, tol) .+
#             (aUΔU .- aVΔV) .* safe_inv.(Sp' .+ Sp, tol)
#     if !(ΔS isa ZeroTangent)
#         UdΔAV[diagind(UdΔAV)] .+= real.(ΔS)
#         # in principle, ΔS is real, but maybe not if coming from an anyonic tensor
#     end
#     mul!(ΔA, Up, UdΔAV * Vp')

#     if r > p # contribution from truncation
#         Ur = view(U, :, (p + 1):r)
#         Vr = view(Vd, (p + 1):r, :)'
#         Sr = view(S, (p + 1):r)

#         if !(ΔU isa AbstractZero)
#             UrΔU = Ur' * ΔU
#             if m > r
#                 ΔU -= Ur * UrΔU # subtract this part from ΔU
#             end
#         else
#             UrΔU = fill!(similar(U, (r - p, p)), 0)
#         end
#         if !(ΔVd isa AbstractZero)
#             VrΔV = Vr' * ΔVd'
#             if n > r
#                 ΔVd -= VrΔV' * Vr' # subtract this part from ΔV
#             end
#         else
#             VrΔV = fill!(similar(Vd, (r - p, p)), 0)
#         end

#         X = (1 // 2) .* ((UrΔU .+ VrΔV) .* safe_inv.(Sp' .- Sr, tol) .+
#                          (UrΔU .- VrΔV) .* safe_inv.(Sp' .+ Sr, tol))
#         Y = (1 // 2) .* ((UrΔU .+ VrΔV) .* safe_inv.(Sp' .- Sr, tol) .-
#                          (UrΔU .- VrΔV) .* safe_inv.(Sp' .+ Sr, tol))

#         # ΔA += Ur * X * Vp' + Up * Y' * Vr'
#         mul!(ΔA, Ur, X * Vp', 1, 1)
#         mul!(ΔA, Up * Y', Vr', 1, 1)
#     end

#     if m > max(r, p) && !(ΔU isa AbstractZero) # remaining ΔU is already orthogonal to U[:,1:max(p,r)]
#         # ΔA += (ΔU .* safe_inv.(Sp', tol)) * Vp'
#         mul!(ΔA, ΔU .* safe_inv.(Sp', tol), Vp', 1, 1)
#     end
#     if n > max(r, p) && !(ΔVd isa AbstractZero) # remaining ΔV is already orthogonal to V[:,1:max(p,r)]
#         # ΔA += U * (safe_inv.(Sp, tol) .* ΔVd)
#         mul!(ΔA, Up, safe_inv.(Sp, tol) .* ΔVd, 1, 1)
#     end
#     return ΔA
# end



# function safe_inv(x, eps_abs)
#     return x / (x*2 + eps_abs)
# end

# #safe_inv(a, tol) = abs(a) < tol ? zero(a) : inv(a)