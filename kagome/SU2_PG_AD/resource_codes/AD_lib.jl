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


function my_pinv(T,trun)
    T_new=deepcopy(T);
    t_max=maximum(diag(convert(Array,T)));
    for (k,dst) in blocks(T_new)
        src = blocks(T_new)[k]
        @inbounds for i in 1:size(dst,1)
            if dst[i,i]/t_max>trun
                dst[i,i] = 1/dst[i,i]
            else
                dst[i,i] = dst[i,i]*0
            end
        end
    end
    return T_new
end



function my_tsvd(T::AbstractTensorMap; kwargs...)
    (U,S,V) = tsvd(T;kwargs...);
    return (U,S,V)
end
function ChainRulesCore.rrule(::typeof(my_tsvd), t::AbstractTensorMap;kwargs...)
    T = eltype(t);
    global chi,multiplet_tol,projector_trun_tol
    (U,S,V) = my_tsvd(t;kwargs...);#println(S.data.values)
    epsilon = 1e-12

    Fp = similar(S);
    for (k,dst) in blocks(Fp)
        src = blocks(S)[k]
        @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
            if (src[j,j] + src[i,i])<projector_trun_tol
                ff = (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon)
            else
                ff = (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon)+1/(src[j,j] + src[i,i])
            end
            dst[i,j] = (i == j) ? zero(eltype(S)) : ff
        end
    end

    Fm = similar(S);
    for (k,dst) in blocks(Fm)
        src = blocks(S)[k]
        @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
            if (src[j,j] + src[i,i])<projector_trun_tol
                ff = (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon)
            else
                ff = (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon)-1/(src[j,j] + src[i,i])
            end
            dst[i,j] = (i == j) ? zero(eltype(S)) : ff
        end
    end


    function pullback(v)
        dU,dS,dV = v
        println("Norm of dU, dS, dV: "*string([norm(dU),norm(dS),norm(dV)]))

        dA = zero(t);
        #A_s bar term
        if dS != ChainRulesCore.ZeroTangent()
            dA += U*_elementwise_mult(dS,one(dS))*V
        end
        #A_uo bar term
        if dU != ChainRulesCore.ZeroTangent()
            Jp = _elementwise_mult((U'*dU),Fp) - _elementwise_mult(dU'*U,Fp)
            
            dA += U*(Jp)*V/2
        end
        #A_vo bar term
        if dV != ChainRulesCore.ZeroTangent()
            VpdV = V*dV';
            Km = _elementwise_mult(VpdV,Fm) - _elementwise_mult(dV*(V'),Fm)
            dA += U*(Km)*V/2
        end

        ####!!! I don't know why below term exist in TensorKitAD. I didn't find below term in the formulas. In my test without such term is more accurate.
        # #A_d bar term, only relevant if matrix is complex
        # if dV != ChainRulesCore.ZeroTangent() && T <: Complex
        #     L = _elementwise_mult(VpdV,one(Fm))
        #     dA += 1/2*U*my_pinv(S,projector_trun_tol)*(L' - L)*V
        # end

        if codomain(t)!=domain(t)
            pru = U*U';
            prv = V'*V;
            dA += (one(pru)-pru)*dU*my_pinv(S,projector_trun_tol)*V
            dA += U*my_pinv(S,projector_trun_tol)*dV*(one(prv)-prv)
        end

        println("Norm of dA: "*string(norm(dA)))
        return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
    end
    return (U,S,V), pullback
end

# function ChainRulesCore.rrule(::typeof(my_tsvd), t::AbstractTensorMap;kwargs...)
#     T = eltype(t);
#     global chi,multiplet_tol,projector_trun_tol
#     (U,S,V) = my_tsvd(t;kwargs...);#println(S.data.values)

#     for cc=1:length(S.data.values)
#         println(diag(S.data.values[cc]))
#     end

#     F = similar(S);
#     for (k,dst) in blocks(F)

#         src = blocks(S)[k]

#         @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
#             if abs(src[j,j] - src[i,i])<1e-8
#                 d = 1e16
#             else
#                 d = src[j,j]^2-src[i,i]^2
#             end

#             dst[i,j] = (i == j) ? zero(eltype(S)) : 1/d
#         end
#         # @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
#         #     if abs(src[j,j] - src[i,i])<1e-12
#         #         d = 1e-12
#         #     else
#         #         d = src[j,j]^2-src[i,i]^2
#         #     end

#         #     dst[i,j] = (i == j) ? zero(eltype(S)) : 1/d
#         # end
#     end


#     function pullback(v)
#         dU,dS,dV = v
#         println("Norm of dU, dS, dV: "*string([norm(dU),norm(dS),norm(dV)]))

#         dA = zero(t);
#         #A_s bar term
#         if dS != ChainRulesCore.ZeroTangent()
#             dA += U*_elementwise_mult(dS,one(dS))*V
#         end
#         #A_uo bar term
#         if dU != ChainRulesCore.ZeroTangent()
#             J = _elementwise_mult((U'*dU),F)
#             dA += U*(J+J')*S*V
#         end
#         #A_vo bar term
#         if dV != ChainRulesCore.ZeroTangent()
#             VpdV = V*dV';
#             K = _elementwise_mult(VpdV,F)
#             dA += U*S*(K+K')*V
#         end
#         #A_d bar term, only relevant if matrix is complex
#         if dV != ChainRulesCore.ZeroTangent() && T <: Complex
#             L = _elementwise_mult(VpdV,one(F))
#             dA += 1/2*U*my_pinv(S,projector_trun_tol)*(L' - L)*V
#         end

#         if codomain(t)!=domain(t)
#             pru = U*U';
#             prv = V'*V;
#             dA += (one(pru)-pru)*dU*my_pinv(S,projector_trun_tol)*V
#             dA += U*my_pinv(S,projector_trun_tol)*dV*(one(prv)-prv)
#         end

#         println("Norm of dA: "*string(norm(dA)))
#         return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
#     end
#     return (U,S,V), pullback
# end
