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


function my_pinv(T)
    epsilon0 = 1e-12
    epsilon=epsilon0*maximum(diag(convert(Array,T)))
    T_new=deepcopy(T);

    for (k,dst) in blocks(T_new)
        src = blocks(T_new)[k]
        @inbounds for i in 1:size(dst,1)
            dst[i,i] = dst[i,i]/(dst[i,i]^2+epsilon)
        end
    end
    return T_new
end

function my_pinv2(T)
    epsilon0 = 1e-8
    epsilon=epsilon0*maximum(diag(convert(Array,T)))
    T_new=deepcopy(T);

    for (k,dst) in blocks(T_new)
        src = blocks(T_new)[k]
        @inbounds for i in 1:size(dst,1)
            if abs(dst[i,i])>epsilon
                dst[i,i] = 1/dst[i,i]
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
    global multiplet_tol, grad_inverse_tol, grad_regulation_epsilon, show_ite_grad_norm
    #grad_inverse_tol=1e-8
    #grad_regulation_epsilon=1e-12;
    (U,S,V) = my_tsvd(t;kwargs...);#println(S.data.values)
    epsilon1=maximum(diag(convert(Array,S)))*grad_inverse_tol;
    epsilon2=maximum(diag(convert(Array,S)))*grad_regulation_epsilon;
    Fp = similar(S);
    for (k,dst) in blocks(Fp)
        src = blocks(S)[k]
        @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
            ff=0;
            if abs(src[j,j]-src[i,i])>epsilon1
                if abs(abs(src[j,j])/abs(src[i,i])-1)>multiplet_tol #relative difference is big
                    ff=ff + (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon2)
                end
            end
            if src[j,j] + src[i,i]>epsilon1
                ff=ff + (src[j,j] + src[i,i])/((src[j,j] + src[i,i])^2+epsilon2)
            end
            dst[i,j] = (i == j) ? zero(eltype(S)) : ff
        end
    end

    Fm = similar(S);
    for (k,dst) in blocks(Fm)
        src = blocks(S)[k]
        @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
            ff=0;
            if abs(src[j,j]-src[i,i])>epsilon1
                if abs(abs(src[j,j])/abs(src[i,i])-1)>multiplet_tol #relative difference is big
                    ff=ff + (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon2)
                end
            end
            if src[j,j] + src[i,i]>epsilon1
                ff=ff - (src[j,j] + src[i,i])/((src[j,j] + src[i,i])^2+epsilon2)
            end
            dst[i,j] = (i == j) ? zero(eltype(S)) : ff
        end
    end
    #jldsave("svd.jld2"; U,S,V)

    function pullback(v)
        dU,dS,dV = v
        #jldsave("svd_backward.jld2"; dU,dS,dV)
        #println("Norm of dU, dS, dV: "*string([norm(dU),norm(dS),norm(dV)]))

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

        # ####!!!  In my test without such term is more accurate.
        # #A_d bar term, only relevant if matrix is complex
        # if dV != ChainRulesCore.ZeroTangent() && T <: Complex
        #     L = _elementwise_mult(V*dV',one(Fm))
        #     dA += 1/2*U*my_pinv(S)*(L' - L)*V
        # end

        if codomain(t)!=domain(t)
            pru = U*U';
            prv = V'*V;
            dA += (one(pru)-pru)*dU*my_pinv(S)*V
            dA += U*my_pinv(S)*dV*(one(prv)-prv)
        end

        if show_ite_grad_norm
            println("Norm of dA: "*string(norm(dA)))
        end
        global grad_norm
        grad_norm=deepcopy(norm(dA));
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
#             if abs(src[j,j]^2 - src[i,i]^2)<1e-12
#                 dst[i,j]=0
#             else
#                 dst[i,j]=1/(src[j,j]^2 - src[i,i]^2)
#             end


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
#             dA += 1/2*U*my_pinv2(S)*(L' - L)*V
#         end

#         if codomain(t)!=domain(t)
#             pru = U*U';
#             prv = V'*V;
#             dA += (one(pru)-pru)*dU*my_pinv2(S)*V
#             dA += U*my_pinv2(S)*dV*(one(prv)-prv)
#         end

#         println("Norm of dA: "*string(norm(dA)))
#         return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
#     end
#     return (U,S,V), pullback
# end
