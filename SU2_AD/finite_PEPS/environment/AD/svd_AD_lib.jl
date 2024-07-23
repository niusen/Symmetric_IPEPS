

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
    epsilon=epsilon0*maximum(abs.(diag(convert(Array,T))))^2
    T_new=deepcopy(T);
    
    if sectortype(space(T_new,1)) == Trivial
        mm=T_new.data;
        @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
        for i=1:size(mm,1)
            mm[i,i] = mm[i,i]/(mm[i,i]^2+epsilon)
        end
        T_new=TensorMap(mm,codomain(T),domain(T));
    else
        for cc=1:length(T_new.data.values)
            mm=T_new.data.values[cc];
            # println((norm(diag(mm))-norm(mm))/norm(mm))
            @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
            for i = 1:size(mm,1)
                mm[i,i] = mm[i,i]/(mm[i,i]^2+epsilon)
            end
            T_new.data.values[cc]=mm;
        end
    end
    return T_new
end

function my_pinv2(T)
    epsilon0 = 1e-8
    epsilon=epsilon0*maximum(abs.(diag(convert(Array,T))))
    T_new=deepcopy(T);

    if sectortype(space(T_new,1)) == Trivial
        mm=T_new.data;
        @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
        for i=1:size(mm,1)
            if abs(mm[i,i])>epsilon
                mm[i,i] = 1/mm[i,i]
            end
        end
        T_new=TensorMap(mm,codomain(T),domain(T));
    else
        for cc=1:length(T_new.data.values)
            mm=T_new.data.values[cc];
            @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
            for i = 1:size(mm,1)
                if abs(mm[i,i])>epsilon
                    mm[i,i] = 1/mm[i,i]
                end
            end
            T_new.data.values[cc]=mm;
        end
    end
    return T_new
end


function ChainRulesCore.rrule(::typeof(my_tsvd), t::AbstractTensorMap;kwargs...)
    global multiplet_tol, backward_settings
    #grad_inverse_tol=1e-8
    #grad_regulation_epsilon=1e-12;
    (U,S,V) = my_tsvd(t;kwargs...);#println(S.data.values)
    epsilon1=maximum(diag(convert(Array,S)))*backward_settings.grad_inverse_tol;
    epsilon2=maximum(diag(convert(Array,S)))*backward_settings.grad_regulation_epsilon;
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

        if backward_settings.show_ite_grad_norm
            println("Norm of dA: "*string(norm(dA)))
        end
        global grad_norm
        grad_norm=deepcopy(norm(dA));
        return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
    end
    return (U,S,V), pullback
end




# function truncate_multiplet(s,chi,multiplet_tol,trun_tol)
#     #the multiplet is not due to su(2) symmetry
#     s_dense=sort(abs.(diag(convert(Array,s))),rev=true);

#     # println(s_dense/s_dense[1])

#     if length(s_dense)>chi
#         value_trun=s_dense[chi+1];
#     else
#         value_trun=0;
#     end
#     value_max=maximum(s_dense);

#     s_Dict=convert(Dict,s);
    
#     space_full=space(s,1);
#     for sp in sectors(space_full)

#         diag_elem=abs.(diag(s_Dict[:data][string(sp)]));
#         for cd=1:length(diag_elem)
#             if ((diag_elem[cd]/value_max)<trun_tol) | (diag_elem[cd]<=value_trun) |(abs((diag_elem[cd]-value_trun)/value_trun)<multiplet_tol)
#                 diag_elem[cd]=0;
#             end
#         end
#         s_Dict[:data][string(sp)]=diagm(diag_elem);
#     end
#     s=convert(TensorMap,s_Dict);

#     # s_=sort(diag(convert(Array,s)),rev=true);
#     # s_=s_/s_[1];
#     # print(s_)
#     # @assert 1+1==3
#     return s
# end

# function delet_zero_block(U,Σ,V)

#     secs=blocksectors(Σ);
#     sec_length=Vector{Int}(undef,length(secs))
#     U_dict = convert(Dict,U)
#     Σ_dict = convert(Dict,Σ)
#     V_dict = convert(Dict,V)

#     #ProductSpace(Rep[SU₂](0=>3, 1/2=>4, 1=>4, 3/2=>2, 2=>1))

#     for cc =1:length(secs)
#         c=secs[cc];
#         if (size(diag(Σ_dict[:data][string(c)]),1)>0) & (sum(abs.(diag(Σ_dict[:data][string(c)])))>0)
#             inds=findall(x->(abs.(x).>0), diag(Σ_dict[:data][string(c)]));
#             U_dict[:data][string(c)]=U_dict[:data][string(c)][:,inds];
#             Σ_dict[:data][string(c)]=Σ_dict[:data][string(c)][inds,inds];
#             V_dict[:data][string(c)]=V_dict[:data][string(c)][inds,:];

#             sec_length[cc]=length(inds);
#         else
#             delete!(U_dict[:data], string(c))
#             delete!(V_dict[:data], string(c))
#             delete!(Σ_dict[:data], string(c))
#             sec_length[cc]=0;
#         end
#     end

#     #define sector string
#     sec_str="ProductSpace(Rep[SU₂](" *string(((dim(secs[1])-1)/2)) * "=>" * string(sec_length[1]);
#     for cc=2:length(secs)
#         sec_str=sec_str*", " * string(((dim(secs[cc])-1)/2)) * "=>" * string(sec_length[cc]);
#     end
#     sec_str=sec_str*"))"

#     U_dict[:domain]=sec_str
#     V_dict[:codomain]=sec_str
#     Σ_dict[:domain]=sec_str
#     Σ_dict[:codomain]=sec_str

#     return convert(TensorMap, U_dict), convert(TensorMap, Σ_dict), convert(TensorMap, V_dict)
# end
