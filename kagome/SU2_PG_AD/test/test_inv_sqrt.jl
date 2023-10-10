using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using MPSKitModels, LinearAlgebra, OptimKit
using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives


cd(@__DIR__)


function truncate_multiplet(s,chi,multiplet_tol,trun_tol)
    #the multiplet is not due to su(2) symmetry
     s_dense=@ignore_derivatives sort(abs.(diag(convert(Array,s))),rev=true);


    # println(s_dense/s_dense[1])

    if length(s_dense)>chi
        value_trun=s_dense[chi+1];
    else
        value_trun=0;
    end
    value_max=maximum(s_dense);

    s_Dict=convert(Dict,s);
    
    space_full=space(s,1);
    for sp in sectors(space_full)

        diag_elem=abs.(diag(s_Dict[:data][string(sp)]));
        for cd=1:length(diag_elem)
            if ((diag_elem[cd]/value_max)<trun_tol) | (diag_elem[cd]<=value_trun) |(abs((diag_elem[cd]-value_trun)/value_trun)<multiplet_tol)
                diag_elem[cd]=0;
            end
        end
        s_Dict[:data][string(sp)]=diagm(diag_elem);
    end
    s=convert(TensorMap,s_Dict);

    s_dense=convert(Array,s);
    s_dense=unique(diag(s_dense));
    s_dense=sort(s_dense);
    if s_dense[1]==0 #return the minimal nonzero element
        return s_dense[2]
    else
        return s_dense[1]
    end

end

# function sdiag_inv_sqrt(S::AbstractTensorMap)
#     toret = similar(S);
    
#     if sectortype(S) == Trivial
#         copyto!(toret.data,LinearAlgebra.diagm(LinearAlgebra.diag(S.data).^(-1/2)));
#     else
#         for (k,b) in blocks(S)
#             copyto!(blocks(toret)[k],(LinearAlgebra.diagm(LinearAlgebra.diag(b).^(-1/2))));
#         end
#     end
#     toret
# end

function sdiag_inv_sqrt(S::AbstractTensorMap)
    toret = similar(S);
    global chi,multiplet_tol,trun_tol
    s_min=truncate_multiplet(S,chi,multiplet_tol,trun_tol)
    if sectortype(S) == Trivial
        copyto!(toret.data,LinearAlgebra.diagm(LinearAlgebra.diag(S.data).^(-1/2)));
    else
        for (k,b) in blocks(S)
            
            copyto!(blocks(toret)[k],(LinearAlgebra.diagm(LinearAlgebra.diag(b).^(-1/2))).*(LinearAlgebra.diagm(LinearAlgebra.diag(b).>=(s_min))));
        end
    end
    toret
end

# function sdiag_inv_sqrt(S::AbstractTensorMap,chi,multiplet_tol,trun_tol)
#     toret = similar(S);
#     s_min=truncate_multiplet(S,chi,multiplet_tol,trun_tol);
    
#     if sectortype(S) == Trivial
#         copyto!(toret.data,LinearAlgebra.diagm(LinearAlgebra.diag(S.data).^(-1/2)));
#     else
#         for (k,b) in blocks(S)
#             copyto!(blocks(toret)[k],(LinearAlgebra.diagm(LinearAlgebra.diag(b).^(-1/2))).*(LinearAlgebra.diagm(LinearAlgebra.diag(b).>=s_min)));
#         end
#     end
#     toret
# end


# function ChainRulesCore.rrule(::typeof(sdiag_inv_sqrt),S::AbstractTensorMap)
#     toret = sdiag_inv_sqrt(S);
#     toret,c̄ -> (ChainRulesCore.NoTangent(),-1/2*_elementwise_mult(c̄,toret'^3))
# end

function ChainRulesCore.rrule(::typeof(sdiag_inv_sqrt),S::AbstractTensorMap)
    toret = sdiag_inv_sqrt(S);
    toret,c̄ -> (ChainRulesCore.NoTangent(),-1/2*_elementwise_mult(c̄,toret'^3))
end

# function ChainRulesCore.rrule(::typeof(sdiag_inv_sqrt),S::AbstractTensorMap,chi,multiplet_tol,trun_tol)
#     toret = sdiag_inv_sqrt(S,chi,multiplet_tol,trun_tol);
#     toret,c̄ -> (ChainRulesCore.NoTangent(),-1/2*_elementwise_mult(c̄,toret'^3))
# end

function _elementwise_mult(a::AbstractTensorMap,b::AbstractTensorMap)
    dst = similar(a);
    for (k,block) in blocks(dst)
        copyto!(block,blocks(a)[k].*blocks(b)[k]);
    end
    dst
end


V=Rep[SU₂](0=>2, 1/2=>2, 1=>1);
S=TensorMap(randn,V,V)
u,S,v=tsvd(S);



function cfun(S)
    
    
    function fun(S)
        #s_min=truncate_multiplet(S,10,1e-5,1e-8)
        S_new=sdiag_inv_sqrt(S)
        E=norm(S_new);
        return E
    end

    ∂E = fun'(s)


    E=fun(S);
    

    @assert !isnan(norm(∂E))
    
    return E,∂E
end

global chi,multiplet_tol,trun_tol
chi=10;
multiplet_tol=1e-5;
trun_tol=1e-8;
a,b=cfun(S)
