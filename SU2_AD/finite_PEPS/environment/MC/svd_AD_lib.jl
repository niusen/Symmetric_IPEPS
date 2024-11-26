

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
    
    tt_dense=convert(Array,T);
    @assert (norm(diag(tt_dense))-norm(tt_dense))/norm(tt_dense)<1e-14;

    if sectortype(space(T_new,1)) == Trivial
        mm=T_new.data;
        # @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
        for i=1:size(mm,1)
            mm[i,i] = mm[i,i]/(mm[i,i]^2+epsilon)
        end
        T_new=TensorMap(mm,codomain(T),domain(T));
    else
        for cc=1:length(T_new.data.values)
            mm=T_new.data.values[cc];
            # @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
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



function truncate_block_svd(U_set,S_set,V_set,M,chi)
    # JLDnm="test.jld2";
    # init___=Dict([("M",M),("chi",chi),("trun_tol",trun_tol)]);
    # save(JLDnm, "init",init___);

    spins=Vector(undef,0);


    #######

    
    for cs=1:length(M.data.keys)
        if typeof(space(M,1))==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
            spin=M.data.keys[cs].n;
            spins=vcat(spins,spin);
        elseif typeof(space(M,1))==GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}
            spin=M.data.keys[cs].charge;
            spins=vcat(spins,spin);
        elseif typeof(space(M,1))==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
            spin=M.data.keys[cs].j;
            spins=vcat(spins,spin);
        elseif typeof(space(M,1))==GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
            spin=M.data.keys[cs].sectors[2].j;
            spins=vcat(spins,spin);
        end
    end
    
    function truncate_S(spins,S_set,chi)
        S_set_dense=[];
        for cc=1:length(S_set)
            for dd=1:length(S_set[cc])
                if typeof(space(M,1))==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
                    S_set_dense=vcat(S_set_dense,S_set[cc][dd]);
                elseif typeof(space(M,1))==GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}
                    S_set_dense=vcat(S_set_dense,S_set[cc][dd]);
                elseif typeof(space(M,1))==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
                    S_set_dense=vcat(S_set_dense,S_set[cc][dd]*ones(Int(2*spins[cc]+1))*sqrt(Int(2*spins[cc]+1)));#large spin have larger weight
                elseif typeof(space(M,1))==GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
                    S_set_dense=vcat(S_set_dense,S_set[cc][dd]*ones(Int(2*spins[cc]+1))*sqrt(Int(2*spins[cc]+1)));#large spin have larger weight
                end
            end
        end
        sorted=sort(S_set_dense,rev=true);
    
        if length(sorted)>chi
            value_trun=sorted[chi+1];
        else
            value_trun=sorted[end];
        end
    
        for cd=1:length(S_set)
            if typeof(space(M,1))==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
                pos=findall(x->x>value_trun,S_set[cd]);
            elseif typeof(space(M,1))==GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}
                pos=findall(x->x>value_trun,S_set[cd]);
            elseif typeof(space(M,1))==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
                pos=findall(x->x>value_trun,S_set[cd]*sqrt(Int(2*spins[cd]+1)));#large spin have larger weight
            elseif typeof(space(M,1))==GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
                pos=findall(x->x>value_trun,S_set[cd]*sqrt(Int(2*spins[cd]+1)));#large spin have larger weight
            end
            
            if length(pos)>0
                S_set[cd]=S_set[cd][pos];
            elseif length(pos)==0
                S_set[cd]=[];
            end
        end
    
        return S_set
    end
    
    if (length(domain(M))==2)&&(length(codomain(M))==2)
        VL=space(M,1)*space(M,2);
        VR=space(M,3)*space(M,4);
    elseif (length(domain(M))==1)&&(length(codomain(M))==1)
        VL=space(M,1);
        VR=space(M,2);
    end


    S_set=truncate_S(spins,S_set,chi);
    for cc=1:length(spins)
        dim=length(S_set[cc]);
        U_set[cc]=U_set[cc][:,1:dim];
        V_set[cc]=V_set[cc][1:dim,:];
    end

    block_dims=Vector{Int64}(undef,length(spins));
    Sector_name=Vector{Sector}(undef,length(spins));
    for cc=1:length(S_set);
        block_dims[cc]=length(S_set[cc]);
        Sector_name[cc]=M.data.keys[cc];
    end
    pos=findall(x->x>0,block_dims);
    U_set=U_set[pos];
    S_set=S_set[pos];
    V_set=V_set[pos];
    spins=spins[pos];
    Sector_name=Sector_name[pos];
    
    
    function group_components(U_set,S_set,V_set,spins,Sector_name,VL,VR)
        VL=fuse(VL);
        VR=fuse(VR);

        spin_dim=deepcopy(spins);

        for cs=1:length(spins)
            spin_dim[cs]=length(S_set[cs]);
        end

        if typeof(VL)==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
            @assert length(spin_dim)<=2
            Vtotal=Rep[ℤ₂](spins[1]=>spin_dim[1]);
            for cs=2:length(spins)
                Vtotal=Vtotal⊕ Rep[ℤ₂](spins[cs]=>spin_dim[cs]);
            end
            # Vtotal=Rep[ℤ₂](Int(spins[1])=>spin_dim[1],Int(spins[2])=>spin_dim[2]);
        elseif typeof(space(M,1))==GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}
            Vtotal=Rep[U₁](spins[1]=>spin_dim[1]);
            for cs=2:length(spins)
                Vtotal=Vtotal⊕ Rep[U₁](spins[cs]=>spin_dim[cs]);
            end
        elseif typeof(space(M,1))==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
            Vtotal=Rep[SU₂](spins[1]=>spin_dim[1]);
            for cs=2:length(spins)
                Vtotal=Vtotal⊕ Rep[SU₂](spins[cs]=>spin_dim[cs]);
            end
        elseif typeof(space(M,1))==GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
            cs=1;
            Vtotal=Rep[U₁ × SU₂]((Sector_name[cs].sectors[1].charge,spins[cs])=>spin_dim[cs]);
            for cs=2:length(spins)
                Vtotal=Vtotal⊕ Rep[U₁ × SU₂]((Sector_name[cs].sectors[1].charge,spins[cs])=>spin_dim[cs]);
            end
        end


        Um=TensorMap(randn,VL,Vtotal)*(0*im);
        Vm=TensorMap(randn,Vtotal,VR')*(0*im);
        Sm=TensorMap(randn,Vtotal,Vtotal)*(0);

        for cs=1:length(spins)
            Um.data.values[cs]=U_set[cs];
            Vm.data.values[cs]=V_set[cs];
            Sm.data.values[cs]=diagm(S_set[cs]);
        end
        return Um,Sm,Vm
    end





    Um,Sm,Vm=group_components(U_set,S_set,V_set,spins,Sector_name,VL,VR);

    if (length(domain(M))==2)&&(length(codomain(M))==2)
        U1=@ignore_derivatives unitary(space(M,1)*space(M,2),space(Um,1));
        Um=U1*Um;
        U2=@ignore_derivatives unitary(space(Vm,2)',space(M,3)'*space(M,4)');
        Vm=Vm*U2;
    elseif (length(domain(M))==1)&&(length(codomain(M))==1)
        U1=@ignore_derivatives unitary(space(M,1),space(Um,1));
        Um=U1*Um;
        U2=@ignore_derivatives unitary(space(Vm,2)',space(M,2)');
        Vm=Vm*U2;
    end


    
    
    return Um,Sm,Vm, M

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
