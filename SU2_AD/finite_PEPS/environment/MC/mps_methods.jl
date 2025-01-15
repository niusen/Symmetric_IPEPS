

function norm_2D_simple(psi_double,chi,multiplet_tol)
    Lx=size(psi_double,1);
    Ly=size(psi_double,2);

    @assert mod(Ly,2)==0

    trun_history=[];

    mps_bot=psi_double[:,1];
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:Int(Ly/2)
        mps_bot_new,_=apply_mpo(psi_double[:,cy],mps_bot);
        mps_bot,trun_errs,_=left_truncate_simple(mps_bot_new, chi, multiplet_tol);
        trun_history=vcat(trun_history,trun_errs);
    end

    mps_top=pi_rotate_mps(psi_double[:,Ly]);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:Int(Ly/2)+1
        mps_top_new,_=apply_mpo(pi_rotate_mpo(psi_double[:,cy]),mps_top);
        mps_top,trun_errs,_=left_truncate_simple(mps_top_new, chi, multiplet_tol);
        trun_history=vcat(trun_history,trun_errs);
    end

    #convert mps_top to normal order such that one can use norm_1D()
    mps_top=mps_top[end:-1:1];
    mps_top[1]=mps_top[1]';
    mps_top[Lx]=mps_top[Lx]';
    for cx=2:Lx-1
        mps_top[cx]=permute(mps_top[cx],(2,1,3,))';
    end

    Norm=norm_1D(mps_top,mps_bot);
    return Norm,trun_history
end

function norm_1D(mps_set1,mps_set2)
    Lx=length(mps_set1);

    cx=1
    @tensor env[:]:=mps_set1[cx]'[-1,1]*mps_set2[cx][-2,1];
    for cx=2:Lx-1
        @tensor env[:]:=env[1,3]*mps_set1[cx]'[1,-1,2]*mps_set2[cx][3,-2,2];
    end
    cx=Lx;
    Norm=@tensor env[1,2]*mps_set1[cx]'[1,3]*mps_set2[cx][2,3];
    return Norm
end



function apply_mpo(mpo_set,mps_set)
    mps_set_new=deepcopy(mpo_set);
    Lx=length(mps_set);
    UR_set=@ignore_derivatives Vector{TensorMap}(undef,Lx);
    UL_set=@ignore_derivatives Vector{TensorMap}(undef,Lx);

    cx=1;
    A=mps_set[cx];
    M=mpo_set[cx];

    UR=@ignore_derivatives unitary(fuse(space(M,2)*space(A,1)), space(M,2)*space(A,1));
    @ignore_derivatives UR_set[cx]=UR;
    @tensor A_new[:]:=M[2,3,-2]*A[1,2]*UR[-1,3,1];
    mps_set_new[cx]=A_new;

    for cx=2:Lx-1
        A=mps_set[cx];
        M=mpo_set[cx];
        UL=@ignore_derivatives deepcopy(UR)';
        @ignore_derivatives UL_set[cx]=UL;
        UR=@ignore_derivatives unitary(fuse(space(M,3)*space(A,2)), space(M,3)*space(A,2));
        @ignore_derivatives UR_set[cx]=UR;
        

        @tensor A_new[:]:=M[2,1,4,-3]*A[3,5,1]*UL[2,3,-1]*UR[-2,4,5];
        mps_set_new[cx]=A_new;
    end

    cx=Lx;
    A=mps_set[cx];
    M=mpo_set[cx];
    UL=@ignore_derivatives deepcopy(UR)';
    @ignore_derivatives UL_set[cx]=UL;
    @tensor A_new[:]:=M[2,1,-2]*A[3,1]*UL[2,3,-1];
    mps_set_new[cx]=A_new;

    return mps_set_new,UR_set,UL_set
end

function pi_rotate_mpo(mpo_set)
    mpo_set_new=deepcopy(mpo_set);
    Lx=length(mpo_set_new);

    cx=1;
    mpo_set_new[cx]=permute(mpo_set_new[cx],(2,3,1,));
    for cx=2:Lx-1
        mpo_set_new[cx]=permute(mpo_set_new[cx],(3,4,1,2));
    end
    cx=Lx;
    mpo_set_new[cx]=permute(mpo_set_new[cx],(3,1,2,));

    return mpo_set_new[end:-1:1]
end

function pi_rotate_mps(mps_set)
    mps_set_new=deepcopy(mps_set);
    Lx=length(mps_set_new);

    cx=1;
    mps_set_new[cx]=permute(mps_set_new[cx],(2,1,));
    for cx=2:Lx-1
        mps_set_new[cx]=permute(mps_set_new[cx],(3,1,2));
    end
    cx=Lx;
    mps_set_new[cx]=permute(mps_set_new[cx],(1,2,));

    return mps_set_new[end:-1:1]
end

function pinv_canonical(T)
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

function right_canonical(mps_set)
    
    Lx=length(mps_set);
    global use_AD;
    mps_set=deepcopy(mps_set);
    Lx=length(mps_set);
    unitarys_R=Vector{TensorMap}(undef,Lx);
    unitarys_L=Vector{TensorMap}(undef,Lx);

    cx=Lx;
    u,s,v=my_tsvd(permute(mps_set[cx],(1,),(2,)));

    @ignore_derivatives unitarys_L[cx]=pinv_canonical(s)*u';
    @ignore_derivatives unitarys_R[cx-1]=u*s;
    u=u*s;
    mps_set[cx]=permute(v,(1,2,));
    A=mps_set[cx-1];
    @tensor A[:]:=A[-1,1,-3]*u[1,-2];
    mps_set[cx-1]=A;

    for cx=Lx-1:-1:2

        u,s,v=my_tsvd(permute(mps_set[cx],(1,),(2,3,)));
        
        @ignore_derivatives unitarys_L[cx]=pinv_canonical(s)*u';
        @ignore_derivatives unitarys_R[cx-1]=u*s;
        u=u*s;
        mps_set[cx]=permute(v,(1,2,3,));
        A=mps_set[cx-1];
        if cx-1>1
            @tensor A[:]:=A[-1,1,-3]*u[1,-2];
            mps_set[cx-1]=permute(A,(1,2,3,));
        elseif cx-1==1
            @tensor A[:]:=A[1,-2]*u[1,-1];
            mps_set[cx-1]=permute(A,(1,2,));
        end
        
    end
    return mps_set,unitarys_R,unitarys_L
end

function left_truncate_simple(mps_set, chi, multiplet_tol)
    #use canonical form, but more expensive

    mps_set,unitarys_R,unitarys_L=right_canonical(mps_set);

    Lx=length(mps_set);
    projectors_R=Vector{TensorMap}(undef,Lx);
    projectors_L=Vector{TensorMap}(undef,Lx);
    trun_errs=[];

    # for c1=1:Lx
    #     println(norm(mps_set[c1]))
    # end

    
    cx=1;
    T=permute(mps_set[cx],(2,),(1,));
    u_trun,s_trun,v_trun=my_tsvd(T; trunc=truncdim(chi+20));
    trun_err=@ignore_derivatives 1-dot(s_trun,s_trun)/dot(T,T);

    @ignore_derivatives projectors_R[cx]=v_trun'*pinv_canonical(s_trun);
    @ignore_derivatives projectors_L[cx+1]=s_trun*v_trun;
    # projectors_R[cx]=@ignore_derivatives v_trun'; #this is incorrect, as the gauge matters for further updates and truncations
    # projectors_L[cx+1]=@ignore_derivatives v_trun; #this is incorrect, as the gauge matters for further updates and truncation

    v_trun=s_trun*v_trun;
    mps_set[cx]=permute(u_trun,(2,1,));
    A=mps_set[cx+1];
    @tensor A[:]:=v_trun[-1,1]*A[1,-2,-3];
    mps_set[cx+1]=permute(A,(1,2,3,));
    trun_errs=vcat(trun_errs,trun_err);

    for cx=2:Lx-1
        T=permute(mps_set[cx],(1,3,),(2,));
        u_trun,s_trun,v_trun=my_tsvd(T; trunc=truncdim(chi+20));     
        trun_err=@ignore_derivatives 1-dot(s_trun,s_trun)/dot(T,T);

        @ignore_derivatives projectors_R[cx]=v_trun'*pinv_canonical(s_trun);
        @ignore_derivatives projectors_L[cx+1]=s_trun*v_trun;
        # projectors_R[cx]=@ignore_derivatives v_trun'; #this is incorrect, as the gauge matters for further updates and truncations
        # projectors_L[cx+1]=@ignore_derivatives v_trun; #this is incorrect, as the gauge matters for further updates and truncations

        #println(norm(T*v_trun'*pinv_canonical(s_trun)-u_trun)/norm(u_trun)); #the rank of u_trun is higher than the rank of s_trun because of extra truncation in my_tsvd()
        v_trun=s_trun*v_trun;
        mps_set[cx]=permute(u_trun,(1,3,2,));
        A=mps_set[cx+1];
        if cx+1<Lx
            @tensor A[:]:=v_trun[-1,1]*A[1,-2,-3];
            mps_set[cx+1]=permute(A,(1,2,3,));
            trun_errs=vcat(trun_errs,trun_err);
        elseif cx+1==Lx
            @tensor A[:]:=v_trun[-1,1]*A[1,-2];
            mps_set[cx+1]=permute(A,(1,2,));
            trun_errs=vcat(trun_errs,trun_err);
        end
        
    end





    return mps_set,trun_errs, unitarys_R,unitarys_L, projectors_R,projectors_L
end

function my_tsvd(T::AbstractTensorMap; kwargs...)

    global chi, multiplet_tol, svd_settings
    if svd_settings.svd_trun_method=="chi"
        (U,S,V) = tsvd(T;kwargs...);
        if length((1,kwargs...,))==1 #truncation not used
            S=truncate_multiplet(S,100000000,0,multiplet_tol);
        else #truncation used
            S=truncate_multiplet(S,chi,0,multiplet_tol);
        end

        U,S,V=delet_zero_block(U,S,V);
    elseif svd_settings.svd_trun_method=="tol"
        (U,S,V) = tsvd(T);
        S_tem=truncate_multiplet(S,chi,0,multiplet_tol);
        if abs(1-dot(S_tem,S_tem)/dot(T,T))<(svd_settings.tol)
            S=S_tem; 
        else #use larger chi to decrease truncate error
            S=truncate_multiplet(S,1000000,svd_settings.tol,multiplet_tol);
            @warn "chi not enough, extend chi to "*string(dim(space(S,1)))*"."
            if dim(space(S,1))>svd_settings.chi_max
                S=truncate_multiplet(S,svd_settings.chi_max,0,multiplet_tol);
                @warn "chi too large, restrict to chi_max."
            end

        end
        U,S,V=delet_zero_block(U,S,V);
    end
    return (U,S,V)
end







function truncate_multiplet(s,chi0,trun_tol,multiplet_tol)
    #the multiplet is not due to su(2) symmetry
    s_dense=sort(abs.(diag(convert(Array,s))),rev=true);

    # println(s_dense/s_dense[1])

    if length(s_dense)>chi0
        value_trun=s_dense[chi0+1];
    else
        if trun_tol==0
            value_trun=0;
        else
            norm_total=sum(s_dense.^2);
            norm_=0;

            for cccc=1:length(s_dense)
                norm_=norm_+s_dense[cccc]^2;
                if (norm_/norm_total)>(1-trun_tol)
                    if cccc<length(s_dense)
                        value_trun=s_dense[cccc+1];
                    else
                        value_trun=0;
                    end
                    break;
                end
            end

        end
    end
    value_max=maximum(s_dense);

    s_Dict=convert(Dict,s);
    
    space_full=space(s,1);
    for sp in sectors(space_full)

        diag_elem=abs.(diag(s_Dict[:data][string(sp)]));
        for cd=1:length(diag_elem)
            if (diag_elem[cd]<=value_trun) |(abs((diag_elem[cd]-value_trun)/value_trun)<multiplet_tol)
                diag_elem[cd]=0;
            end
        end
        s_Dict[:data][string(sp)]=diagm(diag_elem);
    end
    s=convert(TensorMap,s_Dict);

    return s
end



function delet_zero_block(U,Σ,V)
    # T0=U*Σ*V;
    secs=blocksectors(Σ);
    sec_length=Vector{Int}(undef,length(secs))
    U_dict = convert(Dict,U)
    Σ_dict = convert(Dict,Σ)
    V_dict = convert(Dict,V)

    #ProductSpace(Rep[SU₂](0=>3, 1/2=>4, 1=>4, 3/2=>2, 2=>1))
    if sectortype(space(Σ,1)) == Trivial
        inds=findall(x->(abs.(x).>0), diag(Σ_dict[:data]["Trivial()"]));
        U_dict[:data]["Trivial()"]=U_dict[:data]["Trivial()"][:,inds];
        Σ_dict[:data]["Trivial()"]=Σ_dict[:data]["Trivial()"][inds,inds];
        V_dict[:data]["Trivial()"]=V_dict[:data]["Trivial()"][inds,:];

        sec_str="ProductSpace(ℂ^"*string(length(inds))*")";
        U_dict[:domain]=sec_str
        V_dict[:codomain]=sec_str
        Σ_dict[:domain]=sec_str
        Σ_dict[:codomain]=sec_str
    else
        if isa(space(Σ,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}) #SU(2)
            for cc =1:length(secs)
                c=secs[cc];
                if (size(diag(Σ_dict[:data][string(c)]),1)>0) & (sum(abs.(diag(Σ_dict[:data][string(c)])))>0)
                    inds=findall(x->(abs.(x).>0), diag(Σ_dict[:data][string(c)]));
                    U_dict[:data][string(c)]=U_dict[:data][string(c)][:,inds];
                    Σ_dict[:data][string(c)]=Σ_dict[:data][string(c)][inds,inds];
                    V_dict[:data][string(c)]=V_dict[:data][string(c)][inds,:];

                    sec_length[cc]=length(inds);
                else
                    delete!(U_dict[:data], string(c))
                    delete!(V_dict[:data], string(c))
                    delete!(Σ_dict[:data], string(c))
                    sec_length[cc]=0;
                end
            end

            #define sector string
            sec_str="ProductSpace(Rep[SU₂](" *string(((dim(secs[1])-1)/2)) * "=>" * string(sec_length[1]);
            for cc=2:length(secs)
                sec_str=sec_str*", " * string(((dim(secs[cc])-1)/2)) * "=>" * string(sec_length[cc]);
            end
            sec_str=sec_str*"))"

            U_dict[:domain]=sec_str
            V_dict[:codomain]=sec_str
            Σ_dict[:domain]=sec_str
            Σ_dict[:codomain]=sec_str
        elseif isa(space(Σ,1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}) #U(1)
            for cc =1:length(secs)
                c=secs[cc];
                if (size(diag(Σ_dict[:data][string(c)]),1)>0) & (sum(abs.(diag(Σ_dict[:data][string(c)])))>0)
                    inds=findall(x->(abs.(x).>0), diag(Σ_dict[:data][string(c)]));
                    U_dict[:data][string(c)]=U_dict[:data][string(c)][:,inds];
                    Σ_dict[:data][string(c)]=Σ_dict[:data][string(c)][inds,inds];
                    V_dict[:data][string(c)]=V_dict[:data][string(c)][inds,:];

                    sec_length[cc]=length(inds);
                else
                    delete!(U_dict[:data], string(c))
                    delete!(V_dict[:data], string(c))
                    delete!(Σ_dict[:data], string(c))
                    sec_length[cc]=0;
                end
            end

            #define sector string
            sec_str="ProductSpace(Rep[U₁](" *string(secs[1].charge) * "=>" * string(sec_length[1]);
            for cc=2:length(secs)
                sec_str=sec_str*", " * string(secs[cc].charge) * "=>" * string(sec_length[cc]);
            end
            sec_str=sec_str*"))"

            U_dict[:domain]=sec_str
            V_dict[:codomain]=sec_str
            Σ_dict[:domain]=sec_str
            Σ_dict[:codomain]=sec_str
        end
    end

    # T1=convert(TensorMap, U_dict)*convert(TensorMap, Σ_dict)*convert(TensorMap, V_dict);
    # println(norm(T0-T1)/norm(T0))
    return convert(TensorMap, U_dict), convert(TensorMap, Σ_dict), convert(TensorMap, V_dict)
end

