function get_max_dim(psi::Matrix{TensorMap})
    #get maximum bond dimension of a PEPS
    dim_max=0;
    for cc in eachindex(psi)
        T=psi[cc];
        @assert Rank(T)==5 # rank 5 PEPS tensor
        dim_m=maximum([dim(space(T,1)), dim(space(T,2)), dim(space(T,3)), dim(space(T,4))]);
        dim_max=max(dim_max,dim_m);
    end
    return dim_max
end



function prepare_λ_L(A,px,py,λx_set,λy_set)
    λ1=λx_set[px,py];
    λ2=λy_set[px,py];
    λ4=λy_set[px,py+1];
    @tensor A[:]:=A[1,2,-3,4,-5]*λ1[-1,1]*λ2[2,-2]*λ4[-4,4];
    return A
end
function unprepare_λ_L(A,px,py,λx_set,λy_set)
    λ1=my_pinv(λx_set[px,py]);
    λ2=my_pinv(λy_set[px,py]);
    λ4=my_pinv(λy_set[px,py+1]);
    @tensor A[:]:=A[1,2,-3,4,-5]*λ1[-1,1]*λ2[2,-2]*λ4[-4,4];
    return A
end

function prepare_λ_R(A,px,py,λx_set,λy_set)
    λ2=λy_set[px,py];
    λ3=λx_set[px+1,py];
    λ4=λy_set[px,py+1];
    @tensor A[:]:=A[-1,2,3,4,-5]*λ2[2,-2]*λ3[3,-3]*λ4[-4,4];
    return A
end
function unprepare_λ_R(A,px,py,λx_set,λy_set)
    λ2=my_pinv(λy_set[px,py]);
    λ3=my_pinv(λx_set[px+1,py]);
    λ4=my_pinv(λy_set[px,py+1]);
    @tensor A[:]:=A[-1,2,3,4,-5]*λ2[2,-2]*λ3[3,-3]*λ4[-4,4];
    return A
end

function prepare_λ_U(A,px,py,λx_set,λy_set)
    λ1=λx_set[px,py];
    λ3=λx_set[px+1,py];
    λ4=λy_set[px,py+1];
    @tensor A[:]:=A[1,-2,3,4,-5]*λ1[-1,1]*λ3[3,-3]*λ4[-4,4];
    return A
end
function unprepare_λ_U(A,px,py,λx_set,λy_set)
    λ1=my_pinv(λx_set[px,py]);
    λ3=my_pinv(λx_set[px+1,py]);
    λ4=my_pinv(λy_set[px,py+1]);
    @tensor A[:]:=A[1,-2,3,4,-5]*λ1[-1,1]*λ3[3,-3]*λ4[-4,4];
    return A
end

function prepare_λ_D(A,px,py,λx_set,λy_set)
    λ1=λx_set[px,py];
    λ2=λy_set[px,py];
    λ3=λx_set[px+1,py];
    @tensor A[:]:=A[1,2,3,-4,-5]*λ1[-1,1]*λ2[2,-2]*λ3[3,-3];
    return A
end
function unprepare_λ_D(A,px,py,λx_set,λy_set)
    λ1=my_pinv(λx_set[px,py]);
    λ2=my_pinv(λy_set[px,py]);
    λ3=my_pinv(λx_set[px+1,py]);
    @tensor A[:]:=A[1,2,3,-4,-5]*λ1[-1,1]*λ2[2,-2]*λ3[3,-3];
    return A
end



function split_TL(T)
    T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
    T=permute(T,(1,2,3,5,4,));#L,U,d,D,R,
    T=permute(T,(1,2,4,3,5,));#L,U,D,d,R,

    U,S,V=tsvd(T,(1,2,3,),(4,5,));
    T_res=U;#L,U,D, virtual
    T_keep=S*V; #virtual, d,R,
    return T_res, T_keep
end
function back_TL(T)
    #L,U,D,d,R,
    T=permute(T,(1,2,4,3,5,));#L,U,d,D,R,
    T=permute(T,(1,2,3,5,4,));#L,U,d,R,D,
    T=permute(T,(1,5,4,2,3));
    return T
end

function split_TR(T)
    T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
    T=permute(T,(1,3,2,4,5,));#L,d,U,R,D,

    U,S,V=tsvd(T,(1,2,),(3,4,5,));
    T_keep=U*S; #(L,d, virtual)
    T_res=V;#(virtual, U,R,D)
    return T_keep,T_res
end
function back_TR(T)
    #L,d,U,R,D,
    T=permute(T,(1,3,2,4,5,));#L,U,d,R,D,
    T=permute(T,(1,5,4,2,3));
    return T
end

function split_TU(T)
    T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
    T=permute(T,(1,2,4,3,5,));#L,U,R,d,D,
    U,S,V=tsvd(T,(1,2,3,),(4,5,));
    T_res=U;#L,U,R,virtual
    T_keep=S*V;#virtual,d,D,
    return T_res,T_keep
end
function back_TU(T)
    #L,U,R,d,D,
    T=permute(T,(1,2,4,3,5,));#L,U,d,R,D,
    T=permute(T,(1,5,4,2,3));
    return T
end

function split_TD(T)
    T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
    T=permute(T,(2,1,3,4,5,));#U,L,d,R,D,
    T=permute(T,(1,3,2,4,5,));#U,d,L,R,D,

    U,S,V=tsvd(T,(1,2,),(3,4,5,));
    T_keep=U*S;#(U,d, virtual)
    T_res=V;#(virtual, L,R,D)
    return T_keep,T_res
end
function back_TD(T)
    #U,d,L,R,D,
    T=permute(T,(1,3,2,4,5,));#U,L,d,R,D,
    T=permute(T,(2,1,3,4,5,));#L,U,d,R,D,
    T=permute(T,(1,5,4,2,3));
    return T
end



function update_x_bond(Tset,λx_set,λy_set, bond_coord,Dmax)
    coe_total=1;
    Lx,Ly=size(Tset);
    pos1=[Int(bond_coord[1]-0.5),bond_coord[2]];
    pos2=[Int(bond_coord[1]+0.5),bond_coord[2]];

    TL=Tset[pos1[1],pos1[2]];
    TR=Tset[pos2[1],pos2[2]];
    # println(Dmax)
    # println(space(TL))
    # println(space(TR))

    TL=prepare_λ_L(TL,pos1[1],pos1[2],λx_set,λy_set);
    TR=prepare_λ_R(TR,pos2[1],pos2[2],λx_set,λy_set);
    ################################
    TL_res,TL_keep=split_TL(TL);
    TR_keep,TR_res=split_TR(TR);
    @tensor bond[:]:=TL_keep[-1,-2,1]*TR_keep[1,-3,-4];
    u,s,v=tsvd(permute(bond,(1,2,),(3,4,));trunc=truncdim(Dmax));
    u=u*sqrt(s);
    v=sqrt(s)*v;
 
    @tensor TL[:]:=TL_res[-1,-2,-3,1]*u[1,-4,-5];
    @tensor TR[:]:=v[-1,-2,1]*TR_res[1,-3,-4,-5];

    TL=back_TL(TL);
    TR=back_TR(TR);

    λ_new=sqrt(s);


    λ_new=λ_new/norm(λ_new);#coefficient of normalizing lambda should not be tracked, since the lambda tensor does not enter final state  
    λx_set[Int(bond_coord[1]+0.5),bond_coord[2]]=λ_new;
    ################################
    TL=unprepare_λ_L(TL,pos1[1],pos1[2],λx_set,λy_set);
    TR=unprepare_λ_R(TR,pos2[1],pos2[2],λx_set,λy_set);

    # coe_=norm(TL);
    # coe_total=coe_total*coe_;
    # TL=TL/coe_;

    # coe_=norm(TR);
    # coe_total=coe_total*coe_;
    # TR=TR/coe_;

    # println(space(TL))
    # println(space(TR))

    Tset[pos1[1],pos1[2]]=TL;
    Tset[pos2[1],pos2[2]]=TR;
    # println(space(λ_new))
    return Tset,λx_set,λy_set,coe_total
end

function update_y_bond(Tset,λx_set,λy_set, bond_coord,Dmax)
    coe_total=1;
    Lx,Ly=size(Tset);
    pos1=[bond_coord[1],Int(bond_coord[2]+0.5)];
    pos2=[bond_coord[1],Int(bond_coord[2]-0.5)];

    TU=Tset[pos1[1],pos1[2]];
    TD=Tset[pos2[1],pos2[2]];

    TU=prepare_λ_U(TU,pos1[1],pos1[2],λx_set,λy_set);
    TD=prepare_λ_D(TD,pos2[1],pos2[2],λx_set,λy_set);
    ################################
    TU_res,TU_keep=split_TU(TU);
    TD_keep,TD_res=split_TD(TD);
    @tensor bond[:]:=TU_keep[-1,-2,1]*TD_keep[1,-3,-4];
    u,s,v=tsvd(permute(bond,(1,2,),(3,4,));trunc=truncdim(Dmax));
    u=u*sqrt(s);
    v=sqrt(s)*v;
    
    @tensor TU[:]:=TU_res[-1,-2,-3,1]*u[1,-4,-5];
    @tensor TD[:]:=v[-1,-2,1]*TD_res[1,-3,-4,-5];

    TU=back_TU(TU);
    TD=back_TD(TD);

    λ_new=sqrt(s);

    λ_new=λ_new/norm(λ_new);#coefficient of normalizing lambda should not be tracked, since the lambda tensor does not enter final state 
    λy_set[bond_coord[1],Int(bond_coord[2]+0.5)]=λ_new;
    ################################
    TU=unprepare_λ_U(TU,pos1[1],pos1[2],λx_set,λy_set);
    TD=unprepare_λ_D(TD,pos2[1],pos2[2],λx_set,λy_set);

    # coe_=norm(TU);
    # coe_total=coe_total*coe_;
    # TU=TU/coe_;

    # coe_=norm(TD);
    # coe_total=coe_total*coe_;
    # TD=TD/coe_;

    Tset[pos1[1],pos1[2]]=TU;
    Tset[pos2[1],pos2[2]]=TD;
    return Tset,λx_set,λy_set, coe_total
end


function tebd_step(Tset,λx_set,λy_set,Dmax)
    Lx,Ly=size(Tset);
    λx_set=deepcopy(λx_set);
    λy_set=deepcopy(λy_set);

    log_coe=0;

    #odd x bond
    for bx in Vector(1.5:2:Lx-0.5)
        for by=1:Ly
            # println("bond: "*string((bx,by)))
            Tset,λx_set,λy_set,coe_=update_x_bond(Tset,λx_set,λy_set, (bx,by), Dmax);
            log_coe=log_coe+log(coe_);
        end
    end

    #even x bond
    for bx in Vector(2.5:2:Lx-0.5)
        for by=1:Ly
            # println("bond: "*string((bx,by)))
            Tset,λx_set,λy_set,coe_=update_x_bond(Tset,λx_set,λy_set, (bx,by), Dmax);
            log_coe=log_coe+log(coe_);
        end
    end

    #odd y bond
    for bx =1:Lx
        for by in Vector(1.5:2:Ly-0.5)
            # println("bond: "*string((bx,by)))
            Tset,λx_set,λy_set,coe_=update_y_bond(Tset,λx_set,λy_set, (bx,by), Dmax);
            log_coe=log_coe+log(coe_);
        end
    end

    #even y bond
    for bx =1:Lx
        for by in Vector(2.5:2:Ly-0.5)
            # println("bond: "*string((bx,by)))
            Tset,λx_set,λy_set,coe_=update_y_bond(Tset,λx_set,λy_set, (bx,by), Dmax);
            log_coe=log_coe+log(coe_);
        end
    end
    return Tset,λx_set,λy_set,log_coe
end

function PEPS_gauge_fix_simple(psi::finite_PEPS_with_coe,Nstep)
    #use simple tebd to obtain approximate canonical form
    D0=get_max_dim(psi.Tset);


    log_coe=psi.logcoe;
    Tset=deepcopy(psi.Tset);
    Lx,Ly=size(Tset);
    λx_set=Matrix{TensorMap}(undef,Lx+1,Ly);
    λy_set=Matrix{TensorMap}(undef,Lx,Ly+1);
    for cx=1:Lx
        for cy=1:Ly
            λx_set[cx,cy]=unitary(space(Tset[cx,cy],1), space(Tset[cx,cy],1));
        end
    end
    cx=Lx+1;
    for cy=1:Ly
        λx_set[cx,cy]=unitary(space(Tset[cx-1,cy],3)', space(Tset[cx-1,cy],3)');
    end

    for cx=1:Lx
        for cy=2:Ly+1
            λy_set[cx,cy]=unitary(space(Tset[cx,cy-1],4), space(Tset[cx,cy-1],4));
        end
    end
    cy=1;
    for cx=1:Lx
        λy_set[cx,cy]=unitary(space(Tset[cx,cy],2)', space(Tset[cx,cy],2)');
    end

    for ite=1:Nstep
        Tset,λx_set_new,λy_set_new,log_coe_=tebd_step(Tset,λx_set,λy_set,D0);
        log_coe=log_coe+log_coe_;
        errx_set=check_convergence(λx_set_new,λx_set,Lx+1,Ly);
        erry_set=check_convergence(λy_set_new,λy_set,Lx,Ly+1);
        # println(maximum(errx_set))
        # println(maximum(erry_set))
        λx_set=λx_set_new;
        λy_set=λy_set_new;
        if max(maximum(errx_set),maximum(erry_set))<1e-8
            break;
        end
    end

    

    psinew=finite_PEPS_with_coe(Tset,log_coe);

    #Tset=normalize_tensor_group(Tset);
    return psinew,λx_set,λy_set
end

