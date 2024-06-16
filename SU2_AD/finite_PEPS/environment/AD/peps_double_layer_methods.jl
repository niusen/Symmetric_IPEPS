function contract_physical_all(psi_double_open, U_s_s);
    Lx=size(psi_double_open,1);
    Ly=size(psi_double_open,2);
    psi_double=deepcopy(psi_double_open);
    for cx=1:Lx
        for cy=1:Ly
            AA=contract_physical(psi_double_open[cx,cy], U_s_s);
            psi_double=matrix_update(psi_double,cx,cy,AA);
        end
    end
    return psi_double
end

function contract_physical(AA, U_s_s);
    if Rank(AA)==3
        @tensor AA[:]:=AA[-1,-2,1]*U_s_s[2,2,1];
    elseif Rank(AA)==4
        @tensor AA[:]:=AA[-1,-2,-3,1]*U_s_s[2,2,1];
    elseif Rank(AA)==5
        @tensor AA[:]:=AA[-1,-2,-3,-4,1]*U_s_s[2,2,1];
    end
    return AA
end

function construct_double_layer_open(psi)
    psi=deepcopy(psi);
    Lx=size(psi,1);
    Ly=size(psi,2);
    psi_double=Matrix{TensorMap}(undef,Lx,Ly);
    return_unitary=false;
    for cx=2:Lx-1
        for cy=2:Ly-1
            AA,U_s_s=build_double_layer_bulk_open(psi[cx,cy],return_unitary);
            psi_double[cx,cy]=AA;
        end
    end

    cx=1;
    for cy=2:Ly-1
        AA,U_s_s=build_double_layer_left_open(psi[cx,cy],return_unitary);
        psi_double[cx,cy]=AA;
    end

    cx=Lx;
    for cy=2:Ly-1
        AA,U_s_s=build_double_layer_right_open(psi[cx,cy],return_unitary);
        psi_double[cx,cy]=AA;
    end

    cy=1;
    for cx=2:Lx-1
        AA,U_s_s=build_double_layer_bot_open(psi[cx,cy],return_unitary);
        psi_double[cx,cy]=AA;
    end

    cy=Ly;
    for cx=2:Lx-1
        AA,U_s_s=build_double_layer_top_open(psi[cx,cy],return_unitary);
        psi_double[cx,cy]=AA;
    end

    cx=1;
    cy=1;
    AA,U_s_s=build_double_layer_left_bot_open(psi[cx,cy],return_unitary);
    psi_double[cx,cy]=AA;

    cx=1;
    cy=Ly;
    AA,U_s_s=build_double_layer_left_top_open(psi[cx,cy],return_unitary);
    psi_double[cx,cy]=AA;

    cx=Lx;
    cy=1;
    AA,U_s_s=build_double_layer_right_bot_open(psi[cx,cy],return_unitary);
    psi_double[cx,cy]=AA;

    cx=Lx;
    cy=Ly;
    AA,U_s_s=build_double_layer_right_top_open(psi[cx,cy],return_unitary);
    psi_double[cx,cy]=AA;

    return psi_double, U_s_s
end

function construct_double_layer(psi1,psi2)
    psi1=deepcopy(psi1);
    if psi2==nothing
        psi2=deepcopy(psi1)
    else
        psi2=deepcopy(psi2);
    end
    Lx=size(psi1,1);
    Ly=size(psi1,2);
    psi_double=ones(Lx,Ly);
    psi_double=convert(Matrix{Any},psi_double);
    #psi_double=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=2:Lx-1
        for cy=2:Ly-1
            AA,_=build_double_layer_bulk(psi1[cx,cy],psi2[cx,cy],[]);
            psi_double=matrix_update(psi_double,cx,cy,AA);
        end
    end

    cx=1;
    for cy=2:Ly-1
        AA,_=build_double_layer_left(psi1[cx,cy],psi2[cx,cy],[]);
        psi_double=matrix_update(psi_double,cx,cy,AA);
    end

    cx=Lx;
    for cy=2:Ly-1
        AA,_=build_double_layer_right(psi1[cx,cy],psi2[cx,cy],[]);
        psi_double=matrix_update(psi_double,cx,cy,AA);
    end

    cy=1;
    for cx=2:Lx-1
        AA,_=build_double_layer_bot(psi1[cx,cy],psi2[cx,cy],[]);
        psi_double=matrix_update(psi_double,cx,cy,AA);
    end

    cy=Ly;
    for cx=2:Lx-1
        AA,_=build_double_layer_top(psi1[cx,cy],psi2[cx,cy],[]);
        psi_double=matrix_update(psi_double,cx,cy,AA);
    end

    cx=1;
    cy=1;
    AA,_=build_double_layer_left_bot(psi1[cx,cy],psi2[cx,cy],[]);
    psi_double=matrix_update(psi_double,cx,cy,AA);

    cx=1;
    cy=Ly;
    AA,_=build_double_layer_left_top(psi1[cx,cy],psi2[cx,cy],[]);
    psi_double=matrix_update(psi_double,cx,cy,AA);

    cx=Lx;
    cy=1;
    AA,_=build_double_layer_right_bot(psi1[cx,cy],psi2[cx,cy],[]);
    psi_double=matrix_update(psi_double,cx,cy,AA);

    cx=Lx;
    cy=Ly;
    AA,_=build_double_layer_right_top(psi1[cx,cy],psi2[cx,cy],[]);
    psi_double=matrix_update(psi_double,cx,cy,AA);

    return psi_double
end

function construct_double_layer_pos(psi1,psi2,cx,cy)
    Lx=size(psi1,1);
    Ly=size(psi1,2);

    if (1<cx<Lx)& (1<cy<Ly)
        AA,U_L,U_D,U_R,U_U=build_double_layer_bulk(psi1[cx,cy],psi2[cx,cy],[]);
        return AA,U_L,U_D,U_R,U_U
    end

    if (cx==1)& (1<cy<Ly)
        AA,U_D,U_R,U_U=build_double_layer_left(psi1[cx,cy],psi2[cx,cy],[]);
        return AA, nothing,U_D,U_R,U_U
    end

    if (cx==Lx)& (1<cy<Ly)
        AA,U_L,U_D,U_U=build_double_layer_right(psi1[cx,cy],psi2[cx,cy],[]);
        return AA,U_L,U_D,nothing,U_U
    end

    if (1<cx<Lx)& (cy==1)
        AA,U_L,U_R,U_U=build_double_layer_bot(psi1[cx,cy],psi2[cx,cy],[]);
        return AA,U_L,nothing,U_R,U_U
    end


    if (1<cx<Lx)& (cy==Ly)
        AA,U_L,U_D,U_R=build_double_layer_top(psi1[cx,cy],psi2[cx,cy],[]);
        return AA,U_L,U_D,U_R,nothing
    end


    if (cx==1)& (cy==1)
        AA,U_R,U_U=build_double_layer_left_bot(psi1[cx,cy],psi2[cx,cy],[]);
        return AA,nothing,nothing,U_R,U_U
    end

    if (cx==1)& (cy==Ly)
        AA,U_D,U_R=build_double_layer_left_top(psi1[cx,cy],psi2[cx,cy],[]);
        return AA,nothing,U_D,U_R,nothing
    end

    if (cx==Lx)& (cy==1)
        AA,U_L,U_U=build_double_layer_right_bot(psi1[cx,cy],psi2[cx,cy],[]);
        return AA,U_L,nothing,nothing,U_U
    end

    if (cx==Lx)& (cy==Ly)
        AA,U_L,U_D=build_double_layer_right_top(psi1[cx,cy],psi2[cx,cy],[]);
        return AA,U_L,U_D,nothing,nothing
    end

end

function build_double_layer_bulk(A1,A2,operator)
    #display(space(A))
    A1=permute(A1,(1,2,),(3,4,5));
    A2=permute(A2,(1,2,),(3,4,5));
    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A1, 2)' ⊗ space(A2, 2)), space(A1, 2)' ⊗ space(A2, 2))*(1+0*im);

    U_R=@ignore_derivatives unitary(space(A1, 3) ⊗ space(A2, 3)', fuse(space(A1, 3)' ⊗ space(A2, 3)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A1, 4) ⊗ space(A2, 4)', fuse(space(A1, 4)' ⊗ space(A2, 4)))*(1+0*im);


    Up_tem=@ignore_derivatives unitary(fuse(space(A1,1)*space(A1,2)), space(A1,1)*space(A1,2))*(1+0*im);
    vMp=Up_tem*A1;
    uMp=Up_tem';
    @assert(norm(uMp*vMp-A1)/norm(A1)<1e-12);

    U_tem=@ignore_derivatives unitary(fuse(space(A2,1)*space(A2,2)), space(A2,1)*space(A2,2))*(1+0*im);
    vM=U_tem*A2;
    uM=U_tem';
    @assert(norm(uM*vM-A2)/norm(A2)<1e-12);

    uMp=permute(uMp,(1,2,3,),());
    uM=permute(uM,(1,2,3,),());

    Vp=space(vMp,1);
    V=space(vM,1);

    U=@ignore_derivatives unitary(fuse(Vp' ⊗ V), Vp' ⊗ V)*(1+0*im);
    @tensor double_LD[:]:=uMp'[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];


    vMp=permute(vMp,(1,2,3,4,),());
    vM=permute(vM,(1,2,3,4,),());

    if operator==[]
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
        @tensor double_RU[:]:=vMp'[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];
    else
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
        @tensor double_RU[:]:=vMp'[3,-2,-4,1]*operator[2,1]*double_RU[-1,3,-3,-5,2];
    end
    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))
    double_RU=permute(double_RU,(1,4,5,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,3,));
    AA_fused=double_LD*double_RU;


    return AA_fused, U_L,U_D,U_R,U_U
end


function build_double_layer_top(A1,A2,operator)
    #display(space(A))

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A1, 2)' ⊗ space(A2, 2)), space(A1, 2)' ⊗ space(A2, 2))*(1+0*im);

    U_R=@ignore_derivatives unitary(space(A1, 3) ⊗ space(A2, 3)', fuse(space(A1, 3)' ⊗ space(A2, 3)))*(1+0*im);

    if operator==[]
        @tensor AA_fused[:]:=A1'[1,6,4,3]*A2[2,7,5,3]*U_L[-1,1,2]*U_D[-2,6,7]*U_R[4,5,-3];
    else
        @tensor A2[:]:=A2[-1,-2,-3,1]*operator[-4,1];
        @tensor AA_fused[:]:=A1'[1,6,4,3]*A2[2,7,5,3]*U_L[-1,1,2]*U_D[-2,6,7]*U_R[4,5,-3];
    end


    return AA_fused, U_L,U_D,U_R
end

function build_double_layer_bot(A1,A2,operator)
    #display(space(A))

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);

    U_R=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A1, 3) ⊗ space(A2, 3)', fuse(space(A1, 3)' ⊗ space(A2, 3)))*(1+0*im);

    if operator==[]
        @tensor AA_fused[:]:=A1'[1,4,6,3]*A2[2,5,7,3]*U_L[-1,1,2]*U_U[6,7,-3]*U_R[4,5,-2];
    else
        @tensor A2[:]:=A2[-1,-2,-3,1]*operator[-4,1];
        @tensor AA_fused[:]:=A1'[1,4,6,3]*A2[2,5,7,3]*U_L[-1,1,2]*U_U[6,7,-3]*U_R[4,5,-2];
    end


    return AA_fused, U_L,U_R,U_U
end


function build_double_layer_left(A1,A2,operator)
    #display(space(A))

    U_D=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);

    U_R=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A1, 3) ⊗ space(A2, 3)', fuse(space(A1, 3)' ⊗ space(A2, 3)))*(1+0*im);

    if operator==[]
        @tensor AA_fused[:]:=A1'[1,4,6,3]*A2[2,5,7,3]*U_U[6,7,-3]*U_D[-1,1,2]*U_R[4,5,-2];
    else
        @tensor A2[:]:=A2[-1,-2,-3,1]*operator[-4,1];
        @tensor AA_fused[:]:=A1'[1,4,6,3]*A2[2,5,7,3]*U_U[6,7,-3]*U_D[-1,1,2]*U_R[4,5,-2];
    end


    return AA_fused, U_D,U_R,U_U
end

function build_double_layer_right(A1,A2,operator)
    #display(space(A))

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A1, 2)' ⊗ space(A2, 2)), space(A1, 2)' ⊗ space(A2, 2))*(1+0*im);

    U_U=@ignore_derivatives unitary(space(A1, 3) ⊗ space(A2, 3)', fuse(space(A1, 3)' ⊗ space(A2, 3)))*(1+0*im);

    if operator==[]
        @tensor AA_fused[:]:=A1'[1,4,6,3]*A2[2,5,7,3]*U_L[-1,1,2]*U_D[-2,4,5]*U_U[6,7,-3];
    else
        @tensor A2[:]:=A2[-1,-2,-3,1]*operator[-4,1];
        @tensor AA_fused[:]:=A1'[1,4,6,3]*A2[2,5,7,3]*U_L[-1,1,2]*U_D[-2,4,5]*U_U[7,6,-3];
    end


    return AA_fused, U_L,U_D,U_U
end


function build_double_layer_left_top(A1,A2,operator)
    #display(space(A))


    U_D=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_R=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);

    if operator==[]
        @tensor AA_fused[:]:=A1'[1,4,3]*A2[2,5,3]*U_D[-1,1,2]*U_R[4,5,-2];
    else
        @tensor A2[:]:=A2[-1,-2,1]*operator[-3,1];
        @tensor AA_fused[:]:=A1'[1,4,3]*A2[2,5,3]*U_D[-1,1,2]*U_R[4,5,-2];
    end


    return AA_fused, U_D,U_R
end

function build_double_layer_right_top(A1,A2,operator)
    #display(space(A))

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A1, 2)' ⊗ space(A2, 2)), space(A1, 2)' ⊗ space(A2, 2))*(1+0*im);

    if operator==[]
        @tensor AA_fused[:]:=A1'[1,4,3]*A2[2,5,3]*U_L[-1,1,2]*U_D[-2,4,5];
    else
        @tensor A2[:]:=A2[-1,-2,1]*operator[-3,1];
        @tensor AA_fused[:]:=A1'[1,4,3]*A2[2,5,3]*U_L[-1,1,2]*U_D[-2,4,5];
    end


    return AA_fused, U_L,U_D
end

function build_double_layer_left_bot(A1,A2,operator)
    #display(space(A))

    U_R=@ignore_derivatives unitary(space(A1, 1) ⊗ space(A2, 1)', fuse(space(A1, 1)' ⊗ space(A2, 1)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);

    if operator==[]
        @tensor AA_fused[:]:=A1'[1,4,3]*A2[2,5,3]*U_R[1,2,-1]*U_U[4,5,-2];
    else
        @tensor A2[:]:=A2[-1,-2,1]*operator[-3,1];
        @tensor AA_fused[:]:=A1'[1,4,3]*A2[2,5,3]*U_R[1,2,-1]*U_U[4,5,-2];
    end


    return AA_fused, U_R,U_U
end

function build_double_layer_right_bot(A1,A2,operator)
    #display(space(A))

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);

    if operator==[]
        @tensor AA_fused[:]:=A1'[1,4,3]*A2[2,5,3]*U_L[-1,1,2]*U_U[4,5,-2];
    else
        @tensor A2[:]:=A2[-1,-2,1]*operator[-3,1];
        @tensor AA_fused[:]:=A1'[1,4,3]*A2[2,5,3]*U_L[-1,1,2]*U_U[4,5,-2];
    end


    return AA_fused, U_L,U_U
end





function build_double_layer_bulk_open(A0,return_unitary::Bool)

    A_=permute(A0,(1,2,),(3,4,5,));
    U_L=@ignore_derivatives unitary(fuse(space(A_, 1)' ⊗ space(A_, 1)), space(A_, 1)' ⊗ space(A_, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A_, 2)' ⊗ space(A_, 2)), space(A_, 2)' ⊗ space(A_, 2))*(1+0*im);
    # U_R=(U_L)';
    # U_U=(U_D)';
    U_R=@ignore_derivatives unitary(space(A_, 3) ⊗ space(A_, 3)', fuse(space(A_, 3)' ⊗ space(A_, 3)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A_, 4) ⊗ space(A_, 4)', fuse(space(A_, 4)' ⊗ space(A_, 4)))*(1+0*im);

    V_D=@ignore_derivatives space(A0, 4);
    V_s=@ignore_derivatives space(A0, 5);

    A=permute(A0,(1,2,3,4,5,));

    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);

    # uM_dag,sM_dag,vM_dag=tsvd(permute(A_fused',(1,2,3,),(4,5,6,)));
    # uM_dag=uM_dag*sM_dag;
    Ap=permute(A0',(1,2,5,),(3,4,));
    Up_tem=@ignore_derivatives unitary(fuse(space(Ap,1)*space(Ap,2)*space(Ap,3)), space(Ap,1)*space(Ap,2)*space(Ap,3))*(1+0*im);
    vM_dag=Up_tem*Ap;
    uM_dag=Up_tem';


    U_tem=@ignore_derivatives unitary(fuse(space(A0,1)*space(A0,2)), space(A0,1)*space(A0,2))*(1+0*im);
    vM=U_tem*permute(A0,(1,2,),(3,4,5,));
    uM=U_tem';

    
    uM_dag=permute(uM_dag,(1,2,3,4,),());
    uM=permute(uM,(1,2,3,),());
    Vp=space(vM_dag,1);
    V=space(vM,1);
    U=@ignore_derivatives unitary(fuse(Vp ⊗ V), Vp ⊗ V);
    @tensor double_LD[:]:=uM_dag[-1,-2,-3,1]*U'[1,-4,-5];
    @tensor double_LD[:]:=double_LD[-1,-3,-5,1,-6]*uM[-2,-4,1];

    vM_dag=permute(vM_dag,(1,2,3,),());
    vM=permute(vM,(1,2,3,4,));
    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    @tensor double_RU[:]:=vM_dag[1,-2,-4]*double_RU[-1,1,-3,-5,-6];

    double_LD=permute(double_LD,(1,2,),(3,4,5,6,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,5,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,3,4),());#L,D,physical,virtual

    double_RU=permute(double_RU,(1,2,3,6,),(4,5,));
    double_RU=double_RU*U_U;
    @tensor double_RU[:]:=double_RU[-1,1,2,-4,-3]*U_R[1,2,-2];

    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);


    @tensor AA_open[:]:=double_LD[-1,-2,1,3]*double_RU[3,-3,-4,2]*U_s_s[-5,1,2];

    U_s_s=U_s_s';

    if return_unitary
        AA_open, U_s_s, U_L,U_D,U_R,U_U
    else
        return AA_open, U_s_s 
    end

end




function build_double_layer_top_open(A1,return_unitary::Bool)
    A2=A1;

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A1, 2)' ⊗ space(A2, 2)), space(A1, 2)' ⊗ space(A2, 2))*(1+0*im);

    U_R=@ignore_derivatives unitary(space(A1, 3) ⊗ space(A2, 3)', fuse(space(A1, 3)' ⊗ space(A2, 3)))*(1+0*im);

    V_s=@ignore_derivatives space(A2, 4);
    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);

    @tensor AA_open[:]:=A1'[1,6,4,0]*A2[2,7,5,3]*U_L[-1,1,2]*U_D[-2,6,7]*U_R[4,5,-3]*U_s_s[-4,0,3];

    U_s_s=U_s_s';

    if return_unitary
        AA_open, U_s_s, U_L,U_D,U_R,nothing
    else
        return AA_open, U_s_s 
    end
end

function build_double_layer_bot_open(A1,return_unitary::Bool)
    A2=A1;

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);

    U_R=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A1, 3) ⊗ space(A2, 3)', fuse(space(A1, 3)' ⊗ space(A2, 3)))*(1+0*im);

    V_s=@ignore_derivatives space(A2, 4);
    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);

    @tensor AA_open[:]:=A1'[1,4,6,0]*A2[2,5,7,3]*U_L[-1,1,2]*U_U[6,7,-3]*U_R[4,5,-2]*U_s_s[-4,0,3];

    U_s_s=U_s_s';

    if return_unitary
        AA_open, U_s_s, U_L,nothing,U_R,U_U
    else
        return AA_open, U_s_s 
    end
end


function build_double_layer_left_open(A1,return_unitary::Bool)
    A2=A1;

    U_D=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);

    U_R=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A1, 3) ⊗ space(A2, 3)', fuse(space(A1, 3)' ⊗ space(A2, 3)))*(1+0*im);

    V_s=@ignore_derivatives space(A2, 4);
    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);

    @tensor AA_open[:]:=A1'[1,4,6,0]*A2[2,5,7,3]*U_U[6,7,-3]*U_D[-1,1,2]*U_R[4,5,-2]*U_s_s[-4,0,3];

    U_s_s=U_s_s';

    if return_unitary
        AA_open, U_s_s, nothing,U_D,U_R,U_U
    else
        return AA_open, U_s_s 
    end
end

function build_double_layer_right_open(A1,return_unitary::Bool)
    A2=A1;

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A1, 2)' ⊗ space(A2, 2)), space(A1, 2)' ⊗ space(A2, 2))*(1+0*im);

    U_U=@ignore_derivatives unitary(space(A1, 3) ⊗ space(A2, 3)', fuse(space(A1, 3)' ⊗ space(A2, 3)))*(1+0*im);

    V_s=@ignore_derivatives space(A2, 4);
    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);

    @tensor AA_open[:]:=A1'[1,4,6,0]*A2[2,5,7,3]*U_L[-1,1,2]*U_D[-2,4,5]*U_U[6,7,-3]*U_s_s[-4,0,3];

    U_s_s=U_s_s';

    if return_unitary
        AA_open, U_s_s, U_L,U_D,nothing,U_U
    else
        return AA_open, U_s_s 
    end
end



function build_double_layer_left_top_open(A1,return_unitary::Bool)
    A2=A1;


    U_D=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_R=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);

    V_s=@ignore_derivatives space(A2, 3);
    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);

    @tensor AA_open[:]:=A1'[1,4,0]*A2[2,5,3]*U_D[-1,1,2]*U_R[4,5,-2]*U_s_s[-3,0,3];

    U_s_s=U_s_s';

    if return_unitary
        AA_open, U_s_s, nothing,U_D,U_R,nothing
    else
        return AA_open, U_s_s 
    end
end

function build_double_layer_right_top_open(A1,return_unitary::Bool)
    A2=A1;

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A1, 2)' ⊗ space(A2, 2)), space(A1, 2)' ⊗ space(A2, 2))*(1+0*im);

    V_s=@ignore_derivatives space(A2, 3);
    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);

    @tensor AA_open[:]:=A1'[1,4,0]*A2[2,5,3]*U_L[-1,1,2]*U_D[-2,4,5]*U_s_s[-3,0,3];

    U_s_s=U_s_s';

    if return_unitary
        AA_open, U_s_s, U_L,U_D,nothing,nothing
    else
        return AA_open, U_s_s 
    end
end

function build_double_layer_left_bot_open(A1,return_unitary::Bool)
    A2=A1;

    U_R=@ignore_derivatives unitary(space(A1, 1) ⊗ space(A2, 1)', fuse(space(A1, 1)' ⊗ space(A2, 1)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);

    V_s=@ignore_derivatives space(A2, 3);
    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);

    @tensor AA_open[:]:=A1'[1,4,0]*A2[2,5,3]*U_R[1,2,-1]*U_U[4,5,-2]*U_s_s[-3,0,3];

    U_s_s=U_s_s';

    if return_unitary
        AA_open, U_s_s, nothing,nothing,U_R,U_U
    else
        return AA_open, U_s_s 
    end
end

function build_double_layer_right_bot_open(A1,return_unitary::Bool)
    A2=A1;

    U_L=@ignore_derivatives unitary(fuse(space(A1, 1)' ⊗ space(A2, 1)), space(A1, 1)' ⊗ space(A2, 1))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A1, 2) ⊗ space(A2, 2)', fuse(space(A1, 2)' ⊗ space(A2, 2)))*(1+0*im);

    V_s=@ignore_derivatives space(A2, 3);
    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);

    @tensor AA_open[:]:=A1'[1,4,0]*A2[2,5,3]*U_L[-1,1,2]*U_U[4,5,-2]*U_s_s[-3,0,3];

    U_s_s=U_s_s';

    if return_unitary
        AA_open, U_s_s, U_L,nothing,nothing,U_U
    else
        return AA_open, U_s_s 
    end
end


function build_double_layer_open_position(T,px,py,Lx,Ly,return_unitary::Bool)
    if (1<px)&(px<Lx)
        xp="bulk";
    elseif (px==1)
        xp="left";
    elseif (px==Lx)
        xp="right";
    end

    if (1<py)&(py<Ly)
        yp="bulk";
    elseif (py==1)
        yp="bot";
    elseif (py==Ly)
        yp="top";
    end

    if ~return_unitary
        if xp=="bulk"
            if yp=="bulk"
                AA_open,U_s_s=build_double_layer_bulk_open(T,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s=build_double_layer_top_open(T,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s=build_double_layer_bot_open(T,return_unitary);
            end
        elseif xp=="left"
            if yp=="bulk"
                AA_open,U_s_s=build_double_layer_left_open(T,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s=build_double_layer_left_top_open(T,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s=build_double_layer_left_bot_open(T,return_unitary);
            end
        elseif xp=="right"
            if yp=="bulk"
                AA_open,U_s_s=build_double_layer_right_open(T,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s=build_double_layer_right_top_open(T,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s=build_double_layer_right_bot_open(T,return_unitary);
            end
        end
        return AA_open

    else

        if xp=="bulk"
            if yp=="bulk"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_bulk_open(T,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_top_open(T,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_bot_open(T,return_unitary);
            end
        elseif xp=="left"
            if yp=="bulk"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_left_open(T,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_left_top_open(T,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_left_bot_open(T,return_unitary);
            end
        elseif xp=="right"
            if yp=="bulk"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_right_open(T,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_right_top_open(T,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_right_bot_open(T,return_unitary);
            end
        end
        return AA_open,U_L,U_D,U_R,U_U
    end
    
end