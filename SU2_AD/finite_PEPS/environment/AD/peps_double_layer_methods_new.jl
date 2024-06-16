
function construct_double_layer_open(psi_bra,psi_ket)
    psi=deepcopy(psi_ket);
    Lx=size(psi,1);
    Ly=size(psi,2);
    psi_double=Matrix{TensorMap}(undef,Lx,Ly);
    return_unitary=false;
    for cx=2:Lx-1
        for cy=2:Ly-1
            AA,U_s_s=build_double_layer_bulk_open(psi_bra[cx,cy],psi_ket[cx,cy],return_unitary);
            psi_double[cx,cy]=AA;
        end
    end

    cx=1;
    for cy=2:Ly-1
        AA,U_s_s=build_double_layer_left_open(psi_bra[cx,cy],psi_ket[cx,cy],return_unitary);
        psi_double[cx,cy]=AA;
    end

    cx=Lx;
    for cy=2:Ly-1
        AA,U_s_s=build_double_layer_right_open(psi_bra[cx,cy],psi_ket[cx,cy],return_unitary);
        psi_double[cx,cy]=AA;
    end

    cy=1;
    for cx=2:Lx-1
        AA,U_s_s=build_double_layer_bot_open(psi_bra[cx,cy],psi_ket[cx,cy],return_unitary);
        psi_double[cx,cy]=AA;
    end

    cy=Ly;
    for cx=2:Lx-1
        AA,U_s_s=build_double_layer_top_open(psi_bra[cx,cy],psi_ket[cx,cy],return_unitary);
        psi_double[cx,cy]=AA;
    end

    cx=1;
    cy=1;
    AA,U_s_s=build_double_layer_left_bot_open(psi_bra[cx,cy],psi_ket[cx,cy],return_unitary);
    psi_double[cx,cy]=AA;

    cx=1;
    cy=Ly;
    AA,U_s_s=build_double_layer_left_top_open(psi_bra[cx,cy],psi_ket[cx,cy],return_unitary);
    psi_double[cx,cy]=AA;

    cx=Lx;
    cy=1;
    AA,U_s_s=build_double_layer_right_bot_open(psi_bra[cx,cy],psi_ket[cx,cy],return_unitary);
    psi_double[cx,cy]=AA;

    cx=Lx;
    cy=Ly;
    AA,U_s_s=build_double_layer_right_top_open(psi_bra[cx,cy],psi_ket[cx,cy],return_unitary);
    psi_double[cx,cy]=AA;

    return psi_double, U_s_s
end

function build_double_layer_bulk_open(A_bra, A_ket, return_unitary::Bool)
    
    A_=permute(A_ket,(1,2,),(3,4,5,));
    A_bra=permute(A_bra,(1,2,),(3,4,5));
    U_L=@ignore_derivatives unitary(fuse(space(A_bra, 1)' ⊗ space(A_, 1)), space(A_bra, 1)' ⊗ space(A_, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A_bra, 2)' ⊗ space(A_, 2)), space(A_bra, 2)' ⊗ space(A_, 2))*(1+0*im);
    # U_R=(U_L)';
    # U_U=(U_D)';
    U_R=@ignore_derivatives unitary(space(A_bra, 3) ⊗ space(A_, 3)', fuse(space(A_bra, 3)' ⊗ space(A_, 3)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A_bra, 4) ⊗ space(A_, 4)', fuse(space(A_bra, 4)' ⊗ space(A_, 4)))*(1+0*im);

    V_D=@ignore_derivatives space(A_ket, 4);
    V_s=@ignore_derivatives space(A_ket, 5);

    # A=permute(A0,(1,2,3,4,5,));

    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);


    Ap=permute(permute(A_bra,(1,2,3,4,5,))',(1,2,5,),(3,4,));
    Up_tem=@ignore_derivatives unitary(fuse(space(Ap,1)*space(Ap,2)*space(Ap,3)), space(Ap,1)*space(Ap,2)*space(Ap,3))*(1+0*im);
    vM_dag=Up_tem*Ap;
    uM_dag=Up_tem';


    U_tem=@ignore_derivatives unitary(fuse(space(A_ket,1)*space(A_ket,2)), space(A_ket,1)*space(A_ket,2))*(1+0*im);
    vM=U_tem*permute(A_ket,(1,2,),(3,4,5,));
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





function build_double_layer_top_open(A1,A2,return_unitary::Bool)

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

function build_double_layer_bot_open(A1,A2,return_unitary::Bool)

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


function build_double_layer_left_open(A1,A2,return_unitary::Bool)

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

function build_double_layer_right_open(A1,A2,return_unitary::Bool)

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



function build_double_layer_left_top_open(A1,A2,return_unitary::Bool)

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

function build_double_layer_right_top_open(A1,A2,return_unitary::Bool)

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

function build_double_layer_left_bot_open(A1,A2,return_unitary::Bool)

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

function build_double_layer_right_bot_open(A1,A2,return_unitary::Bool)

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


function build_double_layer_open_position(T1,T2,px,py,Lx,Ly,return_unitary)
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
                AA_open,U_s_s=build_double_layer_bulk_open(T1,T2,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s=build_double_layer_top_open(T1,T2,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s=build_double_layer_bot_open(T1,T2,return_unitary);
            end
        elseif xp=="left"
            if yp=="bulk"
                AA_open,U_s_s=build_double_layer_left_open(T1,T2,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s=build_double_layer_left_top_open(T1,T2,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s=build_double_layer_left_bot_open(T1,T2,return_unitary);
            end
        elseif xp=="right"
            if yp=="bulk"
                AA_open,U_s_s=build_double_layer_right_open(T1,T2,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s=build_double_layer_right_top_open(T1,T2,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s=build_double_layer_right_bot_open(T1,T2,return_unitary);
            end
        end
        return AA_open

    else

        if xp=="bulk"
            if yp=="bulk"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_bulk_open(T1,T2,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_top_open(T1,T2,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_bot_open(T1,T2,return_unitary);
            end
        elseif xp=="left"
            if yp=="bulk"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_left_open(T1,T2,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_left_top_open(T1,T2,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_left_bot_open(T1,T2,return_unitary);
            end
        elseif xp=="right"
            if yp=="bulk"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_right_open(T1,T2,return_unitary);
            elseif yp=="top"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_right_top_open(T1,T2,return_unitary);
            elseif yp=="bot"
                AA_open,U_s_s,U_L,U_D,U_R,U_U=build_double_layer_right_bot_open(T1,T2,return_unitary);
            end
        end
        return AA_open,U_L,U_D,U_R,U_U
    end
    
end