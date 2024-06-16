"""
  |        |
--A'=======----
  |        ||
  |        ||
-----------A----
  |        |
  |        |
"""


function construct_single_layer(psi1,psi2)
    psi1=deepcopy(psi1);
    psi2=deepcopy(psi2);
    Lx=size(psi1,1);
    Ly=size(psi1,2);
    psi_single=Matrix{TensorMap}(undef,2*Lx,2*Ly);
    for cx=2:Lx-1
        for cy=2:Ly-1
            A_LU,A_RU,A_LD,A_RD,_=build_single_layer_bulk(psi1[cx,cy],psi2[cx,cy],[]);
            psi_single[2*cx-1,2*cy-1]=A_LD;
            psi_single[2*cx-1,2*cy]=A_LU;
            psi_single[2*cx,2*cy-1]=A_RD;
            psi_single[2*cx,2*cy]=A_RU;
        end
    end

    cx=1;
    for cy=2:Ly-1
        A_LU,A_RU,A_LD,A_RD,_=build_single_layer_left(psi1[cx,cy],psi2[cx,cy],[]);
        psi_single[2*cx-1,2*cy-1]=A_LD;
        psi_single[2*cx-1,2*cy]=A_LU;
        psi_single[2*cx,2*cy-1]=A_RD;
        psi_single[2*cx,2*cy]=A_RU;
    end

    cx=Lx;
    for cy=2:Ly-1
        A_LU,A_RU,A_LD,A_RD,_=build_single_layer_right(psi1[cx,cy],psi2[cx,cy],[]);
        psi_single[2*cx-1,2*cy-1]=A_LD;
        psi_single[2*cx-1,2*cy]=A_LU;
        psi_single[2*cx,2*cy-1]=A_RD;
        psi_single[2*cx,2*cy]=A_RU;
    end

    cy=1;
    for cx=2:Lx-1
        A_LU,A_RU,A_LD,A_RD,_=build_single_layer_bot(psi1[cx,cy],psi2[cx,cy],[]);
        psi_single[2*cx-1,2*cy-1]=A_LD;
        psi_single[2*cx-1,2*cy]=A_LU;
        psi_single[2*cx,2*cy-1]=A_RD;
        psi_single[2*cx,2*cy]=A_RU;
    end

    cy=Ly;
    for cx=2:Lx-1
        A_LU,A_RU,A_LD,A_RD,_=build_single_layer_top(psi1[cx,cy],psi2[cx,cy],[]);
        psi_single[2*cx-1,2*cy-1]=A_LD;
        psi_single[2*cx-1,2*cy]=A_LU;
        psi_single[2*cx,2*cy-1]=A_RD;
        psi_single[2*cx,2*cy]=A_RU;
    end

    cx=1;
    cy=1;
    A_LU,A_RU,A_LD,A_RD,_=build_single_layer_left_bot(psi1[cx,cy],psi2[cx,cy],[]);
    psi_single[2*cx-1,2*cy-1]=A_LD;
    psi_single[2*cx-1,2*cy]=A_LU;
    psi_single[2*cx,2*cy-1]=A_RD;
    psi_single[2*cx,2*cy]=A_RU;

    cx=1;
    cy=Ly;
    A_LU,A_RU,A_LD,A_RD,_=build_single_layer_left_top(psi1[cx,cy],psi2[cx,cy],[]);
    psi_single[2*cx-1,2*cy-1]=A_LD;
    psi_single[2*cx-1,2*cy]=A_LU;
    psi_single[2*cx,2*cy-1]=A_RD;
    psi_single[2*cx,2*cy]=A_RU;

    cx=Lx;
    cy=1;
    A_LU,A_RU,A_LD,A_RD,_=build_single_layer_right_bot(psi1[cx,cy],psi2[cx,cy],[]);
    psi_single[2*cx-1,2*cy-1]=A_LD;
    psi_single[2*cx-1,2*cy]=A_LU;
    psi_single[2*cx,2*cy-1]=A_RD;
    psi_single[2*cx,2*cy]=A_RU;

    cx=Lx;
    cy=Ly;
    A_LU,A_RU,A_LD,A_RD,_=build_single_layer_right_top(psi1[cx,cy],psi2[cx,cy],[]);
    psi_single[2*cx-1,2*cy-1]=A_LD;
    psi_single[2*cx-1,2*cy]=A_LU;
    psi_single[2*cx,2*cy-1]=A_RD;
    psi_single[2*cx,2*cy]=A_RU;

    return psi_single
end


function build_single_layer_bulk(A1,A2,operator)
    U_R=@ignore_derivatives unitary(fuse(space(A1',3)*space(A1',5)),space(A1',3)*space(A1',5));
    U_U=@ignore_derivatives unitary(fuse(space(A2,4)*space(A2,5)),space(A2,4)*space(A2,5));
    id_phy=@ignore_derivatives unitary(space(A1',5),space(A1',5));
    id_y_RU=@ignore_derivatives unitary(space(A2,4),space(A2,4));
    id_x_RU=@ignore_derivatives unitary(space(A1',3),space(A1',3));
    id_y_LD=@ignore_derivatives unitary(space(A1',2),space(A1',2));
    id_x_LD=@ignore_derivatives unitary(space(A2,1),space(A2,1));

    @tensor A_LU[:]:=A1'[-1,-2,1,-4,2]*U_R[-3,1,2]; 
    @tensor A_RD[:]:=A2[-1,-2,-3,1,2]*U_U[-4,1,2];
    if operator==[]
    else
        @tensor id_phy[:]:=id_phy[1,-2]*operator[-1,1];
    end
    @tensor A_RU[:]:= id_x_RU[-3,1]*id_y_RU[-4,4]*id_phy[5,2]*U_R'[1,2,-1]*U_U'[4,5,-2];
    @tensor A_LD[:]:=id_x_LD[-1,-3]*id_y_LD[-2,-4];

    return A_LU,A_RU,A_LD,A_RD, U_R,U_U
end



function build_single_layer_top(A1,A2,operator)

    U_R=@ignore_derivatives unitary(fuse(space(A1',3)*space(A1',4)),space(A1',3)*space(A1',4));
    id_phy=@ignore_derivatives unitary(space(A1',4),space(A1',4));
    id_x_RU=@ignore_derivatives unitary(space(A1',3),space(A1',3));
    id_y_LD=@ignore_derivatives unitary(space(A1',2),space(A1',2));
    id_x_LD=@ignore_derivatives unitary(space(A2,1),space(A2,1));

    @tensor A_LU[:]:=A1'[-1,-2,1,2]*U_R[-3,1,2]; 
    A_RD=deepcopy(A2);
    if operator==[]
    else
        @tensor id_phy[:]:=id_phy[1,-2]*operator[-1,1];
    end
    @tensor A_RU[:]:= id_x_RU[-3,1]*id_phy[-2,2]*U_R'[1,2,-1];
    @tensor A_LD[:]:=id_x_LD[-1,-3]*id_y_LD[-2,-4];

    return A_LU,A_RU,A_LD,A_RD, U_R
end



function build_single_layer_bot(A1,A2,operator)
    #add an extra leg to A1
    U_extra=@ignore_derivatives unitary(space(A1,1)*Rep[SU₂](0=>1)', space(A1,1));
    verify_U_extra(U_extra);
    @tensor A1[:]:=A1[1,-3,-4,-5]*U_extra[-1,-2,1];

    U_R=@ignore_derivatives unitary(fuse(space(A1',3)*space(A1',5)),space(A1',3)*space(A1',5));
    U_U=@ignore_derivatives unitary(fuse(space(A2,3)*space(A2,4)),space(A2,3)*space(A2,4));
    id_phy=@ignore_derivatives unitary(space(A1',5),space(A1',5));
    id_y_RU=@ignore_derivatives unitary(space(A2,3),space(A2,3));
    id_x_RU=@ignore_derivatives unitary(space(A1',3),space(A1',3));
    id_x_LD=@ignore_derivatives unitary(space(A2,1),space(A2,1));

    @tensor A_LU[:]:=A1'[-1,-2,1,-4,2]*U_R[-3,1,2]; 
    @tensor A_RD[:]:=A2[-1,-2,1,2]*U_U[-3,1,2];
    if operator==[]
    else
        @tensor id_phy[:]:=id_phy[1,-2]*operator[-1,1];
    end
    @tensor A_RU[:]:= id_x_RU[-3,1]*id_y_RU[-4,4]*id_phy[5,2]*U_R'[1,2,-1]*U_U'[4,5,-2];
    @tensor A_LD[:]:=id_x_LD[-1,-2];
    U_extra=@ignore_derivatives unitary(space(A_LD,1)*Rep[SU₂](0=>1)', space(A_LD,1));
    verify_U_extra(U_extra);
    @tensor A_LD[:]:=A_LD[1,-2]*U_extra[-1,-3,1];
    

    return A_LU,A_RU,A_LD,A_RD, U_R,U_U
end



function build_single_layer_left(A1,A2,operator)
    U_R=@ignore_derivatives unitary(fuse(space(A1',2)*space(A1',4)),space(A1',2)*space(A1',4));
    U_U=@ignore_derivatives unitary(fuse(space(A2,3)*space(A2,4)),space(A2,3)*space(A2,4));
    id_phy=@ignore_derivatives unitary(space(A1',4),space(A1',4));
    id_y_RU=@ignore_derivatives unitary(space(A2,3),space(A2,3));
    id_x_RU=@ignore_derivatives unitary(space(A1',2),space(A1',2));
    id_y_LD=@ignore_derivatives unitary(space(A1',1),space(A1',1));

    #add an extra leg to A2
    U_extra=@ignore_derivatives unitary(Rep[SU₂](0=>1)*space(A2,1), space(A2,1));
    verify_U_extra(U_extra);
    @tensor A2[:]:=A2[1,-3,-4,-5]*U_extra[-1,-2,1];

    @tensor A_LU[:]:=A1'[-2,1,-4,2]*U_R[-3,1,2]; 
    @tensor A_RD[:]:=A2[-1,-2,-3,1,2]*U_U[-4,1,2];
    if operator==[]
    else
        @tensor id_phy[:]:=id_phy[1,-2]*operator[-1,1];
    end
    @tensor A_RU[:]:= id_x_RU[-3,1]*id_y_RU[-4,4]*id_phy[5,2]*U_R'[1,2,-1]*U_U'[4,5,-2];
    @tensor A_LD[:]:=id_y_LD[-1,-2];
    U_extra=@ignore_derivatives unitary(space(A_LD,1)*Rep[SU₂](0=>1)', space(A_LD,1));
    verify_U_extra(U_extra);
    @tensor A_LD[:]:=A_LD[1,-3]*U_extra[-1,-2,1];

    return A_LU,A_RU,A_LD,A_RD, U_R,U_U
end



function build_single_layer_right(A1,A2,operator)
    U_U=@ignore_derivatives unitary(fuse(space(A2,3)*space(A2,4)),space(A2,3)*space(A2,4));
    id_phy=@ignore_derivatives unitary(space(A1',4),space(A1',4));
    id_y_RU=@ignore_derivatives unitary(space(A2,3),space(A2,3));
    id_y_LD=@ignore_derivatives unitary(space(A1',2),space(A1',2));
    id_x_LD=@ignore_derivatives unitary(space(A2,1),space(A2,1));

    A_LU=permute(A1',(1,2,4,3,)); 
    @tensor A_RD[:]:=A2[-1,-2,1,2]*U_U[-4,1,2];
    if operator==[]
    else
        @tensor id_phy[:]:=id_phy[1,-2]*operator[-1,1];
    end
    @tensor A_RU[:]:= id_y_RU[-3,4]*id_phy[5,-1]*U_U'[4,5,-2];
    @tensor A_LD[:]:=id_x_LD[-1,-3]*id_y_LD[-2,-4];

    return A_LU,A_RU,A_LD,A_RD, U_U
end





function build_single_layer_left_top(A1,A2,operator)
    U_R=@ignore_derivatives unitary(fuse(space(A1',2)*space(A1',3)),space(A1',2)*space(A1',3));
    id_phy=@ignore_derivatives unitary(space(A1',3),space(A1',3));
    id_x_RU=@ignore_derivatives unitary(space(A1',2),space(A1',2));
    id_y_LD=@ignore_derivatives unitary(space(A1',1),space(A1',1));

    #add an extra leg to A2
    U_extra=@ignore_derivatives unitary(Rep[SU₂](0=>1)*space(A2,1), space(A2,1));
    verify_U_extra(U_extra);
    @tensor A2[:]:=A2[1,-3,-4]*U_extra[-1,-2,1];

    @tensor A_LU[:]:=A1'[-1,1,2]*U_R[-2,1,2]; 
    A_RD=deepcopy(A2);
    if operator==[]
    else
        @tensor id_phy[:]:=id_phy[1,-2]*operator[-1,1];
    end
    @tensor A_RU[:]:= id_x_RU[-3,1]*id_phy[-2,2]*U_R'[1,2,-1];
    @tensor A_LD[:]:=id_y_LD[-1,-2];
    U_extra=@ignore_derivatives unitary(space(A_LD,1)*Rep[SU₂](0=>1)', space(A_LD,1));
    verify_U_extra(U_extra);
    @tensor A_LD[:]:=A_LD[1,-3]*U_extra[-1,-2,1];

    return A_LU,A_RU,A_LD,A_RD, U_R
end



function build_single_layer_right_top(A1,A2,operator)

    id_phy=@ignore_derivatives unitary(space(A1',3),space(A1',3));
    id_y_LD=@ignore_derivatives unitary(space(A1',2),space(A1',2));
    id_x_LD=@ignore_derivatives unitary(space(A2,1),space(A2,1));

    A_LU=deepcopy(A1'); 
    A_RD=deepcopy(A2);
    if operator==[]
    else
        @tensor id_phy[:]:=id_phy[1,-2]*operator[-1,1];
    end
    @tensor A_RU[:]:=id_phy[-2,-1];
    @tensor A_LD[:]:=id_x_LD[-1,-3]*id_y_LD[-2,-4];

    return A_LU,A_RU,A_LD,A_RD, nothing
end



function build_single_layer_left_bot(A1,A2,operator)
    #add an extra leg to A1
    U_extra=@ignore_derivatives unitary(Rep[SU₂](0=>1)*space(A1',1), space(A1',1));
    verify_U_extra(U_extra);
    @tensor A1[:]:=A1[1,-3,-4]*U_extra'[1,-1,-2];

    #add an extra leg to A2
    U_extra=@ignore_derivatives unitary(Rep[SU₂](0=>1)'*space(A2,1), space(A2,1));
    verify_U_extra(U_extra);
    @tensor A2[:]:=A2[1,-3,-4]*U_extra[-1,-2,1];

    U_R=@ignore_derivatives unitary(fuse(space(A1',2)*space(A1',4)),space(A1',2)*space(A1',4));
    U_U=@ignore_derivatives unitary(fuse(space(A2,3)*space(A2,4)),space(A2,3)*space(A2,4));
    id_phy=@ignore_derivatives unitary(space(A1',4),space(A1',4));
    id_y_RU=@ignore_derivatives unitary(space(A2,3),space(A2,3));
    id_x_RU=@ignore_derivatives unitary(space(A1',2),space(A1',2));
    id_y_LD=@ignore_derivatives unitary(space(A1',1),space(A1',1));
    id_x_LD=@ignore_derivatives unitary(space(A2,1),space(A2,1));



    @tensor A_LU[:]:=A1'[-1,1,-4,2]*U_R[-3,1,2]; 
    @tensor A_RD[:]:=A2[-1,-3,1,2]*U_U[-4,1,2];
    if operator==[]
    else
        @tensor id_phy[:]:=id_phy[1,-2]*operator[-1,1];
    end
    @tensor A_RU[:]:= id_x_RU[-3,1]*id_y_RU[-4,4]*id_phy[5,2]*U_R'[1,2,-1]*U_U'[4,5,-2];
    @tensor A_LD[:]:=id_x_LD[1,-1]*id_y_LD[1,-2];

    return A_LU,A_RU,A_LD,A_RD, U_R,U_U
end



function build_single_layer_right_bot(A1,A2,operator)
    #add an extra leg to A1
    U_extra=@ignore_derivatives unitary(space(A1',1)*Rep[SU₂](0=>1), space(A1',1));
    verify_U_extra(U_extra);
    @tensor A1[:]:=A1[1,-3,-4]*U_extra'[1,-1,-2];

    U_U=@ignore_derivatives unitary(fuse(space(A2,2)*space(A2,3)),space(A2,2)*space(A2,3));
    id_phy=@ignore_derivatives unitary(space(A1',4),space(A1',4));
    id_y_RU=@ignore_derivatives unitary(space(A2,2),space(A2,2));
    id_x_LD=@ignore_derivatives unitary(space(A2,1),space(A2,1));

    A_LU=permute(A1',(1,2,4,3,)); 
    @tensor A_RD[:]:=A2[-1,1,2]*U_U[-2,1,2];
    if operator==[]
    else
        @tensor id_phy[:]:=id_phy[1,-2]*operator[-1,1];
    end
    @tensor A_RU[:]:= id_y_RU[-3,4]*id_phy[5,-1]*U_U'[4,5,-2];
    @tensor A_LD[:]:=id_x_LD[-1,-2];
    U_extra=@ignore_derivatives unitary(space(A_LD,2)*Rep[SU₂](0=>1)', space(A_LD,2));
    verify_U_extra(U_extra);
    @tensor A_LD[:]:=A_LD[-1,1]*U_extra[-2,-3,1];

    return A_LU,A_RU,A_LD,A_RD, U_U
end



function verify_U_extra(U)
    U_=convert(Array,U);
    U_=reshape(U_,(size(U_,1)*size(U_,2),size(U_,3),));
    @assert norm(U_-Matrix(I,size(U_,1),size(U_,2)))<1e-15;

end