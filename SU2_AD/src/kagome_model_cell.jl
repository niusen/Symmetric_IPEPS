                            
function evaluate_ob_minimal(cx,cy, parameters, U_phy, A_fused_cell::Tuple,PEPS_2x2_special, AA_fused_cell::Tuple, CTM_cell::NamedTuple, ctm_setting, kagome_method,E_up_method="2x2",E_dn_method="simplfied")
    construct_double_layer=ctm_setting.construct_double_layer;
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    global Lx,Ly
    @assert construct_double_layer==true
    @assert kagome_method=="E_triangle" #calculate up and down triangle energy
    norm_1site=ob_1x1(CTM_cell,AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+1,Ly)],cx,cy);

    H_triangle, H_Heisenberg, H12_tensorkit, H31_tensorkit, H23_tensorkit=@ignore_derivatives Hamiltonians(U_phy,parameters["J1"],parameters["J2"],parameters["J3"],parameters["Jchi"],parameters["Jtrip"])
    
    # up triangle
    if E_up_method=="1x1"
        AA_H, _,_,_,_=build_double_layer(A_fused_cell[mod1(cx+1,Lx)][mod1(cy+1,Ly)],H_triangle);
        E_up=ob_1x1(CTM_cell,AA_H,cx,cy);
        E_up=E_up/norm_1site;

    elseif E_up_method=="2x2"
        AA_H, _,_,_,_=build_double_layer(A_fused_cell[mod1(cx+1,Lx)][mod1(cy+1,Ly)],H_triangle);
        E_up=ob_2x2(CTM_cell,AA_H,AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+1,Ly)],AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+2,Ly)],AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+2,Ly)],cx,cy);
        E_up_norm=ob_2x2(CTM_cell,AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+1,Ly)],AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+1,Ly)],AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+2,Ly)],AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+2,Ly)],cx,cy);
        E_up=E_up/E_up_norm;
    end

    # down triangle
    if  E_dn_method=="simplified"
        U_phy2=@ignore_derivatives unitary(fuse(space(U_phy, 3) ⊗ space(U_phy, 3)), space(U_phy, 3) ⊗ space(U_phy, 3));
        U_phy5=@ignore_derivatives unitary(fuse(space(U_phy2, 1) ⊗ space(U_phy, 1)), space(U_phy2, 1) ⊗ space(U_phy, 1));
        U_22=@ignore_derivatives unitary(space(U_phy2,1)',space(U_phy2,1)');
        @tensor H_triangle_enlarged[:]:=U_22[1,3]*H_triangle[2,4]*U_phy5[-2,3,4]*U_phy5'[1,2,-1];

        PEPS_LU=PEPS_2x2_special.LU;
        PEPS_RU=PEPS_2x2_special.RU;
        PEPS_LD=PEPS_2x2_special.LD;

        if ctm_setting.grad_checkpoint
            AA_LU, L_LU,D_LU,R_LU,U_LU=Zygote.checkpointed(build_double_layer, PEPS_LU, H_triangle_enlarged);
            #AA_LU, L_LU,D_LU,R_LU,U_LU=build_double_layer(PEPS_LU, []);
            AA_LD, L_LD,D_LD,R_LD,U_LD=Zygote.checkpointed(build_double_layer, PEPS_LD, []);
            AA_RU, L_RU,D_RU,R_RU,U_RU=Zygote.checkpointed(build_double_layer, PEPS_RU, []);
        else
            AA_LU, L_LU,D_LU,R_LU,U_LU=build_double_layer(PEPS_LU, H_triangle_enlarged);
            #AA_LU, L_LU,D_LU,R_LU,U_LU=build_double_layer(PEPS_LU, []);
            AA_LD, L_LD,D_LD,R_LD,U_LD=build_double_layer(PEPS_LD, []);
            AA_RU, L_RU,D_RU,R_RU,U_RU=build_double_layer(PEPS_RU, []);
        end
        E_down=ob_2x2(CTM_cell,AA_LU,AA_RU,AA_LD,AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+2,Ly)],cx,cy);
        norm_dn=ob_2x2(CTM_cell,AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+1,Ly)],AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+1,Ly)],AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+2,Ly)],AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+2,Ly)],cx,cy);
        E_down=E_down/norm_dn;
    end
    return E_up, E_down   
    
end

function build_special_PEPS(iPESS_tensors_cell,cx,cy)#for evaluating down triangle energy
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    Pos_LU= (mod1(cx+1,Lx),mod1(cy+1,Ly));
    Pos_LD= (mod1(cx+1,Lx),mod1(cy+2,Ly));
    Pos_RU= (mod1(cx+2,Lx),mod1(cy+1,Ly));
    Pos_RD= (mod1(cx+2,Lx),mod1(cy+2,Ly));
    
    global Lx,Ly
    B1_=iPESS_tensors_cell[1,1].B1;
    U_phy=@ignore_derivatives unitary(fuse(space(B1_, 3) ⊗ space(B1_, 3) ⊗ space(B1_, 3)), space(B1_, 3) ⊗ space(B1_, 3) ⊗ space(B1_, 3));
    U_phy2=@ignore_derivatives unitary(fuse(space(B1_, 3) ⊗ space(B1_, 3)), space(B1_, 3) ⊗ space(B1_, 3));
    U_phy5=@ignore_derivatives unitary(fuse(space(U_phy2, 1) ⊗ space(U_phy, 1)), space(U_phy2, 1) ⊗ space(U_phy, 1));
    U_22=@ignore_derivatives unitary(space(U_phy2,1)',space(U_phy2,1)');


    @tensor PEPS_LU_a_[:]:= iPESS_tensors_cell[Pos_LU[1], Pos_LU[2]].B1[-1,1,-4]*iPESS_tensors_cell[Pos_LU[1], Pos_LU[2]].B3[-3,2,-5]*iPESS_tensors_cell[Pos_LU[1], Pos_LU[2]].Tup[1,-2,2];
    @tensor PEPS_LU_a[:]:=PEPS_LU_a_[-1,-2,-3,1,2]*U_phy2[-4,1,2];


    @tensor PEPS_LU_b_[:]:= iPESS_tensors_cell[Pos_LU[1], Pos_LU[2]].B2[1,-1,-4]*iPESS_tensors_cell[Pos_LD[1], Pos_LD[2]].B3[2,-2,-5]*iPESS_tensors_cell[Pos_RU[1], Pos_RU[2]].B1[3,-3,-6]*iPESS_tensors_cell[Pos_LU[1], Pos_LU[2]].Tdn[3,1,2];
    @tensor PEPS_LU_b[:]:=PEPS_LU_b_[-1,-2,-3,1,2,3]*U_phy[-4,1,2,3];
    @tensor PEPS_LU[:]:=PEPS_LU_a[-1,1,-4,2]*PEPS_LU_b[1,-2,-3,3]*U_phy5[-5,2,3];


    @tensor PEPS_LD[:]:= iPESS_tensors_cell[Pos_LD[1], Pos_LD[2]].B1[-1,1,4]*iPESS_tensors_cell[Pos_LD[1], Pos_LD[2]].B2[3,2,5]*iPESS_tensors_cell[Pos_LD[1], Pos_LD[2]].Tup[1,2,-4]*iPESS_tensors_cell[Pos_LD[1], Pos_LD[2]].Tdn[-3,3,-2]*U_phy2[-5,4,5];


    @tensor PEPS_RU[:] := iPESS_tensors_cell[Pos_RU[1], Pos_RU[2]].B2[3,2,4]*iPESS_tensors_cell[Pos_RU[1], Pos_RU[2]].B3[-4,1,5]*iPESS_tensors_cell[Pos_RU[1], Pos_RU[2]].Tup[-1,2,1]*iPESS_tensors_cell[Pos_RU[1], Pos_RU[2]].Tdn[-3,3,-2]*U_phy2[-5,4,5];

    PEPS_2x2_special=(LU=PEPS_LU,RU=PEPS_RU,LD=PEPS_LD);#The size of tuple is fixed to be 2x2, independent on Lx,Ly, since evaluating energy uses 2x2 cell 
    return PEPS_2x2_special
end


function evaluate_ob(parameters, U_phy, iPESS_tensors_cell, A_fused_cell::Tuple, AA_fused_cell, CTM_cell, ctm_setting, kagome_method,E_up_method="2x2",E_dn_method="simplfied")
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    global Lx,Ly
    E_total=0;
    E_up_cell=zeros(Lx,Ly);
    E_down_cell=zeros(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            PEPS_2x2_special=build_special_PEPS(iPESS_tensors_cell,cx,cy);
            E_up_, E_down_=evaluate_ob_minimal(cx,cy,parameters, U_phy, A_fused_cell, PEPS_2x2_special, AA_fused_cell, CTM_cell, ctm_setting, kagome_method,E_up_method,E_dn_method);
            E_total=E_total+real(E_up_)+real(E_down_);
            @ignore_derivatives E_up_cell[cx,cy]=real(E_up_);
            @ignore_derivatives E_down_cell[cx,cy]=real(E_down_);
        end
    end
    println(E_up_cell)
    println(E_down_cell)
    return E_total, E_up_cell, E_down_cell
end




function ob_1x1(CTM,AA,cx,cy)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,-1]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[2,-2,1]*Cset[mod1(cx,Lx)][mod1(cy+2,Ly)].C4[-3,2];
    @tensor envR[:]:=Cset[mod1(cx+2,Lx)][mod1(cy,Ly)].C2[-1,1]*Tset[mod1(cx+2,Lx)][mod1(cy+1,Ly)].T2[1,-2,2]*Cset[mod1(cx+2,Lx)][mod1(cy+2,Ly)].C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[1,3,-1]*AA[2,5,-2,3]*Tset[mod1(cx+1,Lx)][mod1(cy+2,Ly)].T3[-3,5,4];
    Norm=@tensor envL[1,2,3]*envR[1,2,3];

    return Norm;
end



function ob_2x2(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_LU_[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_LD_[3,4,-5,-3]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    rho=@tensor up[1,2,3,4,]*down[1,2,3,4];
    return rho
end