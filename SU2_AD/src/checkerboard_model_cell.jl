function evaluate_ob_minimal(cx,cy, parameters, U_phy, A_fused_cell::Tuple,PEPS_2x2_special, AA_fused_cell::Tuple, CTM_cell::NamedTuple, ctm_setting,)
    construct_double_layer=ctm_setting.construct_double_layer;
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    global Lx,Ly
    @assert construct_double_layer==true

    H_heisenberg, SS_op12, SS_op13, SS_op14, SS_op23, SS_op24, SS_op34=@ignore_derivatives plaquatte_Heisenberg(parameters["J1"],parameters["J2"]);
    H_fused=@ignore_derivatives fuse_H(H_heisenberg,U_phy);
    H_=permute(H_fused',(1,),(2,));
    @assert construct_double_layer==true


    PEPS_LU=PEPS_2x2_special.LU;
    PEPS_RU=PEPS_2x2_special.RU;
    PEPS_LD=PEPS_2x2_special.LD;

    if ctm_setting.grad_checkpoint
        AA_LU, L_LU,D_LU,R_LU,U_LU=Zygote.checkpointed(build_double_layer, PEPS_LU, H_);
        #AA_LU, L_LU,D_LU,R_LU,U_LU=build_double_layer(PEPS_LU, []);
        AA_LD, L_LD,D_LD,R_LD,U_LD=Zygote.checkpointed(build_double_layer, PEPS_LD, []);
        AA_RU, L_RU,D_RU,R_RU,U_RU=Zygote.checkpointed(build_double_layer, PEPS_RU, []);
    else
        AA_LU, L_LU,D_LU,R_LU,U_LU=build_double_layer(PEPS_LU, H_);
        #AA_LU, L_LU,D_LU,R_LU,U_LU=build_double_layer(PEPS_LU, []);
        AA_LD, L_LD,D_LD,R_LD,U_LD=build_double_layer(PEPS_LD, []);
        AA_RU, L_RU,D_RU,R_RU,U_RU=build_double_layer(PEPS_RU, []);
    end
    E_plaquatte=ob_2x2(CTM_cell,AA_LU,AA_RU,AA_LD,AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+2,Ly)],cx,cy);
    norm_=ob_2x2(CTM_cell,AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+1,Ly)],AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+1,Ly)],AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+2,Ly)],AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+2,Ly)],cx,cy);
    E_plaquatte=E_plaquatte/norm_;

    return E_plaquatte  

end

function build_special_PEPS(iPESS_tensors_cell,U_phy,cx,cy)#for evaluating plaquatte energy
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

    U_U_phy=@ignore_derivatives unitary(fuse(space(U_phy,1)*space(U_phy,1)), space(U_phy,1)*space(U_phy,1));
    @tensor ALU[:]:=iPESS_tensors_cell[Pos_LU[1], Pos_LU[2]].B_L[-1,5,1]*iPESS_tensors_cell[Pos_LU[1], Pos_LU[2]].B_U[-4,6,2]*iPESS_tensors_cell[Pos_RU[1], Pos_RU[2]].B_L[8,-3,3]*iPESS_tensors_cell[Pos_LD[1], Pos_LD[2]].B_U[7,-2,4]*iPESS_tensors_cell[Pos_LU[1], Pos_LU[2]].Tm[5,7,8,6]*U_phy[9,1,2]*U_phy[10,3,4]*U_U_phy[-5,9,10];
    @tensor ALD[:]:=iPESS_tensors_cell[Pos_LD[1], Pos_LD[2]].B_L[-1,1,-5]*iPESS_tensors_cell[Pos_LD[1], Pos_LD[2]].Tm[1,-2,-3,-4];
    @tensor ARU[:]:=iPESS_tensors_cell[Pos_RU[1], Pos_RU[2]].B_U[-4,1,-5]*iPESS_tensors_cell[Pos_RU[1], Pos_RU[2]].Tm[-1,-2,-3,1];

    PEPS_2x2_special=(LU=ALU,RU=ARU,LD=ALD);#The size of tuple is fixed to be 2x2, independent on Lx,Ly, since evaluating energy uses 2x2 cell 
    return PEPS_2x2_special
end


function evaluate_ob(parameters, U_phy, iPESS_tensors_cell, A_fused_cell::Tuple, AA_fused_cell, CTM_cell, ctm_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    global Lx,Ly
    E_total=0;
    E_plaquatte_cell=zeros(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            PEPS_2x2_special=build_special_PEPS(iPESS_tensors_cell,U_phy,cx,cy);
            E_plaquatte=evaluate_ob_minimal(cx,cy,parameters, U_phy, A_fused_cell, PEPS_2x2_special, AA_fused_cell, CTM_cell, ctm_setting);
            E_total=E_total+real(E_plaquatte);
            @ignore_derivatives E_plaquatte_cell[cx,cy]=real(E_plaquatte);
        end
    end
    println(E_plaquatte_cell)
    return E_total, E_plaquatte_cell
end


function bond_energy_minimal(cx,cy, U_phy, A_fused_cell::Tuple,PEPS_2x2_special, AA_fused_cell::Tuple, CTM_cell::NamedTuple, ctm_setting,)
    construct_double_layer=ctm_setting.construct_double_layer;
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    global Lx,Ly
    @assert construct_double_layer==true

    H_heisenberg, SS_op12, SS_op13, SS_op14, SS_op23, SS_op24, SS_op34=@ignore_derivatives plaquatte_Heisenberg(parameters["J1"],parameters["J2"]);

    @assert construct_double_layer==true


    PEPS_LU=PEPS_2x2_special.LU;
    PEPS_RU=PEPS_2x2_special.RU;
    PEPS_LD=PEPS_2x2_special.LD;

    Hset=(SS_op12, SS_op13, SS_op14, SS_op23, SS_op24, SS_op34);
    E_set=zeros(6);
    for co in eachindex(Hset)
        H_fused=@ignore_derivatives fuse_H(Hset[co],U_phy);
        H_=permute(H_fused',(1,),(2,));

        if ctm_setting.grad_checkpoint
            AA_LU, L_LU,D_LU,R_LU,U_LU=Zygote.checkpointed(build_double_layer, PEPS_LU, H_);
            #AA_LU, L_LU,D_LU,R_LU,U_LU=build_double_layer(PEPS_LU, []);
            AA_LD, L_LD,D_LD,R_LD,U_LD=Zygote.checkpointed(build_double_layer, PEPS_LD, []);
            AA_RU, L_RU,D_RU,R_RU,U_RU=Zygote.checkpointed(build_double_layer, PEPS_RU, []);
        else
            AA_LU, L_LU,D_LU,R_LU,U_LU=build_double_layer(PEPS_LU, H_);
            #AA_LU, L_LU,D_LU,R_LU,U_LU=build_double_layer(PEPS_LU, []);
            AA_LD, L_LD,D_LD,R_LD,U_LD=build_double_layer(PEPS_LD, []);
            AA_RU, L_RU,D_RU,R_RU,U_RU=build_double_layer(PEPS_RU, []);
        end
        E_bond=ob_2x2(CTM_cell,AA_LU,AA_RU,AA_LD,AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+2,Ly)],cx,cy);
        norm_=ob_2x2(CTM_cell,AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+1,Ly)],AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+1,Ly)],AA_fused_cell[mod1(cx+1,Lx)][mod1(cy+2,Ly)],AA_fused_cell[mod1(cx+2,Lx)][mod1(cy+2,Ly)],cx,cy);
        E_bond=E_bond/norm_;
        E_set[co]=real(E_bond);
    end

    return E_set 

end

function bond_energy(U_phy, iPESS_tensors_cell, A_fused_cell::Tuple, AA_fused_cell, CTM_cell, ctm_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    global Lx,Ly
    E_total=0;
    E_plaquatte_cell=Matrix{Vector}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            PEPS_2x2_special=build_special_PEPS(iPESS_tensors_cell,U_phy,cx,cy);
            E_set=bond_energy_minimal(cx,cy, U_phy, A_fused_cell, PEPS_2x2_special, AA_fused_cell, CTM_cell, ctm_setting);
            @ignore_derivatives E_plaquatte_cell[cx,cy]=E_set;
        end
    end
    
    return E_plaquatte_cell
end


function fuse_H(H,U_phy)
    U_U_phy=@ignore_derivatives unitary(fuse(space(U_phy,1)*space(U_phy,1)), space(U_phy,1)*space(U_phy,1));
    @tensor H_fused[:]:=H[1,2,3,4,7,8,9,10]*U_phy[11,7,8]*U_phy[12,9,10]*U_phy'[1,2,5]*U_phy'[3,4,6]*U_U_phy[-1,11,12]*U_U_phy'[5,6,-2];
    return H_fused
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





