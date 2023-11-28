
function evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    H_Heisenberg, H123chiral, H12, H31, H23 =@ignore_derivatives Hamiltonians();
    J1=parameters["J1"];
    J2=parameters["J2"];
    Jchi=parameters["Jchi"];
    

    global Lx,Ly
    AA_open_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            if ctm_setting.grad_checkpoint
                AA_open, U_s_s=Zygote.checkpointed(build_double_layer_open, A_cell[cx][cy]);
            else
                AA_open, U_s_s =build_double_layer_open(A_cell[cx][cy]);
            end
            AA_open_cell=fill_tuple(AA_open_cell, AA_open, cx,cy);
        end
    end

    if energy_setting.model=="triangle_J1_J2_Jchi";
        H_triangle=J1/4*(H12+H31)+J2/2*H23+Jchi*H123chiral;

        V_s=@ignore_derivatives space(A_cell[1][1], 5);
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s)';
        E_LU_RU_LD_set=zeros(Lx,Ly)*im;
        E_LD_RU_RD_set=zeros(Lx,Ly)*im;
        E_LU_LD_RD_set=zeros(Lx,Ly)*im;
        E_LU_RU_RD_set=zeros(Lx,Ly)*im;

        E_total=0;
        for cx=1:Lx
            for cy=1:Ly
                pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
                pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
                pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
                pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

                rho_LU_RU_LD=ob_LU_RU_LD_cell(cx,cy,CTM_cell,AA_cell[pos_RD[1]][pos_RD[2]],AA_open_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_LD[1]][pos_LD[2]]);  #clockwise
                @tensor rho_LU_RU_LD[:]:=rho_LU_RU_LD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
                norm_LU_RU_LD=@tensor rho_LU_RU_LD[1,2,3,1,2,3];
                E_LU_RU_LD=@tensor rho_LU_RU_LD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
                E_LU_RU_LD=E_LU_RU_LD/norm_LU_RU_LD; 
                @ignore_derivatives E_LU_RU_LD_set[cx,cy]=E_LU_RU_LD;
                
            
                rho_LD_RU_RD=ob_LD_RU_RD_cell(cx,cy,CTM_cell,AA_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_LD[1]][pos_LD[2]],AA_open_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_RD[1]][pos_RD[2]]);  #clockwise
                rho_LD_RU_RD=permute(rho_LD_RU_RD,(3,1,2,));
                @tensor rho_LD_RU_RD[:]:=rho_LD_RU_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
                norm_LD_RU_RD=@tensor rho_LD_RU_RD[1,2,3,1,2,3];
                E_LD_RU_RD=@tensor rho_LD_RU_RD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
                E_LD_RU_RD=E_LD_RU_RD/norm_LD_RU_RD ; 
                @ignore_derivatives E_LD_RU_RD_set[cx,cy]=E_LD_RU_RD;
                
            
                rho_LU_LD_RD=ob_LU_LD_RD_cell(cx,cy,CTM_cell,AA_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_LD[1]][pos_LD[2]],AA_open_cell[pos_RD[1]][pos_RD[2]]);  #clockwise
                rho_LU_LD_RD=permute(rho_LU_LD_RD,(2,1,3,));
                @tensor rho_LU_LD_RD[:]:=rho_LU_LD_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
                norm_LU_LD_RD=@tensor rho_LU_LD_RD[1,2,3,1,2,3];
                E_LU_LD_RD=@tensor rho_LU_LD_RD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
                E_LU_LD_RD=E_LU_LD_RD/norm_LU_LD_RD ; 
                @ignore_derivatives E_LU_LD_RD_set[cx,cy]=E_LU_LD_RD;
                
            
                rho_LU_RU_RD=ob_LU_RU_RD_cell(cx,cy,CTM_cell,AA_cell[pos_LD[1]][pos_LD[2]],AA_open_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_RD[1]][pos_RD[2]]);  #clockwise
                rho_LU_RU_RD=permute(rho_LU_RU_RD,(2,3,1,));
                @tensor rho_LU_RU_RD[:]:=rho_LU_RU_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
                norm_LU_RU_RD=@tensor rho_LU_RU_RD[1,2,3,1,2,3];
                E_LU_RU_RD=@tensor rho_LU_RU_RD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
                E_LU_RU_RD=E_LU_RU_RD/norm_LU_RU_RD;
                @ignore_derivatives E_LU_RU_RD_set[cx,cy]=E_LU_RU_RD;

                
                
                E_total=E_total+real(E_LU_RU_LD)+real(E_LD_RU_RD)+real(E_LU_LD_RD)+real(E_LU_RU_RD);
                
            end
        end
        println(E_LU_RU_LD_set)
        println(E_LD_RU_RD_set)
        println(E_LU_LD_RD_set)
        println(E_LU_RU_RD_set)
        return E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set
    end
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



function ob_LU_RU_LD_cell(cx,cy,CTM,AA_,AA_LU_,AA_RU_,AA_LD_)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-4]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-2,4,1]*AA_LU_[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_LD_[3,4,-5,-3,-1]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[-1,1,2,3,4,-2]*down[-3,1,2,3,4];
    return rho
end

function ob_LD_RU_RD_cell(cx,cy,CTM,AA_,AA_LD_,AA_RU_,AA_RD_)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_LD_[3,4,-5,-3,-1]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[1,2,3,4,-2]*down[-1,1,2,3,4,-3];
    return rho
end

function ob_LU_LD_RD_cell(cx,cy,CTM,AA_,AA_LU_,AA_LD_,AA_RD_)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-4]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-2,4,1]*AA_LU_[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_[-2,-4,4,3]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_LD_[3,4,-5,-3,-1]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[-1,1,2,3,4]*down[-2,1,2,3,4,-3];
    return rho
end

function ob_LU_RU_RD_cell(cx,cy,CTM,AA_,AA_LU_,AA_RU_,AA_RD_)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-4]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-2,4,1]*AA_LU_[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_[3,4,-5,-3]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[-1,1,2,3,4,-2]*down[1,2,3,4,-3];
    return rho
end
