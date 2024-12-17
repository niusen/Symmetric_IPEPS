function Rank(T::TensorMap)
    return length(domain(T))+length(codomain(T))
end


function Hamiltonian_SU2_SU2()
    #SU(4) P_{ij} operator
    P_ij=zeros(4,4,4,4);#d1',d2',d1,d2
    for ca=1:4
        for cb=1:4
            P_ij[ca,cb,cb,ca]=1;
        end
    end

    Vp=Rep[SU₂ × SU₂]((1/2,1/2)=>1);
    P_ij=TensorMap(P_ij,Vp*Vp,Vp*Vp);

    P_ijk=zeros(4,4,4,4,4,4);#d1',d2',d3',d1,d2,d3
    for ca=1:4
        for cb=1:4
            for cc=1:4
                P_ijk[ca,cb,cc,cb,cc,ca]=1;
            end
        end
    end
    P_ijk=TensorMap(P_ijk,Vp*Vp*Vp,Vp*Vp*Vp);

    P_kji=permute(P_ijk,(3,2,1,),(6,5,4,));

    return P_ij,P_ijk,P_kji
end


function build_double_layer_open(A0)

    A_=permute(A0,(1,2,),(3,4,5));
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

    return AA_open, U_s_s 

end

function ob_2x2(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_LU_[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-1]*AA_LD_[3,4,-4,-2]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-3,4,2]; 
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


function rho_2x2_LD_RD_RU(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_LU_[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-1]*AA_LD_[3,4,-4,-2,-5]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4,-5]; 

    @tensor up[:]:=MM_LU[-1,-2,1,2]*MM_RU[1,2,-3,-4,-5];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4,-6];
    @tensor rho[:]:= up[1,2,3,4,-3]*down[1,2,3,4,-1,-2];
    return rho
end


function rho_2x2_LD_RU_LU(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_LU_[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-1]*AA_LD_[3,4,-4,-2,-5]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-3,4,2];  
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4]; 

    @tensor up[:]:=MM_LU[-1,-2,1,2,-5]*MM_RU[1,2,-3,-4,-6];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4];
    @tensor rho[:]:= up[1,2,3,4,-3,-2]*down[1,2,3,4,-1];
    return rho
end



function evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    
    global Lx,Ly

    if isa(space(A_cell[1][1],1),GradedSpace{ProductSector{Tuple{SU2Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{SU2Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonian_SU2_SU2;
    end

    if energy_setting.model=="triangle_SU4_spin"
  
        #for 120 degree magnetic order in the Hofstadter M2 model. Unit-cell for 120 degree order should be at least 3x3.  
        P_ij, P_ijk, P_kji = @ignore_derivatives Hamiltonian_terms();
        J=parameters["J"];
        K=parameters["K"];
        Φ=parameters["Φ"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonal_set=zeros(Lx,Ly)*im;
        triangle_right_bot_set=zeros(Lx,Ly)*im;
        triangle_left_top_set=zeros(Lx,Ly)*im;

        E_total=0;

  
        AA_open_cell=initial_tuple_cell(Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                global U_ss
                AA_open,U_ss=build_double_layer_open(A_cell[cx][cy]);
                AA_open_cell=fill_tuple(AA_open_cell, AA_open, cx,cy);
            end
        end

        @tensor P_ij[:]:=P_ij[1,3,2,4]*U_ss[1,2,-1]*U_ss[3,4,-2];
        @tensor P_ijk[:]:=P_ijk[1,3,5,2,4,6]*U_ss[1,2,-1]*U_ss[3,4,-2]*U_ss[5,6,-3];
        @tensor P_kji[:]:=P_kji[1,3,5,2,4,6]*U_ss[1,2,-1]*U_ss[3,4,-2]*U_ss[5,6,-3];
        

        for cx=1:Lx
            for cy=1:Ly

                pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
                pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
                pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
                pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];
            

                # rho_LD_RD_RU=rho_2x2_LD_RD_RU(CTM_cell,AA_LU,AA_RU_open,AA_LD_open,AA_RD_open,cx,cy);
                # rho_LD_RU_LU=rho_2x2_LD_RU_LU(CTM_cell,AA_LU_open,AA_RU_open,AA_LD_open,AA_RD,cx,cy);
                rho_LD_RD_RU=rho_2x2_LD_RD_RU(CTM_cell,AA_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_LD[1]][pos_LD[2]],AA_open_cell[pos_RD[1]][pos_RD[2]],cx,cy);
                rho_LD_RU_LU=rho_2x2_LD_RU_LU(CTM_cell,AA_open_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);

                @tensor rho_LD_RD[:]:=rho_LD_RD_RU[-1,-2,1]*U_ss[2,2,1];
                @tensor rho_RD_RU[:]:=rho_LD_RD_RU[1,-1,-2]*U_ss[2,2,1];
                @tensor rho_RU_LD[:]:=rho_LD_RD_RU[-2,1,-1]*U_ss[2,2,1];

                ex=@tensor rho_LD_RD[1,2]*P_ij[1,2];
                ey=@tensor rho_RD_RU[1,2]*P_ij[1,2];
                e_diagonal=@tensor rho_RU_LD[1,2]*P_ij[1,2];
                triangle_right_bot=@tensor rho_LD_RD_RU[1,2,3]*P_ijk[1,2,3];
                triangle_left_top=@tensor rho_LD_RU_LU[1,2,3]*P_ijk[1,2,3];

                @ignore_derivatives ex_set[cx,cy]=ex;
                @ignore_derivatives ey_set[cx,cy]=ey;
                @ignore_derivatives e_diagonal_set[cx,cy]=e_diagonal;
                @ignore_derivatives triangle_right_bot_set[cx,cy]=triangle_right_bot;
                @ignore_derivatives triangle_left_top_set[cx,cy]=triangle_left_top;

                E_total=E_total+J*real(ex+ey+e_diagonal) +3*K*cos(Φ)*real(triangle_left_top+triangle_left_top'+triangle_right_bot+triangle_right_bot') +3*K*sin(Φ)*imag(im*triangle_left_top-im*triangle_left_top'+im*triangle_right_bot-im*triangle_right_bot');
                
            end
        end

        E_total=E_total/(Lx*Ly);
        return E_total,  ex_set, ey_set, e_diagonal_set, triangle_right_bot_set, triangle_left_top_set
    end
end











