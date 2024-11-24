




function evaluate_correl(pos,coe,direction, double_B_set,double_T_set, double_B_set_op1,double_T_set_op1, double_B_set_op2,double_T_set_op2, CTM, distance)
    correl_funs=Vector(undef,distance);
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    px=pos[1];
    py=pos[2];
    global Lx,Ly
    
    if direction=="x"
        C1=Cset[mod1(px-1,Lx)][mod1(py-1,Ly)].C1;
        T4=Tset[mod1(px-1,Lx)][mod1(py,Ly)].T4;
        C4=Cset[mod1(px-1,Lx)][mod1(py+1,Ly)].C4;
        T1=Tset[mod1(px,Lx)][mod1(py-1,Ly)].T1;
        T3=Tset[mod1(px,Lx)][mod1(py+1,Ly)].T3;
        BB=double_B_set_op1[mod1(px,Lx)];
        TT=double_T_set_op1[mod1(px,Lx)];
        # @tensor AA[:]:=BB[-1,1,-4]*TT[-2,-3,1];#(L M U),(D R M) =>(L,D,R,U)
        @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[8,2]*T1[3,4,-1]*BB[5,7,4]*TT[6,-2,7]*T3[-3,6,8];
        # println([norm(AA),norm(va)])


        qx=px+1;
        qy=py;
        T1=Tset[mod1(qx,Lx)][mod1(qy-1,Ly)].T1;
        T3=Tset[mod1(qx,Lx)][mod1(qy+1,Ly)].T3;
        C2=Cset[mod1(qx+1,Lx)][mod1(qy-1,Ly)].C2;
        T2=Tset[mod1(qx+1,Lx)][mod1(qy,Ly)].T2;
        C3=Cset[mod1(qx+1,Lx)][mod1(qy+1,Ly)].C3;
        BB=double_B_set_op2[mod1(qx,Lx)];
        TT=double_T_set_op2[mod1(qx,Lx)];
        @tensor vb[:]:=T1[-1,6,5]*BB[-2,8,6]*TT[4,3,8]*T3[2,4,-3]*C2[5,7]*T2[7,3,1]*C3[1,2];

        

        ov=@tensor va[1,2,3]*vb[1,2,3]
        # println(norm(ov))
        correl_funs[1]=ov;
        
        for dis=2:distance

            qx=px+dis-1;
            qy=py;
            T1=Tset[mod1(qx,Lx)][mod1(qy-1,Ly)].T1;
            T3=Tset[mod1(qx,Lx)][mod1(qy+1,Ly)].T3;
            BB=double_B_set[mod1(qx,Lx)];
            TT=double_T_set[mod1(qx,Lx)];
            @tensor va[:]:=va[1,2,6]*T1[1,3,-1]*BB[2,5,3]*TT[4,-2,5]*T3[-3,4,6];
            va=va*coe;

            qx=px+dis;
            qy=py;
            T1=Tset[mod1(qx,Lx)][mod1(qy-1,Ly)].T1;
            T3=Tset[mod1(qx,Lx)][mod1(qy+1,Ly)].T3;
            C2=Cset[mod1(qx+1,Lx)][mod1(qy-1,Ly)].C2;
            T2=Tset[mod1(qx+1,Lx)][mod1(qy,Ly)].T2;
            C3=Cset[mod1(qx+1,Lx)][mod1(qy+1,Ly)].C3;
            BB=double_B_set_op2[mod1(qx,Lx)];
            TT=double_T_set_op2[mod1(qx,Lx)];
            #@tensor vb[:]:=T1[-1,4,3]*BB[-2,6,5,4]*TT[]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
            @tensor vb[:]:=T1[-1,6,5]*BB[-2,8,6]*TT[4,3,8]*T3[2,4,-3]*C2[5,7]*T2[7,3,1]*C3[1,2];
            ov=@tensor va[1,2,3]*vb[1,2,3]
            correl_funs[dis]=ov;
        end
        return correl_funs
    end

end


function correl_TransOp_x(vl,Tset,AAfused_cell,py,Lx,Ly)
    if AAfused_cell==[]
        for cx=1:Lx
            Tup=Tset[cx][mod1(py,Ly)].T1;
            Tdn=Tset[cx][mod1(py+1,Ly)].T3;
            Tup=Tup/norm(Tup)*10;
            Tdn=Tdn/norm(Tdn)*10;
            @tensor vl[:]:=vl[-1,1,3]*Tup[1,2,-2]*Tdn[-3,2,3];
        end
    end
    return vl
end
function solve_correl_length_simple(n_values,Vspace,CTM_cell,direction,Lx,Ly,partly)
    Tset=CTM_cell.Tset;
    Q_set=[];
    if direction=="x"
        if partly
            y_range=1:1;
        else
            y_range=1:Ly;
        end
        eu_cell=Vector{Any}(undef,length(y_range));
        for cy in y_range
            correl_TransOp_fx(x)=correl_TransOp_x(x,Tset,[],cy,Lx,Ly);

            if typeof(Vspace)==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}#SU2
                Q_set=[0,1/2,1,3/2,2];
            elseif typeof(Vspace)==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}#Z2
                Q_set=[0,1];
            end
            eu_set=Vector{Vector}(undef,length(Q_set));
            for cq=1:length(Q_set)
                if typeof(Vspace)==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}#SU2
                    Vp=Rep[SU₂](Q_set[cq]=>1);
                elseif typeof(Vspace)==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}#Z2
                    Vp=Rep[ℤ₂](Q_set[cq]=>1);
                end

                vl_init = permute(TensorMap(randn, Vp⊗space(Tset[1][mod1(cy,Ly)].T1,1)', space(Tset[1][mod1(cy+1,Ly)].T3,3)), (1,2,3,),());
                if norm(vl_init)>0
                    eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                    eu_set[cq]=eu;
                else
                    eu_set[cq]=[];
                end
            end
            eu_cell[cy]=eu_set;
        end
    
        return eu_cell,Q_set
    end
end


function build_AA_spin(S1L,S1R,B_set,T_set,ca,cb)

    B0=B_set[ca,cb];#(LU,M)
    T0=T_set[ca,cb];#(M,dRD)
    @tensor T_new[:]:= T0[-1,1,-3,-4]*S1L[-5,-2,1];#M,d,R,D,virtual
    U1=@ignore_derivatives unitary(fuse(space(T_new,3)⊗space(T_new,5)), space(T_new,3)⊗space(T_new,5)); 
    @tensor T_new[:]:=T_new[-1,-2,1,-4,2]*U1[-3,1,2];#M,d,R',D
    T_new=permute(T_new,(1,),(2,3,4,));
    B_double_spin_L, _ = build_double_layer_swap_Tm(B0',B0, false);#L M U
    T_double_spin_L, _ = build_double_layer_swap_Bm(T0',T_new, true);#D R M
    

    B0=B_set[mod1(ca+1,Lx),cb];#(LU,M)
    T0=T_set[mod1(ca+1,Lx),cb];#(M,dRD)
    Id=unitary(space(S1R,1),space(S1R,1));
    U12=@ignore_derivatives unitary(fuse(space(B0,3)⊗space(Id,2)), space(B0,3)⊗space(Id,2)); #M
    @tensor B_new[:]:=B0[1,-2,3]*Id[2,4]*U1'[1,2,-1]*U12[-3,3,4];#L,U,M
    U2=@ignore_derivatives unitary(fuse(space(T0,3)⊗space(Id,2)), space(T0,3)⊗space(Id,2)); #R
    @tensor T_new[:]:=T0[1,-2,3,-4]*Id[2,4]*U12'[1,2,-1]*U2[-3,3,4];#M,d,R,D
    B_new=permute(B_new,(1,2,),(3,));
    T_new=permute(T_new,(1,),(2,3,4,));
    B_double_spin_mid, _ = build_double_layer_swap_Tm(B0',B_new, false);#L M U
    T_double_spin_mid, _ = build_double_layer_swap_Bm(T0',T_new, true);#D R M
    

    B0=B_set[mod1(ca+2,Lx),cb];#(LU,M)
    T0=T_set[mod1(ca+2,Lx),cb];#(M,dRD)
    U23=@ignore_derivatives unitary(fuse(space(B0,3)⊗space(Id,2)), space(B0,3)⊗space(Id,2)); #M
    @tensor B_new[:]:=B0[1,-2,3]*Id[2,4]*U2'[1,2,-1]*U23[-3,3,4];#L,U,M
    @tensor T_new[:]:= T0[-1,1,-3,-4]*S1R[-5,-2,1];#M,d,R,D,virtual
    @tensor T_new[:]:=T_new[1,-2,-3,-4,2]*U23'[1,2,-1];#M',d,R,D
    B_new=permute(B_new,(1,2,),(3,));
    T_new=permute(T_new,(1,),(2,3,4,));
    B_double_spin_R, _ = build_double_layer_swap_Tm(B0',B_new, false);#L M U
    T_double_spin_R, _ = build_double_layer_swap_Bm(T0',T_new, true);#D R M


    return B_double_spin_L,T_double_spin_L, B_double_spin_mid,T_double_spin_mid, B_double_spin_R,T_double_spin_R
    
end

function build_AA_hop(Cdag,C,B_set,T_set,ca,cb)
    #the first index of O is dummy

    B0=B_set[ca,cb];#(LU,M)
    T0=T_set[ca,cb];#(M,dRD)
    @tensor T_new[:]:= T0[-1,1,-3,-4]*Cdag[-5,-2,1];#M,d,R,D,virtual
    U1=unitary(fuse(space(T_new,3)⊗space(T_new,5)), space(T_new,3)⊗space(T_new,5)); #
    gate=parity_gate(B0,1); @tensor B_new[:]:=B0[1,-2,-3]*gate[-1,1];#L,U,M
    gate=parity_gate(T_new,4); @tensor T_new[:]:=T_new[-1,-2,-3,1,-5]*gate[-4,1];#M,d,R,D,virtual
    gate=parity_gate(B_new,2); @tensor B_new[:]:=B_new[-1,1,-3]*gate[-2,1];
    @tensor T_new[:]:=T_new[-1,-2,1,-4,2]*U1[-3,1,2];#M,d,R',D
    B_new=permute(B_new,(1,2,),(3,));
    T_new=permute(T_new,(1,),(2,3,4,));
    B_double_CdagC_L, _ = build_double_layer_swap_Tm(B0',B_new, false);#L M U
    T_double_CdagC_L, _ = build_double_layer_swap_Bm(T0',T_new, true);#D R M

    

    B0=B_set[mod1(ca+1,Lx),cb];#(LU,M)
    T0=T_set[mod1(ca+1,Lx),cb];#(M,dRD)
    O_string=unitary(space(Cdag,1),space(Cdag,1));
    gate=parity_gate(T0,4); @tensor T_new[:]:=T0[-1,-2,-3,1]*gate[-4,1];#D
    U12=unitary(fuse(space(B0,3)⊗space(O_string,2)'), space(B0,3)⊗space(O_string,2)'); 
    @tensor B_new[:]:=B0[1,-2,3]*O_string[4,2]*U1'[1,2,-1]*U12[-3,3,4];#L,U,M
    U2=unitary(fuse(space(T_new,3)⊗space(O_string,2)'), space(T_new,3)⊗space(O_string,2)'); 
    @tensor T_new[:]:=T_new[1,-2,3,-4]*O_string[4,2]*U12'[1,2,-1]*U2[-3,3,4];#M,d,R,D
    B_new=permute(B_new,(1,2,),(3,));
    T_new=permute(T_new,(1,),(2,3,4,));
    B_double_CdagC_mid, _ = build_double_layer_swap_Tm(B0',B_new, false);#L M U
    T_double_CdagC_mid, _ = build_double_layer_swap_Bm(T0',T_new, true);#D R M
    


    B0=B_set[mod1(ca+2,Lx),cb];#(LU,M)
    T0=T_set[mod1(ca+2,Lx),cb];#(M,dRD)
    @tensor T_new[:]:= T0[-1,1,-3,-4]*C[-5,-2,1];#M,d,R,D,virtual
    gate=parity_gate(B0,1); @tensor B_new[:]:=B0[1,-2,-3]*gate[-1,1];#L
    gate=parity_gate(B_new,2); @tensor B_new[:]:=B_new[-1,1,-3]*gate[-2,1];#U
    U23=unitary(fuse(space(B_new,3)⊗space(O_string,2)'), space(B_new,3)⊗space(O_string,2)'); 
    @tensor B_new[:]:=B_new[1,-2,3]*O_string[4,2]*U2'[1,2,-1]*U23[-3,3,4];#L,U,M
    @tensor T_new[:]:=T_new[1,-2,-3,-4,2]*U23'[1,2,-1];#M,d,R,D
    B_new=permute(B_new,(1,2,),(3,));
    T_new=permute(T_new,(1,),(2,3,4,));
    B_double_CdagC_R, _ = build_double_layer_swap_Tm(B0',B_new, false);#L M U
    T_double_CdagC_R, _ = build_double_layer_swap_Bm(T0',T_new, true);#D R M

    

    return B_double_CdagC_L,T_double_CdagC_L, B_double_CdagC_mid,T_double_CdagC_mid, B_double_CdagC_R,T_double_CdagC_R
end


function cal_correl(CTM_cell,B_set,T_set,B_double_set, T_double_set,D,chi,parameters,direction,distance,partly)
    global Lx,Ly
    Vspace=space(B_set[1],1)
    if typeof(Vspace)==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}#SU2
        Ident_set, N_occu_set, n_hole_set, n_double_set, Cdag_set, C_set, CdagupCdagdn_set, Pairinga_set, Pairingb_set, Sa_set, Sb_set=Operators_spinful_SU2();
    elseif typeof(Vspace)==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}#Z2
        Ident_set, N_occu_set, n_hole_set, n_double_set, Cdag_set, C_set, CdagupCdagdn_set, Pairinga_set, Pairingb_set, Sa_set, Sb_set=Operators_spinful_Z2();
    end
    

    if partly
        x_range=1:1;
        y_range=1:1;
    else
        x_range=1:Lx;
        y_range=1:Ly;
    end

    S1L=Sa_set[1]; 
    S1R=Sb_set[1];
    Cdag=Cdag_set[1];
    C=C_set[1];

    SS_ob_set=Matrix{Vector}(undef,length(x_range),length(y_range));
    CdagC_ob_set=Matrix{Vector}(undef,length(x_range),length(y_range));

    if direction=="x"
        n_values=10;
        eu_x_cell,Q_set=solve_correl_length_simple(n_values,space(B_set[1],1),CTM_cell,"x",Lx,Ly,partly);
        # eu_allspin_x,allspin_x=solve_correl_length(1/norm_coe, 5, AA_cell,CTM_cell,"x");
        # eu_allspin_y,allspin_y=solve_correl_length(1/norm_coe, 5, AA_cell,CTM_cell,"y");


        for cb in y_range
            double_B_spin_L_set=Matrix{TensorMap}(undef,Lx,1);
            double_T_spin_L_set=Matrix{TensorMap}(undef,Lx,1);
            double_B_spin_R_set=Matrix{TensorMap}(undef,Lx,1);
            double_T_spin_R_set=Matrix{TensorMap}(undef,Lx,1);
            double_B_spin_mid_set=Matrix{TensorMap}(undef,Lx,1);
            double_T_spin_mid_set=Matrix{TensorMap}(undef,Lx,1);
            double_B_CdagC_L_set=Matrix{TensorMap}(undef,Lx,1);
            double_T_CdagC_L_set=Matrix{TensorMap}(undef,Lx,1);
            double_B_CdagC_R_set=Matrix{TensorMap}(undef,Lx,1);
            double_T_CdagC_R_set=Matrix{TensorMap}(undef,Lx,1);
            double_B_CdagC_mid_set=Matrix{TensorMap}(undef,Lx,1);
            double_T_CdagC_mid_set=Matrix{TensorMap}(undef,Lx,1);


            for ca=1:Lx
            
                B_double_spin_L,T_double_spin_L, B_double_spin_mid,T_double_spin_mid, B_double_spin_R,T_double_spin_R=build_AA_spin(S1L,S1R,B_set,T_set,ca,cb);
                double_B_spin_L_set[ca]=B_double_spin_L;
                double_T_spin_L_set[ca]=T_double_spin_L;
                double_B_spin_mid_set[mod1(ca+1,Lx)]=B_double_spin_mid;
                double_T_spin_mid_set[mod1(ca+1,Lx)]=T_double_spin_mid;
                double_B_spin_R_set[mod1(ca+2,Lx)]=B_double_spin_R;
                double_T_spin_R_set[mod1(ca+2,Lx)]=T_double_spin_R;


                # @tensor AA_[:]:=B_double_spin_mid[-1,1,-4]*T_double_spin_mid[-2,-3,1];#(L M U),(D R M) =>(L,D,R,U)

                
                B_double_CdagC_L,T_double_CdagC_L, B_double_CdagC_mid,T_double_CdagC_mid, B_double_CdagC_R,T_double_CdagC_R=build_AA_hop(Cdag,C,B_set,T_set,ca,cb);
                double_B_CdagC_L_set[ca]=B_double_CdagC_L;
                double_T_CdagC_L_set[ca]=T_double_CdagC_L;
                double_B_CdagC_mid_set[mod1(ca+1,Lx)]=B_double_CdagC_mid;
                double_T_CdagC_mid_set[mod1(ca+1,Lx)]=T_double_CdagC_mid;
                double_B_CdagC_R_set[mod1(ca+2,Lx)]=B_double_CdagC_R;
                double_T_CdagC_R_set[mod1(ca+2,Lx)]=T_double_CdagC_R;
            
            end
        
            for ca in x_range
            
            
                #################################
                norms=evaluate_correl([ca,cb],1,"x", B_double_set[:,cb], T_double_set[:,cb], B_double_set[:,cb], T_double_set[:,cb], B_double_set[:,cb], T_double_set[:,cb], CTM_cell, distance);
                # println(norms)
                norm_coe=(norms[4+Lx]/norms[4])^(1/Lx); #get a rough normalization coefficient to avoid that the number becomes two small
                norms=evaluate_correl([ca,cb],1/norm_coe,"x", B_double_set[:,cb], T_double_set[:,cb], B_double_set[:,cb], T_double_set[:,cb], B_double_set[:,cb], T_double_set[:,cb], CTM_cell, distance);
                Spin_ob=evaluate_correl([ca,cb], 1/norm_coe, "x", double_B_spin_mid_set,double_T_spin_mid_set, double_B_spin_L_set,double_T_spin_L_set, double_B_spin_R_set,double_T_spin_R_set, CTM_cell, distance);
                hopping_ob=evaluate_correl([ca,cb], 1/norm_coe, "x", double_B_CdagC_mid_set,double_T_CdagC_mid_set, double_B_CdagC_L_set,double_T_CdagC_L_set, double_B_CdagC_R_set,double_T_CdagC_R_set, CTM_cell, distance);
                

                Spin_ob=Spin_ob./norms;
                hopping_ob=hopping_ob./norms;
                SS_ob_set[ca,cb]=Spin_ob;
                CdagC_ob_set[ca,cb]=hopping_ob;
            

            end
        end
    end

    mat_filenm="correl_D"*string(D)*"_chi"*string(chi);
    if partly
        mat_filenm=mat_filenm*"_part";
    else
        mat_filenm=mat_filenm*"_full";
    end
    matwrite(mat_filenm*".mat", Dict(
        "Lx"=>Lx,
        "Ly"=>Ly,
        "SS_ob_set" => SS_ob_set,
        "CdagC_ob_set" => CdagC_ob_set,
        "eu_x_cell"=>eu_x_cell,
        "Q_set"=>Q_set
        # "eu_allspin_x" => eu_allspin_x,
        # "allspin_x"=> allspin_x
    ); compress = false)

    return SS_ob_set,CdagC_ob_set
end