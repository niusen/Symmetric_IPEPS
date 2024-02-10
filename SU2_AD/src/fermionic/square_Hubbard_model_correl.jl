



function build_double_layer_swap_op(A1,A_mid,A2,op1,op2,direction) #for x direction hopping 
    A1=deepcopy(A1)
    A2=deepcopy(A2)
    A_mid=deepcopy(A_mid);
    A1_origin=deepcopy(A1)
    A2_origin=deepcopy(A2)
    A_mid_origin=deepcopy(A_mid);

    if (Rank(op1)==3)&(Rank(op2)==3)
        if direction=="x"
            
            #the first index of O is dummy
            @tensor A1[:]:= A1[-1,-2,-3,-4,1]*op1[-6,-5,1]
            @tensor A2[:]:= A2[-1,-2,-3,-4,1]*op2[-6,-5,1]
            O_string=unitary(space(op1,1),space(op1,1));
            
            gate=parity_gate(A1,1); @tensor A1[:]:=A1[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=parity_gate(A1,2); @tensor A1[:]:=A1[-1,1,-3,-4,-5,-6]*gate[-2,1];
            #gate=parity_gate(A1,3); @tensor A1[:]:=A1[-1,-2,1,-4,-5,-6]*gate[-3,1];
            gate=parity_gate(A1,4); @tensor A1[:]:=A1[-1,-2,-3,1,-5,-6]*gate[-4,1];

            gate=parity_gate(A_mid,2); @tensor A_mid[:]:=A_mid[-1,1,-3,-4,-5]*gate[-2,1];

            gate=parity_gate(A2,1); @tensor A2[:]:=A2[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=parity_gate(A2,4); @tensor A2[:]:=A2[-1,-2,-3,1,-5,-6]*gate[-4,1];
            
            U=unitary(fuse(space(A1,3)⊗space(A1,6)), space(A1,3)⊗space(A1,6)); 
            @tensor A1_new[:]:=A1[-1,-2,1,-4,-5,2]*U[-3,1,2];
            @tensor A_mid_new[:]:=A_mid[1,-2,3,-4,-5]*O_string[4,2]*U'[1,2,-1]*U[-3,3,4];
            @tensor A2_new[:]:=A2[1,-2,-3,-4,-5,2]*U'[1,2,-1];

            A1_double,_,_,_,_=build_double_layer_swap(A1_origin',A1_new)
            A_mid_double,_,_,_,_=build_double_layer_swap(A_mid_origin',A_mid_new)
            A2_double,_,_,_,_=build_double_layer_swap(A2_origin',A2_new)

        elseif direction=="y"
            #the first index of O is dummy
            @tensor A1[:]:= A1[-1,-2,-3,-4,1]*op1[-6,-5,1]
            @tensor A2[:]:= A2[-1,-2,-3,-4,1]*op2[-6,-5,1]
            O_string=unitary(space(op1,1),space(op1,1));
            
            gate=parity_gate(A1,1); @tensor A1[:]:=A1[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=parity_gate(A1,4); @tensor A1[:]:=A1[-1,-2,-3,1,-5,-6]*gate[-4,1];

            gate=parity_gate(A_mid,1); @tensor A_mid[:]:=A_mid[1,-2,-3,-4,-5]*gate[-1,1];

            gate=parity_gate(A2,4); @tensor A2[:]:=A2[-1,-2,-3,1,-5,-6]*gate[-4,1];
            
            U=unitary(fuse(space(A1,3)⊗space(A1,6)), space(A1,3)⊗space(A1,6)); 
            @tensor A1_new[:]:=A1[-1,1,-3,-4,-5,2]*U[-2,1,2];
            @tensor A_mid_new[:]:=A_mid[-1,3,-3,1,-5]*O_string[4,2]*U'[1,2,-4]*U[-2,3,4];
            @tensor A2_new[:]:=A2[-1,-2,-3,1,-5,2]*U'[1,2,-4];

            A1_double,_,_,_,_=build_double_layer_swap(A1_origin',A1_new)
            A_mid_double,_,_,_,_=build_double_layer_swap(A_mid_origin',A_mid_new)
            A2_double,_,_,_,_=build_double_layer_swap(A2_origin',A2_new)
        end
        return A1_double,A_mid_double,A2_double

    elseif (Rank(op1)==2)&(Rank(op2)==2) # No extra leg
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*op1[-5,1]
        @tensor A2[:]:= A2[-1,-2,-3,-4,1]*op2[-5,1]
        A1_new=A1
        A2_new=A2

        A1_double,_,_,_,_=build_double_layer_swap(A1_origin',A1_new)
        A2_double,_,_,_,_=build_double_layer_swap(A2_origin',A2_new)

        return A1_double,nothing,A2_double
    end

end



function evaluate_correl_Cdag_C(direction, AA_mid, AA_op1, AA_op2, CTM, distance,is_odd)
    correl_funs=Vector(undef,distance);

    C1=CTM.Cset.C1;
    C2=CTM.Cset.C2;
    C3=CTM.Cset.C3;
    C4=CTM.Cset.C4;
    T1=CTM.Tset.T1;
    T2=CTM.Tset.T2;
    T3=CTM.Tset.T3;
    T4=CTM.Tset.T4;


    if direction=="x"
        @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4]*T3[-3,6,7];
        @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
        ov=@tensor va[1,2,3]*vb[1,2,3]
        correl_funs[1]=ov;
        
        for dis=2:distance
            @tensor va[:]:=va[1,3,5]*T1[1,2,-1]*AA_mid[3,4,-2,2]*T3[-3,4,5];
            ov=@tensor va[1,2,3]*vb[1,2,3]
            correl_funs[dis]=ov;
        end
        return correl_funs
    elseif direction=="y"
        @tensor va[:]:=C1[3,1]*T1[1,4,2]*C2[2,6]*T4[-1,5,3]*AA_op1[5,-2,7,4]*T2[6,7,-3];
        @tensor vb[:]:=T4[7,6,-1]*AA_op2[6,5,4,-2]*T2[-3,4,3]*C4[1,7]*T3[2,5,1]*C3[3,2];
        ov=@tensor va[1,2,3]*vb[1,2,3]
        correl_funs[1]=ov;
        
        for dis=2:distance
            @tensor va[:]:=va[1,3,5]*T4[-1,2,1]*AA_mid[2,-2,4,3]*T2[5,4,-3];
            ov=@tensor va[1,2,3]*vb[1,2,3]
            correl_funs[dis]=ov;
        end

        return correl_funs
    end

end

function cal_correl(direction,M,A, AA, chi,CTM, distance,ctm_setting)
    #M: number of virtual modes 
    if isa(space(A,1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}) #U1 symmetry
        Ident_op, NA_op, NB_op, NANB_op, CdagA_CB_op, Cdag_A_op, C_A_op, Cdag_B_op, C_B_op =Hamiltonians_spinless_U1_2site(M);
    elseif isa(space(A,1), GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}) #U1*SU2 symmetry
        Ident_op, NA_op, NB_op, n_double_A_op, CdagA_CB_op, Cdag_A_op, C_A_op, Cdag_B_op, C_B_op =Hamiltonians_spinless_U1_SU2_2site(M);
    end

    # if direction=="x"
        NA=ob_onsite(CTM,NA_op,A,AA,ctm_setting);
        NB=ob_onsite(CTM,NB_op,A,AA,ctm_setting);
        CAdagCB_onsite=ob_onsite(CTM,CdagA_CB_op,A,AA,ctm_setting);
        
        println("NA=   "*string(NA))
        println("NB=   "*string(NB))
        println("CAdagCB_onsite=   "*string(CAdagCB_onsite))
        
        
        O1=Cdag_A_op;
        O2=C_A_op;
        AA_CdagA,AA_mid,AA_CA=build_double_layer_swap_op(A,A,A,O1,O2,direction);
        
        
        O1=Cdag_B_op;
        O2=C_B_op;
        AA_CdagB,AA_mid,AA_CB=build_double_layer_swap_op(A,A,A,O1,O2,direction);
        



        
        norms=evaluate_correl_Cdag_C(direction, AA, AA, AA, CTM, 10, false);
        norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
        norms=evaluate_correl_Cdag_C(direction, AA/norm_coe, AA, AA, CTM, distance, false);
        

        CAdag_CA_ob=evaluate_correl_Cdag_C(direction, AA_mid/norm_coe, AA_CdagA, AA_CA, CTM, distance, true);
        CAdag_CB_ob=evaluate_correl_Cdag_C(direction, AA_mid/norm_coe, AA_CdagA, AA_CB, CTM, distance, true);
        CBdag_CA_ob=evaluate_correl_Cdag_C(direction, AA_mid/norm_coe, AA_CdagB, AA_CA, CTM, distance, true);
        CBdag_CB_ob=evaluate_correl_Cdag_C(direction, AA_mid/norm_coe, AA_CdagB, AA_CB, CTM, distance, true);

        CAdag_CA_ob=CAdag_CA_ob./norms;
        CAdag_CB_ob=CAdag_CB_ob./norms;
        CBdag_CA_ob=CBdag_CA_ob./norms;
        CBdag_CB_ob=CBdag_CB_ob./norms;

        println(norms)

        # eus_x,  QN_x=solve_correl_length(5,AA_mid/norm_coe,CTM,direction);
        _,corner_spec=svd(convert(Array,CTM.Cset.C1))



        CAdag_CA_ob=[NA;CAdag_CA_ob];
        CBdag_CB_ob=[NB;CBdag_CB_ob];
        CAdag_CB_ob=[CAdagCB_onsite;CAdag_CB_ob];
        CBdag_CA_ob=[CAdagCB_onsite';CBdag_CA_ob];

    # elseif direction=="y"
    # end

    mat_filenm="correl"*string(direction)*"_M"*string(M)*"_chi"*string(chi)*".mat";
    matwrite(mat_filenm, Dict(
        "corner_spec" => corner_spec,
        "CAdag_CA_ob" => CAdag_CA_ob,
        "CAdag_CB_ob" => CAdag_CB_ob,
        "CBdag_CA_ob" => CBdag_CA_ob,
        "CBdag_CB_ob" => CBdag_CB_ob,
        # "eus_x" => eus_x,
        # "QN_x"=> QN_x,
        "CTM_space"=> string(space(CTM.Cset.C1))
    ); compress = false)
    return CAdag_CA_ob,CAdag_CB_ob,CBdag_CA_ob,CBdag_CB_ob
end


function solve_correl_length(n_values,AA_mid,CTM,direction)
    T1=CTM.Tset.T1;
    T2=CTM.Tset.T2;
    T3=CTM.Tset.T3;
    T4=CTM.Tset.T4;
    println(fuse(space(T1,1)'⊗space(AA_mid,1)', space(T3,3)))

    function correl_TransOp(vl,Tup,Tdown,AAmid)
        if AAmid==[]
            @tensor vl[:]:=vl[-1,1,3]*Tup[1,2,-2]*Tdown[-3,2,3];
        else
            @tensor vl[:]:=vl[-1,1,3,5]*Tup[1,2,-2]*AAmid[3,4,-3,2]*Tdown[-4,4,5];
        end
        return vl
    end

    if direction=="x"
        correl_TransOp_fx(x)=correl_TransOp(x,T1,T3,AA_mid)

        Vl=Rep[U₁](0=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_mid,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
        eus=eu;
        QN=eu*0;

        Vl=Rep[U₁](1=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_mid,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.+1);
        end



        Vl=Rep[U₁](2=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_mid,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.+2);
        end

        Vl=Rep[U₁](-1=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_mid,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.-1);
        end



        Vl=Rep[U₁](-2=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_mid,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.-2);
        end




        eus_abs=abs.(eus);
        @assert maximum(eus_abs)==eus_abs[1]

        eus_abs_sorted=sort(eus_abs,rev=true);
        eus_abs_sorted=eus_abs_sorted/eus_abs_sorted[1];
        QN=QN[sortperm(eus_abs,rev=true)];

        
        return eus_abs_sorted, QN
    end
  
end














