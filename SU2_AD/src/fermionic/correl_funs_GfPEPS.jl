using KrylovKit
using LinearAlgebra
using TensorKit


function create_H_term(O1,O2,direction,is_odd)
    if direction=="x"
        if is_odd
            #c1dag c2
            sign1=[1,1,1,1,0];
            sign2=[0,0,0,1,0];
            ind1=3;#index p
            ind2=1;#index p
            p1=1;
            p2=1;
        else
            # n1 n2 
            sign1=[0,0,0,0,0];
            sign2=[0,0,0,0,0];
            ind1=3;#index p
            ind2=1;#index p
            p1=0;
            p2=0;
        end

        H_term=Dict([("direction","x"),("O1", O1), ("O2", O2), ("sign1",sign1), ("sign2",sign2), ("ind1",ind1), ("ind2",ind2), ("p1",p1), ("p2",p2)]);
    end
    return H_term

end

function Hamiltonians(M,U_phy1,U_phy2)
    if M==1

        Vdummy=ℂ[U1Irrep](-1=>1);
        V=ℂ[U1Irrep](0=>1,1=>1);

        Id=[1 0;0 1];
        sm=[0 1;0 0]; sp=[0 0;1 0]; sz=[1 0; 0 -1]; occu=[0 0; 0 1];
        
        @tensor Ident[:]:=Id[-1,-3]*Id[-2,-4];
        Ident=TensorMap(Ident,  V ⊗ V ← V ⊗ V);

        @tensor NA[:]:=occu[-1,-3]*Id[-2,-4];
        NA=TensorMap(NA,  V ⊗ V ← V ⊗ V);
        
        @tensor NB[:]:=Id[-1,-3]*occu[-2,-4];
        NB=TensorMap(NB,  V ⊗ V ← V ⊗ V);

        @tensor NANB[:]:=occu[-1,-3]*occu[-2,-4];
        NANB=TensorMap(NANB,  V ⊗ V ← V ⊗ V);

        @tensor cAdag[:]:=sp[-1,-3]*Id[-2,-4];
        CAdag=zeros(1,2,2,2,2);
        CAdag[1,:,:,:,:]=cAdag;
        CAdag=TensorMap(CAdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);

        @tensor cBdag[:]:=sz[-1,-3]*sp[-2,-4];
        CBdag=zeros(1,2,2,2,2);
        CBdag[1,:,:,:,:]=cBdag;
        CBdag=TensorMap(CBdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);

        @tensor cA[:]:=sm[-1,-3]*Id[-2,-4];
        CA=zeros(1,2,2,2,2);
        CA[1,:,:,:,:]=cA;
        CA=TensorMap(CA, Vdummy' ⊗ V ⊗ V ← V ⊗ V);

        @tensor cB[:]:=sz[-1,-3]*sm[-2,-4];
        CB=zeros(1,2,2,2,2);
        CB[1,:,:,:,:]=cB;
        CB=TensorMap(CB, Vdummy' ⊗ V ⊗ V ← V ⊗ V);



        @tensor Ident[:]:=Ident[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Ident[:]:=Ident[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NANB[:]:=NANB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NANB[:]:=NANB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor CAdag[:]:=CAdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CAdag[:]:=CAdag[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CBdag[:]:=CBdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CBdag[:]:=CBdag[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CA[:]:=CA[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CA[:]:=CA[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CB[:]:=CB[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CB[:]:=CB[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];
    elseif M==2
        Vdummy=ℂ[U1Irrep](-1=>1)';
        V=ℂ[U1Irrep](0=>1,1=>1)';
    
        Id=[1 0;0 1];
        sm=[0 1;0 0]; sp=[0 0;1 0]; sz=[1 0; 0 -1]; occu=[0 0; 0 1];
        
        @tensor Ident[:]:=Id[-1,-3]*Id[-2,-4];
        Ident=TensorMap(Ident,  V ⊗ V ← V ⊗ V);
    
        @tensor NA[:]:=occu[-1,-3]*Id[-2,-4];
        NA=TensorMap(NA,  V ⊗ V ← V ⊗ V);
        
        @tensor NB[:]:=Id[-1,-3]*occu[-2,-4];
        NB=TensorMap(NB,  V ⊗ V ← V ⊗ V);
    
        @tensor NANB[:]:=occu[-1,-3]*occu[-2,-4];
        NANB=TensorMap(NANB,  V ⊗ V ← V ⊗ V);
    
        @tensor cAdag[:]:=sp[-1,-3]*Id[-2,-4];
        CAdag=zeros(1,2,2,2,2);
        CAdag[1,:,:,:,:]=cAdag;
        CAdag=TensorMap(CAdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);
    
        @tensor cBdag[:]:=sz[-1,-3]*sp[-2,-4];
        CBdag=zeros(1,2,2,2,2);
        CBdag[1,:,:,:,:]=cBdag;
        CBdag=TensorMap(CBdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);
    
        @tensor cA[:]:=sm[-1,-3]*Id[-2,-4];
        CA=zeros(1,2,2,2,2);
        CA[1,:,:,:,:]=cA;
        CA=TensorMap(CA, Vdummy' ⊗ V ⊗ V ← V ⊗ V);
    
        @tensor cB[:]:=sz[-1,-3]*sm[-2,-4];
        CB=zeros(1,2,2,2,2);
        CB[1,:,:,:,:]=cB;
        CB=TensorMap(CB, Vdummy' ⊗ V ⊗ V ← V ⊗ V);
    
    
    
        @tensor Ident[:]:=Ident[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Ident[:]:=Ident[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    
        @tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    
        @tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    
        @tensor NANB[:]:=NANB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NANB[:]:=NANB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    
        @tensor CAdag[:]:=CAdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CAdag[:]:=CAdag[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    
        @tensor CBdag[:]:=CBdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CBdag[:]:=CBdag[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    
        @tensor CA[:]:=CA[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CA[:]:=CA[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    
        @tensor CB[:]:=CB[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CB[:]:=CB[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    end    


    return Ident, NA, NB, NANB, CAdag, CA, CBdag, CB 
end

function evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    H_term=create_H_term(O1,O2,direction,is_odd);
    
    AA1p,_,AA2p=build_double_layer_swap_op(A_fused,A_fused,A_fused,H_term);

    if direction=="x"
        norm=ob_2sites_x(CTM,AA_fused,AA_fused,false);
        ob=ob_2sites_x(CTM,AA1p,AA2p,is_odd);
    end
    
    return ob/norm
    
end

function ob_2sites_x(CTM,AA1,AA2,is_odd)

    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];


    if is_odd
        @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-2]*AA1[-1,2,5,-3,3]*Tset.T3[-4,5,4];
        @tensor envR[:]:=Tset.T1[-1,3,1]*AA2[-2,5,2,3,-4]*Tset.T3[4,5,-3]*envR[1,2,4];

        rho=@tensor envL[4,1,2,3]*envR[1,2,3,4];
    else
        @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA1[2,5,-2,3]*Tset.T3[-3,5,4];
        @tensor envR[:]:=Tset.T1[-1,3,1]*AA2[-2,5,2,3]*Tset.T3[4,5,-3]*envR[1,2,4];
        rho=@tensor envL[1,2,3]*envR[1,2,3];
    end
    return rho;
end





function evaluate_correl_Cdag_C(direction, AA_fused, AA_op1, AA_op2, CTM, distance,is_odd)
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
        if is_odd
            @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4]*T3[-3,6,7];
            @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
            ov=@tensor va[1,2,3]*vb[1,2,3]
            correl_funs[1]=ov;
            
            for dis=2:distance
                @tensor va[:]:=va[1,3,5]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                ov=@tensor va[1,2,3]*vb[1,2,3]
                correl_funs[dis]=ov;
            end
        else
            if direction=="x"
                @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4]*T3[-3,6,7];
                @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
                ov=@tensor va[1,2,3]*vb[1,2,3]
                correl_funs[1]=ov;
                
                for dis=2:distance
                    @tensor va[:]:=va[1,3,5]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                    ov=@tensor va[1,2,3]*vb[1,2,3]
                    correl_funs[dis]=ov;
                end
                return correl_funs
            end
        end
        return correl_funs
    end

end


function correl_TransOp(vl,Tup,Tdown,AAfused)
    if AAfused==[]
        
        @tensor vl[:]:=vl[-1,1,3]*Tup[1,2,-2]*Tdown[-3,2,3];
        
    else
        
        @tensor vl[:]:=vl[-1,1,3,5]*Tup[1,2,-2]*AAfused[3,4,-3,2]*Tdown[-4,4,5];
        
    end
    return vl
end
function solve_correl_length(n_values,AA_fused,CTM,direction)
    T1=CTM.Tset.T1;
    T2=CTM.Tset.T2;
    T3=CTM.Tset.T3;
    T4=CTM.Tset.T4;
    println(fuse(space(T1,1)'⊗space(AA_fused,1)', space(T3,3)))
    if direction=="x"
        correl_TransOp_fx(x)=correl_TransOp(x,T1,T3,AA_fused)

        Vl=Rep[U₁](0=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
        eus=eu;
        QN=eu*0;

        Vl=Rep[U₁](1=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.+1);
        end



        Vl=Rep[U₁](2=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.+2);
        end

        Vl=Rep[U₁](-1=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.-1);
        end



        Vl=Rep[U₁](-2=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
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


function cal_correl(M,A_fused, AA_fused,U_phy1,U_phy2, chi,CTM, distance)
    #M: number of virtual modes 
    
    Ident, NA, NB, NANB, CAdag, CA, CBdag, CB=Hamiltonians(M,U_phy1,U_phy2)

    O1=NA;
    O2=Ident;
    direction="x";
    is_odd=false;
    NA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    O1=NB;
    O2=Ident;
    direction="x";
    is_odd=false;
    NB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    O1=NANB;
    O2=Ident;
    direction="x";
    is_odd=false;
    NANB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    
    @tensor O1[:]:=CAdag[1,-1,2]*CB[1,2,-2];
    O2=Ident;
    direction="x";
    is_odd=false;
    CAdagCB_onsite=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    
    
    
    println("NA=   "*string(NA))
    println("NB=   "*string(NB))
    println("NANB=   "*string(NANB))
    println("CAdagCB_onsite=   "*string(CAdagCB_onsite))
    
    
    
    
    
    O1=CAdag;
    O2=CA;
    direction="x";
    is_odd=true;
    H_term=create_H_term(O1,O2,direction,is_odd);
    AA_CAdag,AA_mid,AA_CA=build_double_layer_swap_op(A_fused,A_fused,A_fused,H_term);
    
    # gate=parity_gate(AA_CAdag,4);
    # @tensor AA_CAdag[:]:=AA_CAdag[-1,-2,-3,1,-5]*gate[-4,1];
    # gate=parity_gate(AA_CA,1);
    # @tensor AA_CA[:]:=AA_CA[1,-2,-3,-4,-5]*gate[-1,1];
    
    
    O1=CBdag;
    O2=CB;
    direction="x";
    is_odd=true;
    H_term=create_H_term(O1,O2,direction,is_odd);
    AA_CBdag,AA_mid,AA_CB=build_double_layer_swap_op(A_fused,A_fused,A_fused,H_term);
    



    
    norms=evaluate_correl_Cdag_C("x", AA_fused, AA_fused, AA_fused, CTM, 10, false);
    norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
    norms=evaluate_correl_Cdag_C("x", AA_fused/norm_coe, AA_fused, AA_fused, CTM, distance, false);
    

    CAdag_CA_ob=evaluate_correl_Cdag_C("x", AA_mid/norm_coe, AA_CAdag, AA_CA, CTM, distance, true);
    CAdag_CB_ob=evaluate_correl_Cdag_C("x", AA_mid/norm_coe, AA_CAdag, AA_CB, CTM, distance, true);
    CBdag_CA_ob=evaluate_correl_Cdag_C("x", AA_mid/norm_coe, AA_CBdag, AA_CA, CTM, distance, true);
    CBdag_CB_ob=evaluate_correl_Cdag_C("x", AA_mid/norm_coe, AA_CBdag, AA_CB, CTM, distance, true);

    CAdag_CA_ob=CAdag_CA_ob./norms;
    CAdag_CB_ob=CAdag_CB_ob./norms;
    CBdag_CA_ob=CBdag_CA_ob./norms;
    CBdag_CB_ob=CBdag_CB_ob./norms;

    println(norms)

    eus_x,  QN_x=solve_correl_length(5,AA_fused/norm_coe,CTM,"x");


    _,corner_spec=svd(convert(Array,CTM.Cset.C1))



    CAdag_CA_ob=[NA;CAdag_CA_ob];
    CBdag_CB_ob=[NB;CBdag_CB_ob];
    CAdag_CB_ob=[CAdagCB_onsite;CAdag_CB_ob];
    CBdag_CA_ob=[CAdagCB_onsite';CBdag_CA_ob];

    mat_filenm="correl_M"*string(M)*"_chi"*string(chi)*".mat";
    matwrite(mat_filenm, Dict(
        "corner_spec" => corner_spec,
        "CAdag_CA_ob" => CAdag_CA_ob,
        "CAdag_CB_ob" => CAdag_CB_ob,
        "CBdag_CA_ob" => CBdag_CA_ob,
        "CBdag_CB_ob" => CBdag_CB_ob,
        "eus_x" => eus_x,
        "QN_x"=> QN_x,
        "CTM_space"=> string(space(CTM.Cset.C1))
    ); compress = false)
    return CAdag_CA_ob,CAdag_CB_ob,CBdag_CA_ob,CBdag_CB_ob
end


function cal_correl_FP_edge(M, AA_fused,AA_SS,AA_SAL,AA_SBL,AA_SAR,AA_SBR, chi,CTM, distance)
    #M: number of virtual modes 
    


    #single-unitcell correlations
    norm=ob_1site_closed(CTM,AA_fused);
    
    SS_cell_ob=ob_1site_closed(CTM,AA_SS);
    SS_cell_ob=SS_cell_ob/norm;

    
    norms=evaluate_correl_spinspin_FP_edge("x", AA_fused, AA_fused, AA_fused, CTM, "dimerdimer", 10);
    norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
    norms=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_fused, AA_fused, CTM, "dimerdimer", distance);
    dimer_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SS, AA_SS, CTM, "dimerdimer", distance);

    SASA_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SAL, AA_SAR, CTM, "spinspin", distance);
    SASB_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SAL, AA_SBR, CTM, "spinspin", distance);
    SBSA_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SBL, AA_SAR, CTM, "spinspin", distance);
    SBSB_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SBL, AA_SBR, CTM, "spinspin", distance);

    dimer_ob=dimer_ob./norms;
    SASA_ob=SASA_ob./norms;
    SASB_ob=SASB_ob./norms;
    SBSA_ob=SBSA_ob./norms;
    SBSB_ob=SBSB_ob./norms;

    println(norms)

    eus_x, Qspin_x, QN_x=solve_correl_length(5,AA_fused/norm_coe,CTM,"x");


    _,corner_spec=svd(convert(Array,CTM.Cset.C1))

    mat_filenm="correl_FP_edge_M"*string(M)*"_chi"*string(chi)*".mat";
    matwrite(mat_filenm, Dict(
        "corner_spec" => corner_spec,
        "SS_cell_ob" => SS_cell_ob,
        "dimer_ob" => dimer_ob,
        "SASA_ob" => SASA_ob,
        "SASB_ob" => SASB_ob,
        "SBSA_ob" => SBSA_ob,
        "SBSB_ob" => SBSB_ob,
        "eus_x" => eus_x,
        "Qspin_x"=> Qspin_x,
        "QN_x"=> QN_x,
        "CTM_space"=> string(space(CTM["Cset"][1]))
    ); compress = false)
end

function ob_1site_closed(CTM,AA_fused)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA_fused[2,5,-2,3]*Tset.T3[-3,5,4];
    Norm=@tensor envL[1,2,3]*envR[1,2,3];
    return Norm;
end