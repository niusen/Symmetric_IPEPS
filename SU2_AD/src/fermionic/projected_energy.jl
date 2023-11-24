



function ob_RU_LD(CTM,AA_fused,AA_RU,AA_LD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 


    @tensor up[:]:=MM_LU[-1,-2,1,2]*MM_RU[1,2,-3,-4,-5];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4];
    @tensor ob[:]:=up[1,2,3,4,5]*down[1,2,3,4,5];
    ob=blocks(ob)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
    return ob
end




function evaluate_correl_spinspin(direction, AA_fused, AA_op1, AA_op2, CTM, method, distance)
    correl_funs=Vector(undef,distance);

    C1=CTM.Cset.C1;
    C2=CTM.Cset.C2;
    C3=CTM.Cset.C3;
    C4=CTM.Cset.C4;
    T1=CTM.Tset.T1;
    T2=CTM.Tset.T2;
    T3=CTM.Tset.T3;
    T4=CTM.Tset.T4;
    if method=="dimerdimer"#operator on a single site conserves su2 symmetry
        if direction=="x"
            @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4]*T3[-3,6,7];
            @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
            @tensor ov[:]:=va[1,2,3]*vb[1,2,3]
            correl_funs[1]=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            
            for dis=2:distance
                @tensor va[:]:=va[1,3,5]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                @tensor ov[:]:=va[1,2,3]*vb[1,2,3]
                correl_funs[dis]=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            end
            return correl_funs
        end
    elseif method=="spinspin" #operator on a single site breaks su2 symmetry, so there is an extra index obtained from svd of two-site operator
        if direction=="x"
            @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4,-4]*T3[-3,6,7];
            @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4,-4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
            @tensor ov[:]:=va[1,2,3,4]*vb[1,2,3,4]
            correl_funs[1]=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            
            for dis=2:distance
                @tensor va[:]:=va[1,3,5,-4]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                @tensor ov[:]:=va[1,2,3,4]*vb[1,2,3,4]
                correl_funs[dis]=blocks(ov)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
            end
            return correl_funs
        end
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



function cal_correl(M, AA_fused,AA_SS,AA_SAL,AA_SBL,AA_SAR,AA_SBR, chi,CTM, distance)
    #M: number of virtual modes 
    


    #single-unitcell correlations
    norm=ob_1site_closed(CTM,AA_fused);
    
    SS_cell_ob=ob_1site_closed(CTM,AA_SS);
    SS_cell_ob=SS_cell_ob/norm;

    
    norms=evaluate_correl_spinspin("x", AA_fused, AA_fused, AA_fused, CTM, "dimerdimer", 10);
    norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
    norms=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_fused, AA_fused, CTM, "dimerdimer", distance);
    dimer_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SS, AA_SS, CTM, "dimerdimer", distance);

    SASA_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SAL, AA_SAR, CTM, "spinspin", distance);
    SASB_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SAL, AA_SBR, CTM, "spinspin", distance);

    dimer_ob=dimer_ob./norms;
    SASA_ob=SASA_ob./norms;
    SASB_ob=SASB_ob./norms;


    eus_x, Qspin_x, QN_x=solve_correl_length(5,AA_fused/norm_coe,CTM,"x");




    mat_filenm="correl_M"*string(M)*"_chi"*string(chi)*".mat";
    matwrite(mat_filenm, Dict(
        "SS_cell_ob" => SS_cell_ob,
        "dimer_ob" => dimer_ob,
        "SASA_ob" => SASA_ob,
        "SASB_ob" => SASB_ob,
        "eus_x" => eus_x,
        "Qspin_x"=> Qspin_x,
        "QN_x"=> QN_x,
        "CTM_space"=> string(space(CTM.Cset.C1))
    ); compress = false)
end

function ob_1site_closed(CTM,AA_fused)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA_fused[2,5,-2,3]*Tset.T3[-3,5,4];
    ob=@tensor envL[1,2,3]*envR[1,2,3];
    return ob;
end

function norm_2sites_x(CTM,AA_fused)

    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA_fused[2,5,-2,3]*Tset.T3[-3,5,4];
    @tensor envR[:]:=Tset.T1[-1,3,1]*AA_fused[-2,5,2,3]*Tset.T3[4,5,-3]*envR[1,2,4];
    ob=@tensor envL[1,2,3]*envR[1,2,3];
    return ob
end

function ob_2sites_x(CTM,AA1,AA2)

    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA1[2,5,-2,3,-4]*Tset.T3[-3,5,4];
    @tensor envR[:]:=Tset.T1[-1,3,1]*AA2[-2,5,2,3,-4]*Tset.T3[4,5,-3]*envR[1,2,4];
    ob=@tensor envL[1,2,3,4]*envR[1,2,3,4];
    return ob
end

function norm_2sites_y(CTM,AA_fused)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envU[:]:=Cset.C2[1,-1]*Tset.T1[2,-2,1]*Cset.C1[-3,2];
    @tensor envD[:]:=Cset.C3[-1,1]*Tset.T3[1,-2,2]*Cset.C4[2,-3];
    @tensor envU[:]:=envU[1,2,4]*Tset.T2[1,3,-1]*AA_fused[5,-2,3,2]*Tset.T4[-3,5,4];
    @tensor envD[:]:=Tset.T2[-1,3,1]*AA_fused[5,2,3,-2]*Tset.T4[4,5,-3]*envD[1,2,4];
    ob=@tensor envU[1,2,3]*envD[1,2,3];
    return ob
end

function ob_2sites_y(CTM,AA1,AA2)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envU[:]:=Cset.C2[1,-1]*Tset.T1[2,-2,1]*Cset.C1[-3,2];
    @tensor envD[:]:=Cset.C3[-1,1]*Tset.T3[1,-2,2]*Cset.C4[2,-3];
    @tensor envU[:]:=envU[1,2,4]*Tset.T2[1,3,-1]*AA1[5,-2,3,2,-4]*Tset.T4[-3,5,4];
    @tensor envD[:]:=Tset.T2[-1,3,1]*AA2[5,2,3,-2,-4]*Tset.T4[4,5,-3]*envD[1,2,4];
    ob=@tensor envU[1,2,3,4]*envD[1,2,3,4];
    return ob
end

function norm_2x2(CTM,AA_fused)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_fused[3,4,-4,-2]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    ob=@tensor up[1,2,3,4]*down[1,2,3,4];
    return ob
end

function ob_LD_LU_RU(CTM,AA_fused,AA_LD,AA_LU,AA_RU,U_chiral)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_LU[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 


    @tensor up[:]:=U_chiral'[-5,4,1]*MM_LU[-1,-2,2,3,1]*MM_RU[2,3,-3,-4,4];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end


function ob_LU_RU_RD(CTM,AA_fused,AA_LU,AA_RU,AA_RD,U_chiral)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_LU[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_fused[3,4,-4,-2]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 


    @tensor up[:]:=MM_LU[-1,-2,2,3,4]*MM_RU[2,3,-3,-4,1]*U_chiral'[4,-5,1];
    @tensor down[:]:=MM_LD[-1,-2,1,2]*MM_RD[1,2,-3,-4,-5];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end




function ob_RU_RD_LD(CTM,AA_fused,AA_RU,AA_RD,AA_LD,U_chiral)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 


    @tensor up[:]:=MM_LU[-1,-2,1,2]*MM_RU[1,2,-3,-4,-5];
    @tensor down[:]:=MM_LD[-1,-2,2,3,4]*MM_RD[2,3,-3,-4,1]*U_chiral'[-5,4,1];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end

function ob_RD_LD_LU(CTM,AA_fused,AA_RD,AA_LD,AA_LU,U_chiral)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_LU[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5];


    @tensor up[:]:=MM_LU[-1,-2,1,2,-5]*MM_RU[1,2,-3,-4];
    @tensor down[:]:=U_chiral'[4,-5,1]*MM_LD[-1,-2,2,3,1]*MM_RD[2,3,-3,-4,4];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end


function ob_LU_RD(CTM,AA_fused,AA_LU,AA_RD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_LU[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_fused[3,4,-4,-2]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5];


    @tensor up[:]:=MM_LU[-1,-2,1,2,-5]*MM_RU[1,2,-3,-4];
    @tensor down[:]:=MM_LD[-1,-2,1,2]*MM_RD[1,2,-3,-4,-5];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end


function ob_RU_LD(CTM,AA_fused,AA_RU,AA_LD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 


    @tensor up[:]:=MM_LU[-1,-2,1,2]*MM_RU[1,2,-3,-4,-5];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end
