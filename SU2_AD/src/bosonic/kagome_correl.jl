using Zygote:@ignore_derivatives

function build_double_layer_extra_leg(A,operator)
    #su2 operator has three legs, such as svd decomposition of Heisenberg interaction 
    #first two indices of operator are physical indices
    A=permute(A,(1,2,),(3,4,5));
    U_L=@ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    U_D=@ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    U_R=(U_L)';
    U_U=(U_D)';
    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uM,sM,vM=tsvd(A);
    uM=uM*sM

    uM=permute(uM,(1,2,3,),())
    V=space(vM,1);
    U=@ignore_derivatives unitary(fuse(V' ⊗ V), V' ⊗ V);
    @tensor double_LD[:]:=uM'[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];
    
    vM=permute(vM,(1,2,3,4,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    @tensor double_RU[:]:=vM'[3,-2,-4,1]*operator[2,1,-6]*double_RU[-1,3,-3,-5,2];
 
    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))
    double_RU=permute(double_RU,(1,4,5,6,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,5,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,4,3,));
    AA_fused=double_LD*double_RU;
    AA_fused=permute(AA_fused,(1,2,3,4,5,),());
    
    return AA_fused, U_L,U_D,U_R,U_U
end

function single_spin_operator(U_phy,posit1,posit2)

    # Heisenberg interaction
    Id=TensorMap(Matrix(I,2,2),space(U_phy,4),space(U_phy,4));
    sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
    @tensor HSS[:]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4];
    HSS=TensorMap(HSS, space(U_phy,4)⊗space(U_phy,4) ← space(U_phy,4)⊗space(U_phy,4));
    HSS=permute(HSS,(1,3,),(2,4,));
    u,s,v=tsvd(HSS);
    H1=u*s;
    H2=permute(v,(2,3,),(1,));

    if posit1==1
        @tensor H1[:]:= U_phy'[1,2,3,-1]*H1[1,4,-3]*Id[2,5]*Id[3,6]*U_phy[-2,4,5,6];
    elseif posit1==2
        @tensor H1[:]:= U_phy'[1,2,3,-1]*Id[1,4]*H1[2,5,-3]*Id[3,6]*U_phy[-2,4,5,6];
    elseif posit1==3
        @tensor H1[:]:= U_phy'[1,2,3,-1]*Id[1,4]*Id[2,5]*H1[3,6,-3]*U_phy[-2,4,5,6];
    end

    if posit2==1
        @tensor H2[:]:= U_phy'[1,2,3,-1]*H2[1,4,-3]*Id[2,5]*Id[3,6]*U_phy[-2,4,5,6];
    elseif posit2==2
        @tensor H2[:]:= U_phy'[1,2,3,-1]*Id[1,4]*H2[2,5,-3]*Id[3,6]*U_phy[-2,4,5,6];
    elseif posit2==3
        @tensor H2[:]:= U_phy'[1,2,3,-1]*Id[1,4]*Id[2,5]*H2[3,6,-3]*U_phy[-2,4,5,6];
    end

    return H1,H2 
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
            correl_funs[1]=blocks(ov)[Irrep[SU₂](0)][1];
            
            for dis=2:distance
                @tensor va[:]:=va[1,3,5]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                @tensor ov[:]:=va[1,2,3]*vb[1,2,3]
                correl_funs[dis]=blocks(ov)[Irrep[SU₂](0)][1];
            end
            return correl_funs
        end
    elseif method=="spinspin" #operator on a single site breaks su2 symmetry, so there is an extra index obtained from svd of two-site operator
        if direction=="x"
            @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4,-4]*T3[-3,6,7];
            @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4,-4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
            @tensor ov[:]:=va[1,2,3,4]*vb[1,2,3,4]
            correl_funs[1]=blocks(ov)[Irrep[SU₂](0)][1];
            
            for dis=2:distance
                @tensor va[:]:=va[1,3,5,-4]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                @tensor ov[:]:=va[1,2,3,4]*vb[1,2,3,4]
                correl_funs[dis]=blocks(ov)[Irrep[SU₂](0)][1];
            end
            return correl_funs
        end
    end
end


function correl_TransOp(vl,Tup,Tdown,AAfused,construct_double_layer)
    if AAfused==[]
        if construct_double_layer
            @tensor vl[:]:=vl[-1,1,3]*Tup[1,2,-2]*Tdown[-3,2,3];
        else
            @tensor vl[:]:=vl[-1,1,4]*Tup[1,2,3,-2]*Tdown[-3,2,3,4];
        end

        
    else
        if construct_double_layer
            @tensor vl[:]:=vl[-1,1,3,5]*Tup[1,2,-2]*AAfused[3,4,-3,2]*Tdown[-4,4,5];
        else
        end
        
    end
    return vl
end
function solve_correl_length(n_values,AA_fused,CTM,direction,ctm_setting)
    construct_double_layer=ctm_setting.construct_double_layer;

    T1=CTM.Tset.T1;
    T2=CTM.Tset.T2;
    T3=CTM.Tset.T3;
    T4=CTM.Tset.T4;
    if direction=="x"
        correl_TransOp_fx(x)=correl_TransOp(x,T1,T3,AA_fused,construct_double_layer);
        eu_allspin=[];
        allspin=[];
        spins=[0,1/2,1,3/2,2];
        for spin in spins
            if AA_fused==[]
                if construct_double_layer
                    vl_init = permute(TensorMap(randn, SU₂Space(spin=>1)⊗space(T1,1)', space(T3,3)), (1,2,3,),());
                else
                    vl_init = permute(TensorMap(randn, SU₂Space(spin=>1)⊗space(T1,1)', space(T3,4)), (1,2,3,),());
                end
            else
                if construct_double_layer
                    vl_init = permute(TensorMap(randn, SU₂Space(spin=>1)⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());
                else
                    vl_init = permute(TensorMap(randn, SU₂Space(spin=>1)⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,4)), (1,2,3,4,),());
                end
            end
            if norm(vl_init)>0
                eu,ev=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                eu_allspin=vcat(eu_allspin,eu);
                allspin=vcat(allspin,(eu*0).+spin);
            end
        end

        eu_allspin_abs=abs.(eu_allspin);
        @assert maximum(eu_allspin_abs)==eu_allspin_abs[1]

        eu_allspin_abs_sorted=sort(eu_allspin_abs,rev=true);
        eu_allspin_abs_sorted=eu_allspin_abs_sorted/eu_allspin_abs_sorted[1];
        allspin=allspin[sortperm(eu_allspin_abs,rev=true)]

        
        return eu_allspin_abs_sorted,allspin
    elseif direction=="y"
        AA_fused_rotate=[];
        if construct_double_layer
            if AA_fused==[]
                AA_fused_rotate=[];
            else
                AA_fused_rotate=permute(AA_fused,(4,1,2,3),());
            end
        end
        correl_TransOp_fy(x)=correl_TransOp(x,T2,T4,AA_fused_rotate,construct_double_layer)
        eu_allspin=[];
        allspin=[];
        spins=[0,1/2,1,3/2,2];
        for spin in spins
            if AA_fused==[]
                if construct_double_layer
                    vl_init = permute(TensorMap(randn, SU₂Space(spin=>1)⊗space(T2,1)', space(T4,3)), (1,2,3,),());
                else
                    vl_init = permute(TensorMap(randn, SU₂Space(spin=>1)⊗space(T2,1)', space(T4,4)), (1,2,3,),());
                end
            else
                if construct_double_layer
                    vl_init = permute(TensorMap(randn, SU₂Space(spin=>1)⊗space(T2,1)'⊗space(AA_fused_rotate,1)', space(T4,3)), (1,2,3,4,),());
                else
                    vl_init = permute(TensorMap(randn, SU₂Space(spin=>1)⊗space(T2,1)'⊗space(AA_fused_rotate,1)', space(T4,4)), (1,2,3,4,),());
                end
            end
            if norm(vl_init)>0
                eu,ev=eigsolve(correl_TransOp_fy, vl_init, n_values,:LM,Arnoldi());
                eu_allspin=vcat(eu_allspin,eu);
                allspin=vcat(allspin,(eu*0).+spin);
            end
        end
        
        eu_allspin_abs=abs.(eu_allspin);
        @assert maximum(eu_allspin_abs)==eu_allspin_abs[1]
        
        eu_allspin_abs_sorted=sort(eu_allspin_abs,rev=true);
        eu_allspin_abs_sorted=eu_allspin_abs_sorted/eu_allspin_abs_sorted[1];
        allspin=allspin[sortperm(eu_allspin_abs,rev=true)]
        
        return eu_allspin_abs_sorted,allspin
    end
end

function cal_correl(iPESS_tensors, A_unfused, A_fused, AA_fused,U_phy,CTM,D,chi,parameters,distance)
    global ctm_setting,backward_settings,energy_setting
    global U_L,U_D,U_R,U_U


    println("spcae of C1: "*string(space(CTM.Cset.C1)))
    println("spcae of C2: "*string(space(CTM.Cset.C2)))
    println("spcae of C3: "*string(space(CTM.Cset.C3)))
    println("spcae of C4: "*string(space(CTM.Cset.C4)))
    flush(stdout);




    @time E_up, E_down=evaluate_ob(parameters, U_phy, iPESS_tensors, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method, energy_setting.E_up_method, energy_setting.E_dn_method)
    @time E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23=evaluate_ob(parameters, U_phy, iPESS_tensors, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, "E_NN_bond", nothing,nothing);
    println((E_up+E_down)/3);flush(stdout);



    #(direction,parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM)
    _, _, SS12, SS31, SS23=Hamiltonians(U_phy,1,0,0,0,0);
    S1L,S1R=single_spin_operator(U_phy,1,1);
    S2L,S2R=single_spin_operator(U_phy,2,2);
    S3L,S3R=single_spin_operator(U_phy,3,3);
    AA_S1L,_,_,_,_=build_double_layer_extra_leg(A_fused,S1L);
    AA_S1R,_,_,_,_=build_double_layer_extra_leg(A_fused,S1R);
    AA_S2L,_,_,_,_=build_double_layer_extra_leg(A_fused,S2L);
    AA_S2R,_,_,_,_=build_double_layer_extra_leg(A_fused,S2R);
    AA_S3L,_,_,_,_=build_double_layer_extra_leg(A_fused,S3L);
    AA_S3R,_,_,_,_=build_double_layer_extra_leg(A_fused,S3R);

    AA_SS12, _,_,_,_=build_double_layer(A_fused,SS12);
    AA_SS31, _,_,_,_=build_double_layer(A_fused,SS31);
    AA_SS23, _,_,_,_=build_double_layer(A_fused,SS23);


    
    norms=evaluate_correl_spinspin("x", AA_fused, AA_fused, AA_fused, CTM, "dimerdimer", 10);
    norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
    norms=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_fused, AA_fused, CTM, "dimerdimer", distance);
    SS12_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SS12, AA_SS12, CTM, "dimerdimer", distance);
    SS23_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SS23, AA_SS23, CTM, "dimerdimer", distance);
    SS31_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SS12, AA_SS31, CTM, "dimerdimer", distance);
    S1_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_S1L, AA_S1R, CTM, "spinspin", distance);
    S2_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_S2L, AA_S2R, CTM, "spinspin", distance);
    S3_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_S3L, AA_S3R, CTM, "spinspin", distance);


    SS12_ob=SS12_ob./norms;
    SS23_ob=SS23_ob./norms;
    SS31_ob=SS31_ob./norms;
    S1_ob=S1_ob./norms;
    S2_ob=S2_ob./norms;
    S3_ob=S3_ob./norms;


    eu_allspin_x,allspin_x=solve_correl_length(5,AA_fused/norm_coe,CTM,"x",ctm_setting);
    eu_allspin_y,allspin_y=solve_correl_length(5,AA_fused/norm_coe,CTM,"y",ctm_setting);



    mat_filenm="correl_D"*string(D)*"_chi"*string(chi)*".mat";
    matwrite(mat_filenm, Dict(
        "E_up"=>E_up,
        "E_down"=>E_down,
        "E_up_12"=>E_up_12,
        "E_up_31"=>E_up_31,
        "E_up_23"=>E_up_23,
        "E_down_12"=>E_down_12,
        "E_down_31"=>E_down_31,
        "E_down_23"=>E_down_23,
        "SS12_ob" => SS12_ob,
        "SS23_ob" => SS23_ob,
        "SS31_ob" => SS31_ob,
        "S1_ob" => S1_ob,
        "S2_ob" => S2_ob,
        "S3_ob" => S3_ob,
        "eu_allspin_x" => eu_allspin_x,
        "allspin_x"=> allspin_x,
        "eu_allspin_y" => eu_allspin_y,
        "allspin_y"=> allspin_y,
        "CTM_space"=> string(space(CTM.Cset.C1))
    ); compress = false)
end