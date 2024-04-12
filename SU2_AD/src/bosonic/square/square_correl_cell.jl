function build_double_layer_extra_leg(A,operator)
    #su2 operator has three legs, such as svd decomposition of Heisenberg interaction 
    #first two indices of operator are physical indices
    A=permute(A,(1,2,),(3,4,5));
    U_L=@ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2))*(1+0*im);
    # U_R=(U_L)';
    # U_U=(U_D)';
    U_R=@ignore_derivatives unitary(space(A, 3) ⊗ space(A, 3)', fuse(space(A, 3)' ⊗ space(A, 3)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A, 4) ⊗ space(A, 4)', fuse(space(A, 4)' ⊗ space(A, 4)))*(1+0*im);
    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uM,sM,vM=tsvd(A);
    uM=uM*sM

    uM=permute(uM,(1,2,3,),())
    V=space(vM,1);
    U=unitary(fuse(V' ⊗ V), V' ⊗ V);
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

function single_spin_operator(U_phy)

    # Heisenberg interaction
    Id=TensorMap(Matrix(I,2,2),U_phy,U_phy);
    sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
    @tensor HSS[:]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4];
    HSS=TensorMap(HSS, U_phy⊗U_phy ← U_phy⊗U_phy);
    HSS=permute(HSS,(1,3,),(2,4,));
    u,s,v=tsvd(HSS);
    H1=u*s;
    H2=permute(v,(2,3,),(1,));

    return H1,H2 
end



function evaluate_correl_spinspin(pos,coe,direction, AA_fused, AA_op1, AA_op2, CTM, method, distance)
    correl_funs=Vector(undef,distance);

    Cset=CTM.Cset;
    Tset=CTM.Tset;
    px=pos[1];
    py=pos[2];
    global Lx,Ly
    if method=="dimerdimer"#operator on a single site conserves su2 symmetry
        if direction=="x"
            @tensor va[:]:=Cset[mod1(px-1,Lx)][mod1(py-1,Ly)].C1[1,3]*Tset[mod1(px-1,Lx)][mod1(py,Ly)].T4[2,5,1]*Cset[mod1(px-1,Lx)][mod1(py+1,Ly)].C4[7,2]*Tset[mod1(px,Lx)][mod1(py-1,Ly)].T1[3,4,-1]*AA_op1[mod1(px,Lx)][mod1(py,Ly)][5,6,-2,4]*Tset[mod1(px,Lx)][mod1(py+1,Ly)].T3[-3,6,7];
            qx=px+1;
            qy=py;
            @tensor vb[:]:=Tset[mod1(qx,Lx)][mod1(qy-1,Ly)].T1[-1,4,3]*AA_op2[mod1(qx,Lx)][mod1(qy,Ly)][-2,6,5,4]*Tset[mod1(qx,Lx)][mod1(qy+1,Ly)].T3[7,6,-3]*Cset[mod1(qx+1,Lx)][mod1(qy-1,Ly)].C2[3,1]*Tset[mod1(qx+1,Lx)][mod1(qy,Ly)].T2[1,5,2]*Cset[mod1(qx+1,Lx)][mod1(qy+1,Ly)].C3[2,7];
            ov=@tensor va[1,2,3]*vb[1,2,3]
            correl_funs[1]=ov;
            
            for dis=2:distance
                qx=px+dis-1;
                qy=py;
                @tensor va[:]:=va[1,3,5]*Tset[mod1(qx,Lx)][mod1(qy-1,Ly)].T1[1,2,-1]*AA_fused[mod1(qx,Lx)][mod1(qy,Ly)][3,4,-2,2]*Tset[mod1(qx,Lx)][mod1(qy+1,Ly)].T3[-3,4,5];
                va=va*coe;
                
                qx=px+dis;
                qy=py;
                @tensor vb[:]:=Tset[mod1(qx,Lx)][mod1(qy-1,Ly)].T1[-1,4,3]*AA_op2[mod1(qx,Lx)][mod1(qy,Ly)][-2,6,5,4]*Tset[mod1(qx,Lx)][mod1(qy+1,Ly)].T3[7,6,-3]*Cset[mod1(qx+1,Lx)][mod1(qy-1,Ly)].C2[3,1]*Tset[mod1(qx+1,Lx)][mod1(qy,Ly)].T2[1,5,2]*Cset[mod1(qx+1,Lx)][mod1(qy+1,Ly)].C3[2,7];
                ov=@tensor va[1,2,3]*vb[1,2,3]
                correl_funs[dis]=ov;
            end
            return correl_funs
        end
    elseif method=="spinspin" #operator on a single site breaks su2 symmetry, so there is an extra index obtained from svd of two-site operator
        if direction=="x"
            @tensor va[:]:=Cset[mod1(px-1,Lx)][mod1(py-1,Ly)].C1[1,3]*Tset[mod1(px-1,Lx)][mod1(py,Ly)].T4[2,5,1]*Cset[mod1(px-1,Lx)][mod1(py+1,Ly)].C4[7,2]*Tset[mod1(px,Lx)][mod1(py-1,Ly)].T1[3,4,-1]*AA_op1[mod1(px,Lx)][mod1(py,Ly)][5,6,-2,4,-4]*Tset[mod1(px,Lx)][mod1(py+1,Ly)].T3[-3,6,7];
            qx=px+1;
            qy=py;
            @tensor vb[:]:=Tset[mod1(qx,Lx)][mod1(qy-1,Ly)].T1[-1,4,3]*AA_op2[mod1(qx,Lx)][mod1(qy,Ly)][-2,6,5,4,-4]*Tset[mod1(qx,Lx)][mod1(qy+1,Ly)].T3[7,6,-3]*Cset[mod1(qx+1,Lx)][mod1(qy-1,Ly)].C2[3,1]*Tset[mod1(qx+1,Lx)][mod1(qy,Ly)].T2[1,5,2]*Cset[mod1(qx+1,Lx)][mod1(qy+1,Ly)].C3[2,7];
            ov=@tensor va[1,2,3,4]*vb[1,2,3,4]
            correl_funs[1]=ov;
            
            for dis=2:distance

                qx=px+dis-1;
                qy=py;
                @tensor va[:]:=va[1,3,5,-4]*Tset[mod1(qx,Lx)][mod1(qy-1,Ly)].T1[1,2,-1]*AA_fused[mod1(qx,Lx)][mod1(qy,Ly)][3,4,-2,2]*Tset[mod1(qx,Lx)][mod1(qy+1,Ly)].T3[-3,4,5];
                va=va*coe;

                qx=px+dis;
                qy=py;
                @tensor vb[:]:=Tset[mod1(qx,Lx)][mod1(qy-1,Ly)].T1[-1,4,3]*AA_op2[mod1(qx,Lx)][mod1(qy,Ly)][-2,6,5,4,-4]*Tset[mod1(qx,Lx)][mod1(qy+1,Ly)].T3[7,6,-3]*Cset[mod1(qx+1,Lx)][mod1(qy-1,Ly)].C2[3,1]*Tset[mod1(qx+1,Lx)][mod1(qy,Ly)].T2[1,5,2]*Cset[mod1(qx+1,Lx)][mod1(qy+1,Ly)].C3[2,7];
                ov=@tensor va[1,2,3,4]*vb[1,2,3,4]
                correl_funs[dis]=ov;
            end
            return correl_funs
        end
    end
end


function correl_TransOp(vl,Tup_cell,Tdown_cell,AAfused_cell,direction)
    if AAfused_cell==[]
        if direction=="x"
        
            @tensor vl[:]:=vl[-1,1,3]*Tup_cell[1,2][1,2,4]*Tup_cell[2,2][4,6,-2]*Tdown_cell[1,1][5,2,3]*Tdown_cell[2,1][-3,6,5];
        elseif direction=="y"
            @tensor vl[:]:=vl[-1,1,3]*Tup_cell[1,1][1,2,4]*Tup_cell[1,2][4,6,-2]*Tdown_cell[2,1][5,2,3]*Tdown_cell[2,2][-3,6,5];
        end
        
    else
        
        @tensor vl[:]:=vl[-1,1,3,5]*Tup[1,2,-2]*AAfused[3,4,-3,2]*Tdown[-4,4,5];
        
    end
    return vl
end
function solve_correl_length_single_layer(n_values,AA_fused,CTM,direction)
    T1=CTM["Tset"][1];
    T2=CTM["Tset"][2];
    T3=CTM["Tset"][3];
    T4=CTM["Tset"][4];
    if direction=="x"
        correl_TransOp_fx(x)=correl_TransOp(x,T1,T3,[],direction)
        vl_init = permute(TensorMap(randn, SU₂Space(0=>1)⊗space(T1[1,2],1)', space(T3[1,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
        eu_S0=eu;

        eu_allspin=eu_S0;
        allspin=eu_S0*0;
        vl_init = permute(TensorMap(randn, SU₂Space(1/2=>1)⊗space(T1[1,2],1)', space(T3[1,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu_S0d5,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eu_allspin=vcat(eu_allspin,eu_S0d5)
            allspin=vcat(allspin,0*eu_S0d5.+0.5)
        end

        vl_init = permute(TensorMap(randn, SU₂Space(1=>1)⊗space(T1[1,2],1)', space(T3[1,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu_S1,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eu_allspin=vcat(eu_allspin,eu_S1)
            allspin=vcat(allspin,0*eu_S1.+1)
        end

        vl_init = permute(TensorMap(randn, SU₂Space(3/2=>1)⊗space(T1[1,2],1)', space(T3[1,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu_S1d5,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eu_allspin=vcat(eu_allspin,eu_S1d5)
            allspin=vcat(allspin,0*eu_S1d5.+1.5)
        end

        vl_init = permute(TensorMap(randn, SU₂Space(2=>1)⊗space(T1[1,2],1)', space(T3[1,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu_S2,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eu_allspin=vcat(eu_allspin,eu_S2)
            allspin=vcat(allspin,0*eu_S2.+2)
        end

        eu_allspin_abs=abs.(eu_allspin);
        @assert maximum(eu_allspin_abs)==eu_allspin_abs[1]

        eu_allspin_abs_sorted=sort(eu_allspin_abs,rev=true);
        eu_allspin_abs_sorted=eu_allspin_abs_sorted/eu_allspin_abs_sorted[1];
        allspin=allspin[sortperm(eu_allspin_abs,rev=true)]

        
        return eu_allspin_abs_sorted,allspin
    elseif direction=="y"

        correl_TransOp_fy(x)=correl_TransOp(x,T2,T4,[],direction)
        vl_init = permute(TensorMap(randn, SU₂Space(0=>1)⊗space(T2[1,1],1)', space(T4[2,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(correl_TransOp_fy, vl_init, n_values,:LM,Arnoldi());
        eu_S0=eu;
        
        eu_allspin=eu_S0;
        allspin=eu_S0*0;
        vl_init = permute(TensorMap(randn, SU₂Space(1/2=>1)⊗space(T2[1,1],1)', space(T4[2,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu_S0d5,_=eigsolve(correl_TransOp_fy, vl_init, n_values,:LM,Arnoldi());
            eu_allspin=vcat(eu_allspin,eu_S0d5)
            allspin=vcat(allspin,0*eu_S0d5.+0.5)
        end
        
        vl_init = permute(TensorMap(randn, SU₂Space(1=>1)⊗space(T2[1,1],1)', space(T4[2,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu_S1,_=eigsolve(correl_TransOp_fy, vl_init, n_values,:LM,Arnoldi());
            eu_allspin=vcat(eu_allspin,eu_S1)
            allspin=vcat(allspin,0*eu_S1.+1)
        end
        
        vl_init = permute(TensorMap(randn, SU₂Space(3/2=>1)⊗space(T2[1,1],1)', space(T4[2,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu_S1d5,_=eigsolve(correl_TransOp_fy, vl_init, n_values,:LM,Arnoldi());
            eu_allspin=vcat(eu_allspin,eu_S1d5)
            allspin=vcat(allspin,0*eu_S1d5.+1.5)
        end
        
        vl_init = permute(TensorMap(randn, SU₂Space(2=>1)⊗space(T2[1,1],1)', space(T4[2,1],3)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu_S2,_=eigsolve(correl_TransOp_fy, vl_init, n_values,:LM,Arnoldi());
            eu_allspin=vcat(eu_allspin,eu_S2)
            allspin=vcat(allspin,0*eu_S2.+2)
        end
        
        eu_allspin_abs=abs.(eu_allspin);
        @assert maximum(eu_allspin_abs)==eu_allspin_abs[1]
        
        eu_allspin_abs_sorted=sort(eu_allspin_abs,rev=true);
        eu_allspin_abs_sorted=eu_allspin_abs_sorted/eu_allspin_abs_sorted[1];
        allspin=allspin[sortperm(eu_allspin_abs,rev=true)]
        
        return eu_allspin_abs_sorted,allspin
    end
end


function cal_correl(CTM_cell,A_cell,AA_cell,D,chi,parameters,distance)
    global Lx,Ly
    E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);

    println(E_total/Lx/Ly);flush(stdout);

    U_phy=space(A_cell[1][1],5)';
    S1L,S1R=single_spin_operator(U_phy);
    AA_L_set=initial_tuple_cell(Lx,Ly);
    AA_R_set=initial_tuple_cell(Lx,Ly);
    Spin_ob_set=Vector{Any}(undef,Lx*Ly);
    for ca=1:Lx
        for cb=1:Ly
            AA_L,_,_,_,_=build_double_layer_extra_leg(A_cell[ca][cb],S1L);
            AA_R,_,_,_,_=build_double_layer_extra_leg(A_cell[ca][cb],S1R);
            AA_L_set=fill_tuple(AA_L_set, AA_L, ca,cb);
            AA_R_set=fill_tuple(AA_R_set, AA_R, ca,cb);
        end
    end

    step=1;
    for ca=1:Lx
        for cb=1:Ly
            norms=evaluate_correl_spinspin([ca,cb],1,"x", AA_cell, AA_cell, AA_cell, CTM_cell, "dimerdimer", distance);
            norm_coe=(norms[4+Lx]/norms[4])^(1/Lx); #get a rough normalization coefficient to avoid that the number becomes two small
            norms=evaluate_correl_spinspin([ca,cb],1/norm_coe,"x", AA_cell, AA_cell, AA_cell, CTM_cell, "dimerdimer", distance);
            Spin_ob=evaluate_correl_spinspin([ca,cb], 1/norm_coe, "x", AA_cell, AA_L_set, AA_R_set, CTM_cell, "spinspin", distance);

            # SS12_ob=SS12_ob./norms;
            # SS23_ob=SS23_ob./norms;
            # SS31_ob=SS31_ob./norms;
            Spin_ob=Spin_ob./norms;
            Spin_ob_set[step]=Spin_ob;
            step=step+1;
            # eu_allspin_x,allspin_x=solve_correl_length(1/norm_coe, 5, AA_cell,CTM_cell,"x");
            # eu_allspin_y,allspin_y=solve_correl_length(1/norm_coe, 5, AA_cell,CTM_cell,"y");
        end
    end

    mat_filenm="correl_D"*string(D)*"_chi"*string(chi)*".mat";
    matwrite(mat_filenm, Dict(
        "Spin_ob_set" => Spin_ob_set
        # "eu_allspin_x" => eu_allspin_x,
        # "allspin_x"=> allspin_x,
        # "eu_allspin_y" => eu_allspin_y,
        # "allspin_y"=> allspin_y
    ); compress = false)
end