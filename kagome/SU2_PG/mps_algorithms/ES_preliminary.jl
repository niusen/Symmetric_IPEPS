function try_CTM(D,chi,parameters, U_phy, A_unfused, A_fused)
    CTM=[];
    U_L=[];
    U_D=[];
    U_R=[];
    U_U=[];
    try
        #load data
        jld_filenm="CTM_D"*string(D)*"_chi"*string(chi)*".jld";
        CTM_dict=load(jld_filenm)["CTM_dict"];
        U_L_dict=load(jld_filenm)["U_L_dict"];
        U_D_dict=load(jld_filenm)["U_D_dict"];
        U_R_dict=load(jld_filenm)["U_R_dict"];
        U_U_dict=load(jld_filenm)["U_U_dict"];
        CTM=deepcopy(CTM_dict)
        for cc=1:4
            CTM["Tset"][cc]=convert(TensorMap,CTM_dict["Tset"][cc]);
            CTM["Cset"][cc]=convert(TensorMap,CTM_dict["Cset"][cc]);
        end
        U_L=convert(TensorMap,U_L_dict);
        U_D=convert(TensorMap,U_D_dict);
        U_R=convert(TensorMap,U_R_dict);
        U_U=convert(TensorMap,U_U_dict);

        display("load CTM from saved data directly")
    catch e

        display("No CTM found from saved data, now do CTMRG")


        init=Dict([("CTM", []), ("init_type", "PBC")]);
        conv_check="singular_value";
        @time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(A_fused,chi,conv_check,tol,init);
        
        @time E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
        @time E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_bond");
        display((E_up+E_down)/3)

        
        CTM_dict=deepcopy(CTM)
        for cc=1:4
            CTM_dict["Tset"][cc]=convert(Dict,CTM_dict["Tset"][cc]);
            CTM_dict["Cset"][cc]=convert(Dict,CTM_dict["Cset"][cc]);
        end
        U_L_dict=convert(Dict,U_L);
        U_D_dict=convert(Dict,U_D);
        U_R_dict=convert(Dict,U_R);
        U_U_dict=convert(Dict,U_U);

        jld_filenm="CTM_D"*string(D)*"_chi"*string(chi)*".jld";
        save(jld_filenm, "CTM_dict",CTM_dict, "U_L_dict",U_L_dict,"U_D_dict",U_D_dict,"U_R_dict",U_R_dict,"U_U_dict",U_U_dict);


    end
    return CTM,U_L,U_D,U_R,U_U
end


function try_ITEBD(D,chi,W,CTM)
    O1=[];
    O2=[];
    Ag=[];
    try
        #load data
        jld_filenm="itebd_ground_D"*string(D)*"_chi"*string(chi)*"_W"*string(W)*".jld";
        Ag_dict=load(jld_filenm)["Ag_dict"];
        O1_dict=load(jld_filenm)["O1_dict"];
        O2_dict=load(jld_filenm)["O2_dict"];
        Ag=convert(TensorMap,Ag_dict);
        O1=convert(TensorMap,O1_dict);
        O2=convert(TensorMap,O2_dict);
        display("load itebd ground state from saved data directly")
    catch e

        display("No itebd ground state found from saved data, now do itebd")
        
        Tleft=CTM["Tset"][4];
        Tright=CTM["Tset"][2];
        @tensor O1[:]:=Tleft[-3,1,-1]*U_L[1,-2,-4];
        @tensor O2[:]:=Tright[-1,1,-3]*U_R[-4,-2,1];
        
        mps_virtual=SU₂Space(0=>1,1/2=>1,1=>1);mps_phy=space(O1,2);
        A_init=permute(TensorMap(randn, mps_virtual*mps_virtual', mps_phy),(1,2,3,),());

        #Ag,A_init=ITEBD_boundary_groundstate(O1,O2,W,A_init,"OO");
        Ag,A_init=ITEBD_boundary_groundstate(O1,O2,W,A_init,"O_O");

        #save itebd data
        O1_dict=convert(Dict,O1);
        O2_dict=convert(Dict,O2);
        Ag_dict=convert(Dict,Ag);
        jld_filenm="itebd_ground_D"*string(D)*"_chi"*string(chi)*"_W"*string(W)*".jld";
        save(jld_filenm, "O1_dict", O1_dict,"O2_dict", O2_dict,"Ag_dict", Ag_dict);

        #save initial CTM to compare with other codes
        matwrite("itebd_matfile.mat", Dict(
            "O1" => convert(Array,O1),
            "O2" => convert(Array,O2),
            "A_init" => convert(Array,A_init),
            "Ag" => convert(Array,Ag)
        ); compress = false)


    end
    return Ag,O1,O2
end