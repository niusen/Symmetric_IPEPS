function cost_fun(coe,Sigma,A_set,E_set,CTM)
    
    PEPS_tensor,A_fused,U_phy=build_PEPS(A_set,E_set,coe);
    


    # CTM=[];
    # U_L=[];
    # U_D=[];
    # U_R=[];
    # U_U=[];

    # init=Dict([("CTM", []), ("init_type", "PBC")]);
    # conv_check="singular_value";
    # CTM_ite_info=true;
    # CTM_conv_info=true;
    # CTM_conv_tol=1e-6;
    # CTM_ite_nums=100;
    # CTM_trun_tol=1e-12;
    # chi=40;
    # @time CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);






    rho=build_density_op(U_phy, PEPS_tensor, AA_fused, U_L,U_D,U_R,U_U, CTM);#L',U',R',D',  L,U,R,D


    E=real(plaquatte_ob(rho,Sigma))

    
    return E

end