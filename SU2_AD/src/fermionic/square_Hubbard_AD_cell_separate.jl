

function cost_fun_separate(x::iPEPS_ansatz)  
    global Lx,Ly
    global psi,px,py



    global chi, parameters, energy_setting, grad_ctm_setting

    A_cell=initial_tuple_cell(Lx,Ly);
    if (Lx==2)&(Ly==1)
        if px==1
            A1=x.T;
            A2=psi[2,1].T;

            norm_A1=norm(A1)
            A1=A1/norm_A1;
            norm_A2=norm(A2)
            A2=A2/norm_A2;

            A_cell=fill_tuple(A_cell, A1, 1,1);
            A_cell=fill_tuple(A_cell, A2, 2,1);
        elseif px==2
            A1=psi[1,1].T;
            A2=x.T;

            norm_A1=norm(A1)
            A1=A1/norm_A1;
            norm_A2=norm(A2)
            A2=A2/norm_A2;

            A_cell=fill_tuple(A_cell, A1, 1,1);
            A_cell=fill_tuple(A_cell, A2, 2,1);
        end
        
    end
    
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init,[],grad_ctm_setting);
    E_total,  _=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
    E=real(E_total);

    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM_cell);
    E_tem=deepcopy(E)
    return E
end





function energy_CTM_separate(x, chi, parameters, ctm_setting, energy_setting, init, init_CTM)
    global Lx,Ly
    global psi,px,py
    ##############
    psi[px,py]=x; #update cell
    ##############
    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=psi[cx,cy].T;
            norm_A=norm(A)
            A=A/norm_A;

            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end


    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,ctm_setting);
    

    if energy_setting.model=="spinless_Hubbard"
        E_total,  ex_set, ey_set, e0_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, e0_set, ite_num,ite_err,CTM_cell
    elseif energy_setting.model=="spinless_Hubbard_pairing"
        E_total,  ex_set, ey_set, px_set, py_set, e0_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, px_set, py_set, e0_set, ite_num,ite_err,CTM_cell
    elseif energy_setting.model=="spinless_triangle_lattice"
        E_total,  ex_set, ey_set, e_diagonala_set, e0_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, e_diagonala_set, e0_set, ite_num,ite_err,CTM_cell
    elseif energy_setting.model=="spinful_triangle_lattice"
        E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, e_diagonala_set, e0_set, eU_set, ite_num,ite_err,CTM_cell
    end
end




