
function cost_fun(x::Matrix{T})  where T<:iPEPS_ansatz_immutable #variational parameters are vector of TensorMap
    global Lx,Ly
    global chi, parameters, energy_setting, grad_ctm_setting

    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            if isa(x[cx,cy],Triangle_iPESS_immutable)
                tm=x[cx,cy].Tm;#|LU><M|
                bm=x[cx,cy].Bm;#|Md><|RD
                A=permute(tm*bm,(1,5,4,2,3,));#L,D,R,U,d,
            end
            norm_A=norm(A)
            A=A/norm_A;

            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end
    
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(A_cell,chi,init,[],grad_ctm_setting);
    E_total,  _=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
    E=real(E_total);

    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM_cell);
    E_tem=deepcopy(E)
    return E
end





function energy_CTM(x, chi, parameters, ctm_setting, energy_setting, init, init_CTM)
    global Lx,Ly
    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            if isa(x[cx,cy],Triangle_iPESS)
                tm=x[cx,cy].Tm;#|LU><M|
                bm=x[cx,cy].Bm;#|Md><|RD
                A=permute(tm*bm,(1,5,4,2,3,));#L,D,R,U,d,
            end
            norm_A=norm(A)
            A=A/norm_A;

            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end

    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(x0,chi,init, init_CTM,ctm_setting);    

    if energy_setting.model == "spinful_triangle_lattice"
        E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting);
        E=real(E_total);

        if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
            sx_set,sy_set,sz_set=evaluate_spin_cell(A_cell, AA_cell, CTM_cell, ctm_setting);
            S2=sqrt.(sx_set.^2+sy_set.^2+sz_set.^2);
            println("S2= "*string(abs.(S2))*", sx= "*string(sx_set)*", sy= "*string(sy_set)*", sz= "*string(sz_set));flush(stdout);
        end

        return E, ex_set, ey_set, e_diagonala_set, e0_set, eU_set, ite_num,ite_err,CTM_cell
    elseif energy_setting.model == "standard_triangle_Hubbard"
        if ctm_setting.grad_checkpoint #use checkpoint to save memory
            E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set, triangle_up_set, triangle_dn_set=Zygote.checkpointed(evaluate_ob_cell, parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting);
        else
            E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set, triangle_up_set, triangle_dn_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting);
        end
        E=real(E_total);

        if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
            sx_set,sy_set,sz_set=evaluate_spin_cell(A_cell, AA_cell, CTM_cell, ctm_setting);
            S2=sqrt.(sx_set.^2+sy_set.^2+sz_set.^2);
            println("S2= "*string(abs.(S2))*", sx= "*string(sx_set)*", sy= "*string(sy_set)*", sz= "*string(sz_set));flush(stdout);
        end

        return E, ex_set, ey_set, e_diagonala_set, e0_set, eU_set, triangle_up_set, triangle_dn_set, ite_num,ite_err,CTM_cell
    end
end



function observable_CTM(x, chi, parameters, ctm_setting, energy_setting, init, init_CTM)
    global Lx,Ly
    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            if isa(x[cx,cy],Square_iPEPS)
                A=x[cx,cy].T;
            elseif isa(x[cx,cy],Triangle_iPESS)
                tm=x[cx,cy].Tm;#|LU><M|
                bm=x[cx,cy].Bm;#|Md><|RD
                A=permute(tm*bm,(1,5,4,2,3,));#L,D,R,U,d,
            end
            norm_A=norm(A)
            A=A/norm_A;

            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end


    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(A_cell,chi,init, init_CTM,ctm_setting);
    

    if energy_setting.model=="spinful_triangle_lattice"
        E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, e_diagonala_set, e0_set, eU_set, ite_num,ite_err,CTM_cell
    end
end




function evaluate_all_ob_cell(A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    

    global Lx,Ly

    if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        
        if (energy_setting.model == "Triangle_Hofstadter_Hubbard")|(energy_setting.model == "spinful_triangle_lattice")
            Hamiltonian_terms=Hamiltonians_spinful_Z2;
        elseif (energy_setting.model == "Triangle_Hofstadter_spinless")
            Hamiltonian_terms=Hamiltonians_spinless_Z2;
        end
    elseif isa(space(A_cell[1][1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(A_cell[1][1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(A_cell[1][1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        if mod(energy_setting.Magnetic_cell,2)==1 #odd number of sites in unitcell
            @assert mod(Ly,2)==0;
            #if use U1 symmetry, use different dummy physical space along y direction along Ly, where Ly should be even number
        end
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end


    if energy_setting.model=="spinful_triangle_lattice"
        Ident_set, N_occu_set, N_hole_set, N_double_set, Cdag_set, C_set, CdagupCdagdn_set, Pairinga_set, Pairingb_set, Sa_set, Sb_set =@ignore_derivatives Operators_spinful_SU2();

        hop_x_set=zeros(Lx,Ly)*im;
        hop_y_set=zeros(Lx,Ly)*im;
        hop_diagonala_set=zeros(Lx,Ly)*im;
        occu_set=zeros(Lx,Ly)*im;
        holon_set=zeros(Lx,Ly)*im;
        doublon_set=zeros(Lx,Ly)*im;
        cdagupcdagdn_set=zeros(Lx,Ly)*im;
        pairing_x_set=zeros(Lx,Ly)*im;
        pairing_y_set=zeros(Lx,Ly)*im;
        pairing_diagonala_set=zeros(Lx,Ly)*im;
        ss_x_set=zeros(Lx,Ly)*im;
        ss_y_set=zeros(Lx,Ly)*im;
        ss_diagonala_set=zeros(Lx,Ly)*im;

        
        for cx=1:Lx
            for cy=1:Ly

                hop_x_set[cx,cy]=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                hop_y_set[cx,cy]=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                hop_diagonala_set[cx,cy]=hopping_diagonala(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                occu_set[cx,cy]=ob_onsite(CTM_cell,N_occu_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                doublon_set[cx,cy]=ob_onsite(CTM_cell,N_double_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                holon_set[cx,cy]=ob_onsite(CTM_cell,N_hole_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                cdagupcdagdn_set[cx,cy]=ob_onsite(CTM_cell,CdagupCdagdn_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);

                pairing_x_set[cx,cy]=hopping_x_no_sign(CTM_cell,Pairinga_set[mod1(cx+1,Lx)],Pairingb_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                pairing_y_set[cx,cy]=hopping_y_no_sign(CTM_cell,Pairinga_set[mod1(cx+2,Lx)],Pairingb_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                pairing_diagonala_set[cx,cy]=hopping_diagonala_no_sign(CTM_cell,Pairinga_set[mod1(cx+1,Lx)],Pairingb_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                ss_x_set[cx,cy]=hopping_x_no_sign(CTM_cell,Sa_set[mod1(cx+1,Lx)],Sb_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                ss_y_set[cx,cy]=hopping_y_no_sign(CTM_cell,Sa_set[mod1(cx+2,Lx)],Sb_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
                ss_diagonala_set[cx,cy]=hopping_diagonala_no_sign(CTM_cell,Sa_set[mod1(cx+1,Lx)],Sb_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);



            end
        end
        

        println("hop_x_set: "*string(hop_x_set));
        println("hop_y_set: "*string(hop_y_set));
        println("hop_diagonala_set: "*string(hop_diagonala_set));
        println("occu_set: "*string(occu_set));
        println("doublon_set: "*string(doublon_set));
        println("holon_set: "*string(holon_set));
        println("cdagupcdagdn_set: "*string(cdagupcdagdn_set));
        println("pairing_x_set: "*string(pairing_x_set));
        println("pairing_y_set: "*string(pairing_y_set));
        println("pairing_diagonala_set: "*string(pairing_diagonala_set));
        println("ss_x_set: "*string(ss_x_set));
        println("ss_y_set: "*string(ss_y_set));
        println("ss_diagonala_set: "*string(ss_diagonala_set));
        flush(stdout);
        
        return   hop_x_set,hop_y_set,hop_diagonala_set,occu_set,doublon_set,holon_set,cdagupcdagdn_set,pairing_x_set,pairing_y_set,pairing_diagonala_set,ss_x_set,ss_y_set,ss_diagonala_set
    end
end