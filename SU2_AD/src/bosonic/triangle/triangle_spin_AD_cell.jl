function create_isometry(V1,V2)
    #V1 is larger than V2
    @assert dim(V1)>=dim(V2)
    tt=TensorMap(randn,V1,V2);
    for cc=1:length(tt.data.values)
        mm=tt.data.values[cc];
        tt.data.values[cc]=Matrix(I, size(mm,1), size(mm,2));
    end
    return tt
end
function add_noise(A::TensorMap,nois,init_complex_tensor)
    if init_complex_tensor
        A_noise=TensorMap(randn,codomain(A),domain(A))+im*TensorMap(randn,codomain(A),domain(A));
    else
        A_noise=TensorMap(randn,codomain(A),domain(A));
    end

    A_new=A+A_noise*nois*norm(A)/norm(A_noise);
    return A_new
end
function initial_iPESS_SU2_SU2(init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    Vp=SU2Space(0=>2,1/2=>1)';
    global Lx,Ly
    if init_statenm=="nothing" 
        
        # println("Random initial state");flush(stdout);
        
        # state=Matrix{Square_iPEPS}(undef,Lx,Ly);
        # for cx=1:Lx
        #     for cy=1:Ly
        #         if init_complex_tensor
        #             A=TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp)+TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp)*im;
        #         else
        #             A=TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp);
        #         end
        #         A=permute(A,(1,2,3,4,5,));
        #         A=A/norm(A);
        #         state[cx,cy]=Square_iPEPS(A);
        #     end
        # end
        # return state
    else
        println("load iPESS state: "*init_statenm);flush(stdout);
        x=load(init_statenm)["x"];
        state=similar(x);
        for cc in eachindex(x)
            ansatz=x[cc];
            tm=ansatz.Tm;
            tm=add_noise(tm,init_noise,init_complex_tensor);
            bm=ansatz.Bm;
            bm=add_noise(bm,init_noise,init_complex_tensor);
            ansatz_new=Triangle_iPESS(bm,tm);
            state[cc]=ansatz_new;
            println(space(tm))
        end
        return state
    end
end



function cost_fun(x::Matrix{T})  where T<:iPEPS_ansatz_immutable #variational parameters are vector of TensorMap
    global Lx,Ly
    global chi, parameters, energy_setting, grad_ctm_setting

    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            if isa(x[cx,cy],Square_iPEPS_immutable)
                A=x[cx,cy].T;
            elseif isa(x[cx,cy],Triangle_iPESS_immutable)
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
    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell,chi,init,[],grad_ctm_setting);
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


    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell,chi,init, init_CTM,ctm_setting);
    

    if energy_setting.model =="triangle_SU4_spin"
        E_total,  ex_set, ey_set, e_diagonal_set, triangle_right_bot_set, triangle_left_top_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting);
        E=real(E_total);

        # if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #     sx_set,sy_set,sz_set=evaluate_spin_cell(A_cell, AA_cell, CTM_cell, ctm_setting);
        #     S2=sqrt.(sx_set.^2+sy_set.^2+sz_set.^2);
        #     println("S2= "*string(abs.(S2))*", sx= "*string(sx_set)*", sy= "*string(sy_set)*", sz= "*string(sz_set));flush(stdout);
        # end

        return E, ex_set, ey_set, e_diagonala_set, e0_set, eU_set, ite_num,ite_err,CTM_cell
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


    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell,chi,init, init_CTM,ctm_setting);
    

    if energy_setting.model=="spinful_triangle_lattice"
        E_total,  ex_set, ey_set, e_diagonal_set, triangle_right_bot_set, triangle_left_top_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, e_diagonal_set, triangle_right_bot_set, triangle_left_top_set, ite_num,ite_err,CTM_cell
    end
end






