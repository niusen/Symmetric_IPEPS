import Base: +, -, *,/
import LinearAlgebra: norm,dot
#Define operations of groups of TensorMap
*(coe::Number,tt :: Vector{TensorMap})=add_group(tt,[],coe,0);
*(tt :: Vector{TensorMap}, coe:: Number)=add_group(tt,[],coe,0);
/(coe::Number,tt :: Vector{TensorMap})=add_group(tt,[],1/coe,0);
+(tt1 :: Vector{TensorMap}, tt2 :: Vector{TensorMap})=add_group(tt1,tt2,1,+1);
-(tt1 :: Vector{TensorMap}, tt2 :: Vector{TensorMap})=add_group(tt1,tt2,1,-1);
norm(tt :: Vector{TensorMap})=norm_tensor_group(tt);
dot(tt1 :: Vector{TensorMap}, tt2 :: Vector{TensorMap})=dot_tensor_group(tt1,tt2);

function define_tensor_group(b1,b2,b3,tup,tdn)
    state=Vector{TensorMap}(undef,5);
    state[1]=b1;
    state[2]=b2;
    state[3]=b3;
    state[4]=tup;
    state[5]=tdn;
    return state
end
function initial_SU2_state(Vspace,init_statenm="nothing",init_noise=0)
    if init_statenm=="nothing" 
        println("Random initial state");flush(stdout);
        Vp=SU2Space(1/2=>1);
        b1=TensorMap(randn,Vv*Vv,Vp);
        b2=TensorMap(randn,Vv*Vv,Vp);
        b3=TensorMap(randn,Vv*Vv,Vp);
        tup=TensorMap(randn,Vv',Vv*Vv);
        tdn=TensorMap(randn,Vv',Vv*Vv);
        b1=permute(b1,(1,2,3,));
        b2=permute(b2,(1,2,3,));
        b3=permute(b3,(1,2,3,));
        tup=permute(tup,(1,2,3,));
        tdn=permute(tdn,(1,2,3,));

        state=define_tensor_group(b1,b2,b3,tup,tdn)
        return state
    else
        
        println("load state: "*init_statenm);flush(stdout);
        data=load(init_statenm);
        B_a=data["B_a"];
        B_b=data["B_b"];
        B_c=data["B_c"];
        T_u=data["T_u"];
        T_d=data["T_d"];

        @assert space(B_a,1)==Vspace
        Ba_noise=TensorMap(randn,codomain(B_a),domain(B_a));
        Bb_noise=TensorMap(randn,codomain(B_b),domain(B_b));
        Bc_noise=TensorMap(randn,codomain(B_c),domain(B_c));
        Tu_noise=TensorMap(randn,codomain(T_u),domain(T_u));
        Td_noise=TensorMap(randn,codomain(T_d),domain(T_d));
        
        Ba=B_a+Ba_noise*init_noise*norm(B_a)/norm(Ba_noise);
        Bb=B_b+Bb_noise*init_noise*norm(B_b)/norm(Bb_noise);
        Bc=B_c+Bc_noise*init_noise*norm(B_c)/norm(Bc_noise);
        Tu=T_u+Tu_noise*init_noise*norm(T_u)/norm(Tu_noise);
        Td=T_d+Td_noise*init_noise*norm(T_d)/norm(Td_noise);

        state=define_tensor_group(Ba,Bb,Bc,Tu,Td)
        return state
    end

end


function get_grad(x)
    ∂E = cost_fun'(x)
    #E=fun(state_vec)
    global E_tem, CTM_tem
    x_tem=x;

    if isa(∂E, Vector{Float64})
        @assert !isnan(norm(∂E))
    elseif isa(∂E, Vector)
        for elem in ∂E
            @assert !isnan(norm(elem))
        end
    end
    
    return E_tem,∂E,CTM_tem
end

function norm_tensor_group(x0)
    Norm=0;
    for tt in x0
        Norm=Norm+norm(tt)^2;
    end
    Norm=sqrt(Norm);
    return Norm
end

function normalize_tensor_group(x0)
    Norm=norm_tensor_group(x0);
    #x_new=[[tt/Norm for tt in x0]...,]
    x_new=deepcopy(x0);
    for cc in eachindex(x0)
        x_new[cc]=x_new[cc]/Norm;
    end
    return x_new
end

function add_group(Tp1, Tp2, coe1, coe2)
    # if Tp2==[]
    #     x_new=[[tt*coe1 for tt in Tp1]...,];
    # else
    #     x_new=[[Tp1[cc]*coe1+Tp2[cc]*coe2 for cc in eachindex(Tp1)]...,];
    # end 
    x_new=deepcopy(Tp1);
    if Tp2==[]
        for cc in eachindex(Tp1)
            x_new[cc]=Tp1[cc]*coe1;
        end
    else
        for cc in eachindex(Tp1)
            x_new[cc]=Tp1[cc]*coe1+Tp2[cc]*coe2;
        end
    end
    return x_new
end
function dot_tensor_group(Tp1,Tp2)
    y=0;
    for cc in eachindex(Tp1)
        y=y+dot(Tp1[cc],Tp2[cc])
    end
    if imag(y)/real(y)<1e-10
        y=real(y);
    end
    return y
end


function FD(state_vec)

    dt=0.000001

    E0=cost_fun(state_vec);

    grad=similar(state_vec);
    for ct in eachindex(state_vec)
        grad[ct]=similar(state_vec[ct])*0;
    end

    for ct in eachindex(state_vec)
        for n_block in eachindex(state_vec[ct].data.values)
            for elem in eachindex(state_vec[ct].data.values[n_block])
                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem[ct].data.values[n_block];
                T[elem]=T[elem]+dt;
                state_vec_tem[ct].data.values[n_block]=T;
                real_part=(cost_fun(state_vec_tem)-E0)/dt;

                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem[ct].data.values[n_block];
                T[elem]=T[elem]+dt*im;
                state_vec_tem[ct].data.values[n_block]=T;
                imag_part=(cost_fun(state_vec_tem)-E0)/dt;

                grad[ct].data.values[n_block][elem]=real_part+im*imag_part;
            end
        end
    end
    return E0, grad
end

function cost_fun(x) #variational parameters are vector of TensorMap
    B1=x[1];
    B2=x[2];
    B3=x[3];
    Tup=x[4];
    Tdn=x[5];

    global chi, parameters, energy_setting, grad_ctm_setting

    @tensor PEPS_tensor[:] := B1[-1,1,-5]*B2[4,3,-6]*B3[-4,2,-7]*Tup[1,3,2]*Tdn[-3,4,-2];
    A_unfused=PEPS_tensor;

    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

    # norm_A=norm(A_fused)
    # A_fused= A_fused/norm_A;
    #CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true,true);
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,[],grad_ctm_setting)
    E_up, E_down=evaluate_ob(parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, grad_ctm_setting, energy_setting.kagome_method, energy_setting.E_up_method, energy_setting.E_dn_method);
    E=real(E_up+E_down)/3;
    #println(E)
    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM);
    E_tem=deepcopy(E)
    return E
end





function energy_CTM(x, chi, parameters, ctm_setting, energy_setting, init, init_CTM)
    B1=x[1];
    B2=x[2];
    B3=x[3];
    Tup=x[4];
    Tdn=x[5];

    @tensor PEPS_tensor[:] := B1[-1,1,-5]*B2[4,3,-6]*B3[-4,2,-7]*Tup[1,3,2]*Tdn[-3,4,-2];
    A_unfused=PEPS_tensor;

    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];




    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init, init_CTM, ctm_setting);

    #@assert ite_err<3*(1e-5)

    if (parameters["J2"]==0) & (parameters["J3"]==0)
        #kagome_method="E_single_triangle"
        E_up, E_down=evaluate_ob(parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method, energy_setting.E_up_method, energy_setting.E_dn_method);
        energy=(E_up+E_down)/3;
    elseif parameters["Jtrip"]==0
        #kagome_method="E_bond"
        E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23,   E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b,  E_NNNN_11,E_NNNN_22,E_NNNN_33=evaluate_ob(parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        E_NN=parameters["J1"]*(E_up_12+E_up_31+E_up_23+E_down_12+E_down_31+E_down_23);
        E_NNN=parameters["J2"]*(E_NNN_23a+E_NNN_12a+E_NNN_31a+E_NNN_23b+E_NNN_12b+E_NNN_31b);
        E_NNNN=parameters["J3"]*(E_NNNN_11+E_NNNN_22+E_NNNN_33);
        energy=(E_NN+E_NNN+E_NNNN)/3;
        println(real([E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23]))
        println(real([E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b]))
        println(real([E_NNNN_11,E_NNNN_22,E_NNNN_33]))
    else
        #kagome_method="E_triangle";
        E_up, E_down=evaluate_ob(parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        #kagome_method="E_bond";
        E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23,   E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b,  E_NNNN_11,E_NNNN_22,E_NNNN_33=evaluate_ob(parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        E_NN=parameters["J1"]*(E_up_12+E_up_31+E_up_23+E_down_12+E_down_31+E_down_23);
        E_NNN=parameters["J2"]*(E_NNN_23a+E_NNN_12a+E_NNN_31a+E_NNN_23b+E_NNN_12b+E_NNN_31b);
        E_NNNN=parameters["J3"]*(E_NNNN_11+E_NNNN_22+E_NNNN_33);
        energy=(E_up+E_down)/3+(E_NNN+E_NNNN)/3;
    end


    #return energy,CTM,U_L,U_D,U_R,U_U
    if energy_setting.cal_chiral_order
        chiral_order_parameters=Dict([("J1", 0), ("J2", 0), ("J3", 0), ("Jchi", 0), ("Jtrip", 1)]);
        chiral_order_up, chiral_order_down=evaluate_ob(chiral_order_parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM,  ctm_setting, "E_triangle");
        return real(energy),chiral_order_up, chiral_order_down,ite_num,ite_err,CTM
    else
        chiral_order_up=[];
        chiral_order_down=[];
        return real(energy),real(E_up), real(E_down),ite_num,ite_err,CTM
    end
    
end






function grad_line_search(E_old, state_vec, D, chi, parameters, optim_setting, LS_ctm_setting, energy_setting, grad0=None, direction0=None, alpha0=1, ls_ratio=1/3, ls_max=10)
    
    filenm="LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"

    state_vec=normalize_tensor_group(state_vec);

    global E_tem, CTM_tem
    @time E0_, grad,CTM_tem=get_grad(state_vec);

    if E_old==[]
        E0=E0_;
    else
        E0=E_old;#if there is only a few CTM steps are used for obtaining grad, then the energy from the last line search is more reliable
    end

    global grad_norm
    println("Norm of dA: "*string(grad_norm))

    
    # println("state: "*string(state_vec));flush(stdout);
    # println("grad: "*string(grad));flush(stdout);

    if optim_setting.linesearch_CTM_method=="from_converged_CTM"
        init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
        CTM0=deepcopy(CTM_tem);
    elseif optim_setting.linesearch_CTM_method=="restart"
        init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
        CTM0=[];
    end


    global LS_ctm_setting

    
    

    direction=add_group(grad,[],-1.0, 0);
    #print(grad0);flush(stdout);
    #print(grad);flush(stdout);
    if grad0==nothing
        #direction=-grad;
        direction=add_group(grad,[],-1.0, 0);
    else
        norm_grad=norm_tensor_group(grad)
        norm_grad0=norm_tensor_group(grad0)
        beta=(norm_grad^2)/(norm_grad0^2)
        #direction=-grad+beta*direction0;
        direction=add_group(grad,direction0,-1.0,beta);
    end
    vec0=deepcopy(state_vec);
    vec_tem=[];

    #line search
    improved=false
    alpha=alpha0
    println("line search");flush(stdout);
    println("E,chiral_order_up, chiral_order_down,ite_num,ite_err")
    println("conjugate gradient opt");flush(stdout);
    for ls_step=0:ls_max-1
        #vec_tem=vec0+direction*alpha*(ls_ratio^ls_step);
        vec_tem=add_group(vec0, direction, 1, alpha*(ls_ratio^ls_step));
        E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(vec_tem, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0);
        #println("Number of iterations for linesearch: "*string(ite_num));
        E=real(E);
        println(string(E)*", "*string(real(chiral_order_up))*", "*string(real(chiral_order_down))*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
        if E<E0
            improved=true
            break
        end
    end
    if improved
        vec=deepcopy(vec_tem);
        jldsave(filenm; B_a=vec[1],B_b=vec[2],B_c=vec[3],T_u=vec[4],T_d=vec[5]);
    else
        println("gradient opt");flush(stdout);
        for ls_step = 0:ls_max-1
            #vec_tem=vec0-grad*alpha*(ls_ratio^ls_step);
            vec_tem=add_group(vec0, grad, 1, -alpha*(ls_ratio^ls_step));
            
            E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(vec_tem, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0);
            #println("Number of iterations for linesearch: "*string(ite_num));
            E=real(E);
            println(string(E)*", "*string(real(chiral_order_up))*", "*string(real(chiral_order_down))*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
            if E<E0
                improved=true
                break
            end
        end
        if improved
            vec=deepcopy(vec_tem);
            jldsave(filenm; B_a=vec[1],B_b=vec[2],B_c=vec[3],T_u=vec[4],T_d=vec[5]);
        else
            vec=deepcopy(vec0)
            E=deepcopy(E0)
        end
    end
    improvement=E-E0
    
    return E,vec,grad,direction,improvement
end



function run_FiniteDiff(parameters, Vspace, chi, LS_ctm_setting, optim_setting, energy_setting)
    
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    if energy_setting.kagome_method =="E_single_triangle"
        println("Only compute energy in a single triangle");
    end


    state_vec=initial_SU2_state(Vspace, optim_setting.init_statenm, optim_setting.init_noise)
    state_vec=normalize_tensor_group(state_vec);



    println("optimization start");flush(stdout);
    grad=nothing;
    direction=nothing;
    alpha0=0.1;
    ls_ratio=1/3;
    ls_max=5;

    E0=[];
    for ite=1:100
        E,state_vec,grad,direction,improvement=grad_line_search(E0, state_vec, D, chi, parameters, optim_setting, LS_ctm_setting, energy_setting, grad, direction, alpha0, ls_ratio, ls_max)
        E0=deepcopy(E);
        println("grad norm: "*string(norm_tensor_group(grad)));flush(stdout)
        if -improvement<1e-7
            break
        end
    end

end

