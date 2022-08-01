function read_json_state(filenm)
    json_dict = Dict()
    open(filenm, "r") do f
        json_dict
        dicttxt = read(f,String)  # file information to string
        json_dict=JSON.parse(dicttxt)  # parse and transform data
    end
    return json_dict
end


function wrap_json_state(Bond_irrep,Triangle_irrep,Bond_A_coe,Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe)
    if Bond_irrep=="A"
        if Triangle_irrep=="A1"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe))]);
        elseif Triangle_irrep=="A2"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        elseif Triangle_irrep=="A1+iA2"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe)),("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        end
    elseif Bond_irrep=="B"
        if Triangle_irrep=="A1"
            coes=Dict([("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe))]);
        elseif Triangle_irrep=="A2"
            coes=Dict([("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        elseif Triangle_irrep=="A1+iA2"
            coes=Dict([("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe)),("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        end
    elseif Bond_irrep=="A+iB"
        if Triangle_irrep=="A1"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe))]);
        elseif Triangle_irrep=="A2"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        elseif Triangle_irrep=="A1+iA2"
            coes=Dict([("Bond_A_coe", create_coe_dict(Bond_A_coe)), ("Bond_B_coe", create_coe_dict(Bond_B_coe)), ("Triangle_A1_coe", create_coe_dict(Triangle_A1_coe)),("Triangle_A2_coe", create_coe_dict(Triangle_A2_coe))]);
        end
    end
    json_state=Dict([("coes" , coes), ("Bond_irrep", Bond_irrep), ("Triangle_irrep", Triangle_irrep)]);
    
    return json_state
end
function create_coe_dict(coe)
    #print(coe)
    entries=Vector(undef,length(coe));
    for cc=1:length(coe)
        entries[cc]=string(cc-1)*" "*string(coe[cc]);
    end
    dims=Vector(undef,1);
    dims[1]=length(coe);

    coe_dict=Dict([("dtype", "float64"), ("numEntries", length(coe)),("entries", entries), ("dims", dims)]);
    return coe_dict
end

function initial_state(Bond_irrep,Triangle_irrep,D,init_statenm=nothing,init_noise=0)
    if init_statenm==nothing 
        println("Random initial state");flush(stdout);
        A_set,B_set,A1_set,A2_set, _,_,_,_, _, _, _, _, _=construct_tensor(D);
        if Bond_irrep=="A"
            Bond_A_coe=randn(Float64, length(A_set));
            Bond_B_coe=[];
        elseif Bond_irrep=="B"
            Bond_A_coe=[];
            Bond_B_coe=randn(Float64, length(B_set));
        elseif Bond_irrep=="A+iB"
            Bond_A_coe=randn(Float64, length(A_set));
            Bond_B_coe=randn(Float64, length(B_set));
        end
        if Triangle_irrep=="A1"
            Triangle_A1_coe=randn(Float64, length(A1_set));
            Triangle_A2_coe=[];
        elseif Triangle_irrep=="A2"
            Triangle_A1_coe=[];
            Triangle_A2_coe=randn(Float64, length(A2_set));
        elseif Triangle_irrep=="A1+iA2"
            Triangle_A1_coe=randn(Float64, length(A1_set));
            Triangle_A2_coe=randn(Float64, length(A2_set));
        end
        
        json_state_dict=wrap_json_state(Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe)
    else
        println("load state: "*init_statenm);flush(stdout);
        json_state_dict=read_json_state(init_statenm);
        Bond_irrep_, Triangle_irrep_, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(json_state_dict)
        @assert Bond_irrep_==Bond_irrep
        @assert Triangle_irrep_==Triangle_irrep

        #add initial noise
        if Bond_irrep=="A"
            Bond_A_coe=Bond_A_coe+(rand(Float64, length(Bond_A_coe)).-0.5)*init_noise;
            Bond_B_coe=[];
        elseif Bond_irrep=="B"
            Bond_A_coe=[];
            Bond_B_coe=Bond_B_coe+(rand(Float64, length(Bond_B_coe)).-0.5)*init_noise;
        elseif Bond_irrep=="A+iB"
            Bond_A_coe=Bond_A_coe+(rand(Float64, length(Bond_A_coe)).-0.5)*init_noise;
            Bond_B_coe=Bond_B_coe+(rand(Float64, length(Bond_B_coe)).-0.5)*init_noise;
        end

        if Triangle_irrep=="A1"
            Triangle_A1_coe=Triangle_A1_coe+(rand(Float64, length(Triangle_A1_coe)).-0.5)*init_noise;
            Triangle_A2_coe=[];
        elseif Triangle_irrep=="A2"
            Triangle_A1_coe=[];
            Triangle_A2_coe=Triangle_A2_coe+(rand(Float64, length(Triangle_A2_coe)).-0.5)*init_noise;
        elseif Triangle_irrep=="A1+iA2"
            Triangle_A1_coe=Triangle_A1_coe+(rand(Float64, length(Triangle_A1_coe)).-0.5)*init_noise;
            Triangle_A2_coe=Triangle_A2_coe+(rand(Float64, length(Triangle_A2_coe)).-0.5)*init_noise;
        end


        #wrap the changed state due to initial noise 
        json_state_dict=wrap_json_state(Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe);
    end
    return json_state_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe

end


function energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state_dict,cal_chiral_order=false)

    A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
    
    bond_tensor,triangle_tensor=construct_su2_PG_IPESS(state_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);


    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;
    
    U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

    CTM=[];
    U_L=[];
    U_D=[];
    U_R=[];
    U_U=[];

    init=Dict([("CTM", []), ("init_type", "PBC")]);
    conv_check="singular_value";
    CTM_ite_info=false;
    CTM_conv_info=true;
    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);
    
    E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
    #E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_bond");
    energy=(E_up+E_down)/3;

    #return energy,CTM,U_L,U_D,U_R,U_U
    if cal_chiral_order
        chiral_order_parameters=Dict([("J1", 0), ("J2", 0), ("J3", 0), ("Jchi", 0), ("Jtrip", 1)]);
        chiral_order_up, chiral_order_down=evaluate_ob(chiral_order_parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
        return energy,chiral_order_up, chiral_order_down,ite_num,ite_err
    else
        return energy,ite_num,ite_err
    end
end



function get_vector(json_dict)
    Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(json_dict);
    if Bond_irrep=="A"
        if Triangle_irrep=="A1"
            vec=vcat(Bond_A_coe,Triangle_A1_coe);
        elseif Triangle_irrep=="A2"
            vec=vcat(Bond_A_coe,Triangle_A2_coe);
        elseif Triangle_irrep=="A1+iA2"
            vec=vcat(Bond_A_coe,Triangle_A1_coe,Triangle_A2_coe);
        end
    elseif Bond_irrep=="B"
        if Triangle_irrep=="A1"
            vec=vcat(Bond_B_coe,Triangle_A1_coe);
        elseif Triangle_irrep=="A2"
            vec=vcat(Bond_B_coe,Triangle_A2_coe);
        elseif Triangle_irrep=="A1+iA2"
            vec=vcat(Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe);
        end
    elseif Bond_irrep=="A+iB"
        if Triangle_irrep=="A1"
            vec=vcat(Bond_A_coe,Bond_B_coe,Triangle_A1_coe);
        elseif Triangle_irrep=="A2"
            vec=vcat(Bond_A_coe,Bond_B_coe,Triangle_A2_coe);
        elseif Triangle_irrep=="A1+iA2"
            vec=vcat(Bond_A_coe,Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe);
        end
    end
    return vec
end


function set_vector(json_dict, vec)
    Bond_irrep, Triangle_irrep, Bond_A_coes0, Bond_B_coes0, Triangle_A1_coes0, Triangle_A2_coes0=get_tensor_coes(json_dict);
    if Bond_irrep=="A"
        if Triangle_irrep=="A1"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]

            Triangle_A2_coe=nothing
        elseif Triangle_irrep=="A2"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            # siz=length(Triangle_A1_coes0)
            # Triangle_A1_coe=vec[1:siz]
            # vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]

            Triangle_A1_coe=nothing
        elseif Triangle_irrep=="A1+iA2"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]
        end

        Bond_B_coe=nothing;
    elseif Bond_irrep=="B"
        if Triangle_irrep=="A1"
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]

            Triangle_A2_coe=nothing
        elseif Triangle_irrep=="A2"
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            # siz=length(Triangle_A1_coes0)
            # Triangle_A1_coe=vec[1:siz]
            # vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]

            Triangle_A1_coe=nothing
        elseif Triangle_irrep=="A1+iA2"
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]
        end

        Bond_A_coe=nothing;
    elseif Bond_irrep=="A+iB"
        if Triangle_irrep=="A1"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]

            Triangle_A2_coe=nothing
        elseif Triangle_irrep=="A2"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            # siz=length(Triangle_A1_coes0)
            # Triangle_A1_coe=vec[1:siz]
            # vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]

            Triangle_A1_coe=nothing
        elseif Triangle_irrep=="A1+iA2"
            siz=length(Bond_A_coes0)
            Bond_A_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Bond_B_coes0)
            Bond_B_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A1_coes0)
            Triangle_A1_coe=vec[1:siz]
            vec=vec[siz+1:length(vec)]
            siz=length(Triangle_A2_coes0)
            Triangle_A2_coe=vec[1:siz]
        end

    end
    json_dict_new=wrap_json_state(Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe)
    #return Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe
    return json_dict_new
end




function normalize_IPESS_SU2_PG(state_dict)
    Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(state_dict)
    if Bond_irrep=="A"
        Bond_norm=norm(Bond_A_coe)
        Bond_A_coe=Bond_A_coe/Bond_norm
    elseif Bond_irrep=="B"
        Bond_norm=norm(Bond_B_coe)
        Bond_B_coe=Bond_B_coe/Bond_norm
    elseif Bond_irrep=="A+iB"
        Bond_norm=sqrt(norm(Bond_A_coe)^2+norm(Bond_B_coe)^2)
        Bond_A_coe=Bond_A_coe/Bond_norm
        Bond_B_coe=Bond_B_coe/Bond_norm
    end

    if Triangle_irrep=="A1"
        Triangle_norm=norm(Triangle_A1_coe)
        Triangle_A1_coe=Triangle_A1_coe/Triangle_norm
    elseif Triangle_irrep=="A2"
        Triangle_norm=norm(Triangle_A2_coe)
        Triangle_A2_coe=Triangle_A2_coe/Triangle_norm
    elseif Triangle_irrep=="A1+iA2"
        Triangle_norm=sqrt(norm(Triangle_A1_coe)^2+norm(Triangle_A2_coe)^2)
        Triangle_A1_coe=Triangle_A1_coe/Triangle_norm
        Triangle_A2_coe=Triangle_A2_coe/Triangle_norm
    end

    state_dict=wrap_json_state(Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe)
    return state_dict
end




function Grad_FiniteDiff(state, D, chi, parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol, dt=0.001, E0=nothing)

    state=normalize_IPESS_SU2_PG(state);
    #print(E0);flush(stdout);

    if E0==nothing
        E0,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state,false);
        E0=real(E0);
    end
    Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(state);

    println("energy E0 is "*string(E0));flush(stdout);

    Grad_FD=Dict([("Bond_A_coe", zeros(Float64, length(Bond_A_coe))), ("Bond_B_coe", zeros(Float64, length(Bond_B_coe))), ("Triangle_A1_coe", zeros(Float64, length(Triangle_A1_coe))),("Triangle_A2_coe", zeros(Float64, length(Triangle_A2_coe)))]);
    dE_data=[]
    Grad_FD_data=[]

    #println(state["coes"])

    #Bond A tensor diff
    if Bond_irrep in ["A","A+iB"]
        Bond_A_grad=zeros(Float64, length(Bond_A_coe))
        for ct =1:length(Bond_A_coe)
            Bond_A_coe_tem=deepcopy(Bond_A_coe);
            Bond_A_coe_tem[ct]=Bond_A_coe_tem[ct]+dt;
            state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, Bond_A_coe_tem, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe);
            #println(state_tem["coes"])
            E,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state_tem,false);
            E=real(E);
            Bond_A_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        #print(Bond_A_grad);flush(stdout);
        Grad_FD["Bond_A_grad"]=Bond_A_grad;
        Grad_FD_data=vcat(Grad_FD_data, Bond_A_grad);
    end

    #Bond B tensor diff
    if Bond_irrep in ["B","A+iB"]
        Bond_B_grad=zeros(Float64, length(Bond_B_coe))
        for ct=1:length(Bond_B_coe)
            Bond_B_coe_tem=deepcopy(Bond_B_coe);
            Bond_B_coe_tem[ct]=Bond_B_coe_tem[ct]+dt
            state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe_tem, Triangle_A1_coe, Triangle_A2_coe);
            #println(state_tem["coes"])
            E,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state_tem,false);
            E=real(E);
            Bond_B_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        #print(Bond_B_grad);flush(stdout);
        Grad_FD["Bond_B_grad"]=Bond_B_grad;
        Grad_FD_data=vcat(Grad_FD_data, Bond_B_grad);
    end

    #triangle A1 tensor diff
    if Triangle_irrep in ["A1","A1+iA2"]
        Triangle_A1_grad=zeros(Float64, length(Triangle_A1_coe))
        for ct=1:length(Triangle_A1_coe)
            Triangle_A1_coe_tem=deepcopy(Triangle_A1_coe);
            Triangle_A1_coe_tem[ct]=Triangle_A1_coe_tem[ct]+dt
            state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe_tem, Triangle_A2_coe);
            #println(state_tem["coes"])
            E,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state_tem,false);
            E=real(E);
            Triangle_A1_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        #print(Triangle_A1_grad);flush(stdout);
        Grad_FD["Triangle_A1_grad"]=Triangle_A1_grad;
        Grad_FD_data=vcat(Grad_FD_data, Triangle_A1_grad);
    end

    #triangle A2 tensor diff
    if Triangle_irrep in ["A2","A1+iA2"]
        Triangle_A2_grad=zeros(Float64, length(Triangle_A2_coe))
        for ct=1:length(Triangle_A2_coe)
            Triangle_A2_coe_tem=deepcopy(Triangle_A2_coe);
            Triangle_A2_coe_tem[ct]=Triangle_A2_coe_tem[ct]+dt
            state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe_tem);
            #println(state_tem["coes"])
            E,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state_tem,false);
            E=real(E);
            Triangle_A2_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        #print(Triangle_A2_grad)
        Grad_FD["Triangle_A2_grad"]=Triangle_A2_grad;
        Grad_FD_data=vcat(Grad_FD_data, Triangle_A2_grad);
    end

    # print("Energy difference is:");flush(stdout);
    # print(dE_data);flush(stdout);
    # print("Grad is:");flush(stdout);
    # print(Grad_FD_data);flush(stdout);
    # print("Normalized grad is:");flush(stdout);
    # print(Grad_FD_data/max(abs(Grad_FD_data)));flush(stdout);

    return E0,Grad_FD,Grad_FD_data
end



function grad_line_search(state, D, chi, parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol, dt, E0, grad0=None, direction0=None, alpha0=1, ls_ratio=1/3, ls_max=10,nonchiral="no")
    if nonchiral=="no"
        filenm="julia_LS_D_"*string(D)*"_chi_"*string(chi)*".json"
    elseif nonchiral=="A1_even"
        filenm="julia_LS_A1even_D_"*string(D)*"_chi_"*string(chi)*".json"
    elseif nonchiral=="A1_odd"
        filenm="julia_LS_A1odd_D_"*string(D)*"_chi_"*string(chi)*".json"
    end

    
    state=normalize_IPESS_SU2_PG(state)

    _,_,grad=Grad_FiniteDiff(state, D, chi, parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol, dt, E0)
    
    println("state: "*string(get_vector(state)));flush(stdout);
    println("grad: "*string(grad));flush(stdout);

    E0,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state,false);
    E0=real(E0);
    Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(state)

    println("E0= "*string(E0));flush(stdout);

    direction=-grad
    #print(grad0);flush(stdout);
    #print(grad);flush(stdout);
    if grad0==nothing
        direction=-grad;
    else
        norm_grad=norm(grad)
        norm_grad0=norm(grad0)
        beta=(norm_grad^2)/(norm_grad0^2)
        direction=-grad+beta*direction0;
    end
    vec0=deepcopy(get_vector(state));
    vec_tem=[];

    #line search
    improved=false
    alpha=alpha0
    println("line search");flush(stdout);
    println("E,chiral_order_up, chiral_order_down,ite_num,ite_err")
    println("conjugate gradient opt");flush(stdout);
    for ls_step=0:ls_max-1
        vec_tem=vec0+direction*alpha*(ls_ratio^ls_step);
        state_tem=set_vector(state, vec_tem)

        E,chiral_order_up, chiral_order_down,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state_tem,true);
        E=real(E);
        println(string(E)*", "*string(real(chiral_order_up))*", "*string(real(chiral_order_down))*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
        if E<E0
            improved=true
            break
        end
    end
    if improved
        state=set_vector(state, vec_tem)
        E,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state,false);
        E=real(E);
        open(filenm,"w") do f
            JSON.print(f, state)
        end
    else
        println("gradient opt");flush(stdout);
        for ls_step = 0:ls_max-1
            vec_tem=vec0-grad*alpha*(ls_ratio^ls_step)
            state_tem=set_vector(state, vec_tem)
            E,chiral_order_up, chiral_order_down,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state_tem,true);
            E=real(E);
            println(string(E)*", "*string(real(chiral_order_up))*", "*string(real(chiral_order_down))*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
            if E<E0
                improved=true
                break
            end
        end
    
            
        if improved
            state=set_vector(state, vec_tem)
            E,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state,false);
            E=real(E);
            open(filenm,"w") do f
                JSON.print(f, state)
            end
        else
            state=set_vector(state, vec0)
            E=E0
        end
    end
    improvement=E-E0
    
    open(filenm,"w") do f
        JSON.print(f, state)
    end
    return E,state,grad,direction,improvement
end



function run_FiniteDiff(parameters,D,chi,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,Bond_irrep,Triangle_irrep,init_statenm,init_noise)
    
    multi_threads=true;if Threads.nthreads()==1; multi_threads=false; end
    println("number of threads: "*string(Threads.nthreads()));flush(stdout);
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    # CTM_conv_tol=1e-6;
    # CTM_ite_nums=50;
    # CTM_trun_tol=1e-12;
   
    #init_statenm="LS_D_"*string(D)*"_chi_40.json"
    #init_statenm=nothing
    state, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=initial_state(Bond_irrep, Triangle_irrep,D,init_statenm,init_noise)

    println("optimization start");flush(stdout);
    #E0,_,_=Grad_FiniteDiff(state, cfg.ctm_args, args.chi)
    dt=0.001;
    grad0=nothing;
    direction0=nothing;
    alpha0=3;
    ls_ratio=1/3;
    ls_max=5;
    E0=nothing;
    nonchiral="no";
    for ite=1:100
        
        @time E0,state,grad,direction,improvement=grad_line_search(state,  D, chi, parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol, dt, E0, grad0, direction0, alpha0, ls_ratio, ls_max, nonchiral)
        println("grad norm: "*string(norm(grad)));flush(stdout)
        if -improvement<1e-7
            break
        end
    end

end

