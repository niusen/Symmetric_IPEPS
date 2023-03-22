function read_json_state(filenm)
    json_dict = Dict()
    open(filenm, "r") do f
        json_dict
        dicttxt = read(f,String)  # file information to string
        json_dict=JSON.parse(dicttxt)  # parse and transform data
    end
    return json_dict
end


function wrap_json_state(Bond_irrep,Triangle_irrep,nonchiral,Bond_A_coe,Bond_B_coe,Triangle_A1_coe,Triangle_A2_coe)
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
    json_state=Dict([("coes" , coes), ("Bond_irrep", Bond_irrep), ("Triangle_irrep", Triangle_irrep), ("nonchiral", nonchiral)]);
    
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

function initial_state(Bond_irrep,Triangle_irrep,nonchiral,D,init_statenm=nothing,init_noise=0)
    A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, _, _, virtual_particle, _, _=construct_tensor(D);
    if init_statenm==nothing 
        println("Random initial state");flush(stdout);
        
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
        
        #projection to ninchiral state if needed
        Triangle_A1_coe,Triangle_A2_coe, A1_has_odd, A2_has_odd=nonchiral_projection(nonchiral,Triangle_A1_coe,Triangle_A2_coe,A1_set_occu,A2_set_occu,virtual_particle);

        json_state_dict=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe)
    else
        
        println("load state: "*init_statenm);flush(stdout);
        json_state_dict=read_json_state(init_statenm);
        Bond_irrep_, Triangle_irrep_, nonchiral_, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(json_state_dict);#projection to nonchiral is inside this function if needed 
        @assert Bond_irrep_==Bond_irrep
        @assert Triangle_irrep_==Triangle_irrep
        if nonchiral_==nonchiral
        else
            if nonchiral!="No"
                println("Initial state is not nonchiral. Now project it to nonchiral.");flush(stdout);
            end
        end

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

        #projection to ninchiral state if needed
        Triangle_A1_coe,Triangle_A2_coe, A1_has_odd, A2_has_odd=nonchiral_projection(nonchiral,Triangle_A1_coe,Triangle_A2_coe,A1_set_occu,A2_set_occu,virtual_particle);

        #wrap the changed state due to initial noise 
        json_state_dict=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe);
    end
    return json_state_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd

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
    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(json_dict);
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
    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coes0, Bond_B_coes0, Triangle_A1_coes0, Triangle_A2_coes0=get_tensor_coes(json_dict);
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
    json_dict_new=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe)
    #return Bond_irrep, Triangle_irrep, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe
    return json_dict_new
end


function parameter_linear_change(stateL, stateR, D, chi, parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol, Np)
    state=deepcopy(stateL);
    vecL=deepcopy(get_vector(stateL));
    vecR=deepcopy(get_vector(stateR));


    states=Matrix(undef,Np+1,length(vecL));
    Es=Vector(undef,Np+1);
    chiral_orders=Vector(undef,Np+1);
    println("linear change parameter");
    for ls_step=0:Np
        C1=ls_step/Np;
        C2=1-C1;
        state_tem=set_vector(state, vecL*C1+vecR*C2);
        println("state: "*string(get_vector(state_tem)));flush(stdout);
        
        println("E,chiral_order_up, chiral_order_down,ite_num,ite_err")

        

        E,chiral_order_up, chiral_order_down,ite_num,ite_err=energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state_tem,true);
        
        println(string(E)*", "*string(real(chiral_order_up))*", "*string(real(chiral_order_down))*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
        Es[ls_step+1]=real(E);
        chiral_orders[ls_step+1]=chiral_order_up;
        states[ls_step+1,1:length(vecL)]=vecL*C1+vecR*C2;

    end


    filenm="linear_parameter_change"*"_D"*string(D)*"_chi"*string(chi)*".mat";
    matwrite(filenm, Dict(
        "Es" => Es,
        "chiral_orders" => chiral_orders,
        "states" => states
    ); compress = false)
end



function run_parameter_change(parameters,D,chi,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,Bond_irrep,Triangle_irrep,nonchiral,init_statenmL,init_statenmR,Np)
    stateL, _,_, _,_,_,_=initial_state(Bond_irrep, Triangle_irrep, nonchiral, D,init_statenmL,0);
    stateR, _,_, _,_,_,_=initial_state(Bond_irrep, Triangle_irrep, nonchiral, D,init_statenmR,0);

    multi_threads=true;if Threads.nthreads()==1; multi_threads=false; end
    println("number of threads: "*string(Threads.nthreads()));flush(stdout);
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    println("Bond_irrep: "*Bond_irrep);flush(stdout);
    println("nonchiral: "*nonchiral);flush(stdout);


    parameter_linear_change(stateL, stateR, D, chi, parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol, Np)

end

