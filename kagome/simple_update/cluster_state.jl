function built_cluster(A_fused,Size)
    if Size=="2x2"
        @tensor psi[:]:=A_fused[2,5,1,6,-1]*A_fused[1,7,2,8,-2]*A_fused[4,6,3,5,-3]*A_fused[3,8,4,7,-4];

    elseif Size=="2x3"
        U_phy=unitary(fuse(space(A_fused,5)*space(A_fused,5)),  space(A_fused,5)*space(A_fused,5));
        U_v=unitary(fuse(space(A_fused,3)*space(A_fused,3)),  space(A_fused,3)*space(A_fused,3));
        @tensor AA[:]:=A_fused[3,1,5,2,7]*A_fused[4,2,6,1,8]*U_phy[-3,7,8]*U_v[-2,5,6]*U_v'[3,4,-1];

        @tensor psi[:]:=AA[3,1,-1]*AA[1,2,-2]*AA[2,3,-3];
    elseif Size=="2x4"
        U_phy=unitary(fuse(space(A_fused,5)*space(A_fused,5)),  space(A_fused,5)*space(A_fused,5));
        U_v=unitary(fuse(space(A_fused,3)*space(A_fused,3)),  space(A_fused,3)*space(A_fused,3));
        @tensor AA[:]:=A_fused[3,1,5,2,7]*A_fused[4,2,6,1,8]*U_phy[-3,7,8]*U_v[-2,5,6]*U_v'[3,4,-1];

        @tensor psi[:]:=AA[4,1,-1]*AA[1,2,-2]*AA[2,3,-3]*AA[3,4,-4];
    elseif Size=="2x5"
        U_phy=unitary(fuse(space(A_fused,5)*space(A_fused,5)),  space(A_fused,5)*space(A_fused,5));
        U_v=unitary(fuse(space(A_fused,3)*space(A_fused,3)),  space(A_fused,3)*space(A_fused,3));
        @tensor AA[:]:=A_fused[3,1,5,2,7]*A_fused[4,2,6,1,8]*U_phy[-3,7,8]*U_v[-2,5,6]*U_v'[3,4,-1];

        @tensor psi[:]:=AA[5,1,-1]*AA[1,2,-2]*AA[2,3,-3]*AA[3,4,-4]*AA[4,5,-5];
    elseif Size=="2x6"
        U_phy=unitary(fuse(space(A_fused,5)*space(A_fused,5)),  space(A_fused,5)*space(A_fused,5));
        U_v=unitary(fuse(space(A_fused,3)*space(A_fused,3)),  space(A_fused,3)*space(A_fused,3));
        @tensor AA[:]:=A_fused[3,1,5,2,7]*A_fused[4,2,6,1,8]*U_phy[-3,7,8]*U_v[-2,5,6]*U_v'[3,4,-1];

        @tensor psi[:]:=AA[6,1,-1]*AA[1,2,-2]*AA[2,3,-3]*AA[3,4,-4]*AA[4,5,-5]*AA[5,6,-6];
    elseif Size=="3x3"
        U_phy=unitary(fuse(space(A_fused,5)*space(A_fused,5)),  space(A_fused,5)*space(A_fused,5));
        U_v=unitary(fuse(space(A_fused,3)*space(A_fused,3)),  space(A_fused,3)*space(A_fused,3));
        @tensor AA[:]:=A_fused[2,1,4,-4,6]*A_fused[3,-2,5,1,7]*U_phy[-5,6,7]*U_v[-3,4,5]*U_v'[2,3,-1];

        U_phy2=unitary(fuse(space(AA,5)*space(A_fused,5)),  space(AA,5)*space(A_fused,5));
        U_v2=unitary(fuse(space(AA,3)*space(A_fused,3)),  space(AA,3)*space(A_fused,3));
        @tensor AAA[:]:=AA[3,1,5,2,7]*A_fused[4,2,6,1,8]*U_phy2[-3,7,8]*U_v2[-2,5,6]*U_v2'[3,4,-1];

        @tensor psi[:]:=AAA[3,1,-1]*AAA[1,2,-2]*AAA[2,3,-3];

    end

    return psi
end

function overlap_(psi,psi1)
    ov=norm(psi1'*psi)/sqrt(norm(psi'*psi)*norm(psi1'*psi1));
    return ov
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
function normalize_IPESS_SU2_PG(state_dict)
    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(state_dict)
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

    state_dict=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe)
    return state_dict
end


# function initial_symmetric_state(D,init_statenm,init_noise,Bond_irrep,Triangle_irrep,nonchiral)


    
    
#     state_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, _, _=initial_state(Bond_irrep,Triangle_irrep,nonchiral,D,init_statenm,init_noise);
    
    
    
#     A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
        
#     bond_tensor,triangle_tensor=construct_su2_PG_IPESS(state_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);
    
    
#     PEPS_tensor=bond_tensor;
#     @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
#     A_unfused=PEPS_tensor;
    
#     U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
#     @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];
    
#     psi=built_cluster(A_fused,Size);
    
    
#     return psi, A_fused,  Bond_A_coe, Triangle_A1_coe, Triangle_A2_coe

# end

function cost_fun(psi,D,state_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb)

   
    bond_tensor,triangle_tensor=construct_su2_PG_IPESS(state_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;
    
    U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];
    
    psi1=built_cluster(A_fused,Size);
    ov=overlap_(psi,psi1);
    return ov

end




function Grad_FiniteDiff(state, nonchiral, A1_has_odd, A2_has_odd, D, dt=0.001, ov0=nothing)

    state=normalize_IPESS_SU2_PG(state);

    
    ov0=cost_fun(psi,D,state,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);
    
    
    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(state);

    println("overlap ov0 is "*string(ov0));flush(stdout);

    Grad_FD=Dict([("Bond_A_coe", zeros(Float64, length(Bond_A_coe))), ("Bond_B_coe", zeros(Float64, length(Bond_B_coe))), ("Triangle_A1_coe", zeros(Float64, length(Triangle_A1_coe))),("Triangle_A2_coe", zeros(Float64, length(Triangle_A2_coe)))]);
    dE_data=[]
    Grad_FD_data=[]

    #println(state["coes"]["Bond_A_coe"])

    #Bond A tensor diff
    if Bond_irrep in ["A","A+iB"]
        Bond_A_grad=zeros(Float64, length(Bond_A_coe))
        for ct =1:length(Bond_A_coe)
            Bond_A_coe_tem=deepcopy(Bond_A_coe);
            Bond_A_coe_tem[ct]=Bond_A_coe_tem[ct]+dt;
            
            state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe_tem, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe);
            #println(state_tem["coes"]["Bond_A_coe"])
            ov=cost_fun(psi,D,state_tem,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);


            Bond_A_grad[ct]=(ov-ov0)/dt;
            dE_data=vcat(dE_data, ov-ov0);
            #println("overlap is "*string(ov));flush(stdout);
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
            state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe_tem, Triangle_A1_coe, Triangle_A2_coe);
            #println(state_tem["coes"])

            ov=cost_fun(psi,D,state_tem,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);


            Bond_B_grad[ct]=(ov-ov0)/dt;
            dE_data=vcat(dE_data, ov-ov0);
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
            if (nonchiral=="No") | ((nonchiral=="A1_even")&(A1_has_odd[ct]==0)) | ((nonchiral=="A1_odd")&(A1_has_odd[ct]==1))
                Triangle_A1_coe_tem=deepcopy(Triangle_A1_coe);
                Triangle_A1_coe_tem[ct]=Triangle_A1_coe_tem[ct]+dt
                state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe_tem, Triangle_A2_coe);
                #println(state_tem["coes"])

                ov=cost_fun(psi,D,state_tem,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

                Triangle_A1_grad[ct]=(ov-ov0)/dt;
                dE_data=vcat(dE_data, ov-ov0);
                #println("energy is "*string(E));flush(stdout);
            elseif ((nonchiral=="A1_even")&(A1_has_odd[ct]==1)) | ((nonchiral=="A1_odd")&(A1_has_odd[ct]==0))
                Triangle_A1_grad[ct]=0;
                dE_data=vcat(dE_data, 0);
            else
                error("incorrect type 'nonchiral'")
            end
        end
        #print(Triangle_A1_grad);flush(stdout);
        Grad_FD["Triangle_A1_grad"]=Triangle_A1_grad;
        Grad_FD_data=vcat(Grad_FD_data, Triangle_A1_grad);
    end

    #triangle A2 tensor diff
    if Triangle_irrep in ["A2","A1+iA2"]
        Triangle_A2_grad=zeros(Float64, length(Triangle_A2_coe))
        for ct=1:length(Triangle_A2_coe)
            if (nonchiral=="No") | ((nonchiral=="A1_even")&(A2_has_odd[ct]==1)) | ((nonchiral=="A1_odd")&(A2_has_odd[ct]==0))
                Triangle_A2_coe_tem=deepcopy(Triangle_A2_coe);
                Triangle_A2_coe_tem[ct]=Triangle_A2_coe_tem[ct]+dt
                state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe_tem);
                #println(state_tem["coes"])

                ov=cost_fun(psi,D,state_tem,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);


                Triangle_A2_grad[ct]=(ov-ov0)/dt;
                dE_data=vcat(dE_data, ov-ov0);
                #println("energy is "*string(E));flush(stdout);
            elseif ((nonchiral=="A1_even")&(A2_has_odd[ct]==0)) | ((nonchiral=="A1_odd")&(A2_has_odd[ct]==1))
                Triangle_A2_grad[ct]=0;
                dE_data=vcat(dE_data, 0);
            else
                error("incorrect type 'nonchiral'")
            end
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

    return ov0,Grad_FD,Grad_FD_data
end



function grad_line_search(state, nonchiral, A1_has_odd, A2_has_odd, D, dt, ov0, grad0=None, direction0=None, alpha0=1, ls_ratio=1/3, ls_max=10)
    
    if nonchiral=="No"
        filenm="projection_LS_D_"*string(D)*".json"
    elseif nonchiral=="A1_even"
        filenm="projection_LS_A1even_D_"*string(D)*".json"
    elseif nonchiral=="A1_odd"
        filenm="projection_LS_A1odd_D_"*string(D)*".json"
    end
    
    state=normalize_IPESS_SU2_PG(state)
    
    ov0,_,grad=Grad_FiniteDiff(state, nonchiral, A1_has_odd, A2_has_odd, D, dt, ov0)
    grad=-grad;
    println("state: "*string(get_vector(state)));flush(stdout);
    println("grad: "*string(grad));flush(stdout);


    ov=ov0;
    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(state)

    println("ov0= "*string(ov0));flush(stdout);

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
    println("overlap")
    println("conjugate gradient opt");flush(stdout);
    for ls_step=0:ls_max-1
        vec_tem=vec0+direction*alpha*(ls_ratio^ls_step);
        state_tem=set_vector(state, vec_tem)
        ov=cost_fun(psi,D,state_tem,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

 
        println(string(ov));flush(stdout);
        if ov>ov0
            improved=true
            break
        end
    end
    if improved
        state=set_vector(state, vec_tem)
        open(filenm,"w") do f
            JSON.print(f, state)
        end
    else
        println("gradient opt");flush(stdout);
        for ls_step = 0:ls_max-1
            vec_tem=vec0-grad*alpha*(ls_ratio^ls_step)
            state_tem=set_vector(state, vec_tem)
            ov=cost_fun(psi,D,state_tem,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

            println(string(ov));flush(stdout);
            if ov>ov0
                improved=true
                break
            end
        end
    
            
        if improved
            state=set_vector(state, vec_tem)
            open(filenm,"w") do f
                JSON.print(f, state)
            end
        else
            state=set_vector(state, vec0)
            ov=ov0
        end
    end
    improvement=ov-ov0
    
    open(filenm,"w") do f
        JSON.print(f, state)
    end
    return ov,state,grad,direction,improvement
end



function run_FiniteDiff(psi, D,Bond_irrep,Triangle_irrep,nonchiral,init_statenm,init_noise)
    
    println("D="*string(D));flush(stdout);
    println("Bond_irrep: "*Bond_irrep);flush(stdout);
    println("nonchiral: "*nonchiral);flush(stdout);



    #init_statenm="LS_D_"*string(D)*"_chi_40.json"
    #init_statenm=nothing
    state, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd=initial_state(Bond_irrep, Triangle_irrep, nonchiral, D,init_statenm,init_noise)


    println("optimization start");flush(stdout);

    dt=0.01;

    grad=nothing;
    direction=nothing;
    alpha0=3;
    ls_ratio=1/3;
    ls_max=10;
    ov0=nothing;
    nonchiral=nonchiral;
    for ite=1:100
        
        @time ov0,state,grad,direction,improvement=grad_line_search(state, nonchiral,A1_has_odd, A2_has_odd, D, dt, ov0, grad, direction, alpha0, ls_ratio, ls_max)
        println("grad norm: "*string(norm(grad)));flush(stdout)
        if (improvement<1e-7)
            break
        end
    end

end




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
    global A_set,B_set,A1_set,A2_set,A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, virtual_particle
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