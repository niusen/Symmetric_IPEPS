function _ipeps_fu_to_storage_like(x, ref)
    if isdefined(@__MODULE__, :ipeps_to_storage_like)
        return ipeps_to_storage_like(x,ref)
    end
    if (x isa TensorKit.AbstractTensorMap) && (ref isa TensorKit.AbstractTensorMap)
        y=x
        if TensorKit.scalartype(x) !== TensorKit.scalartype(ref)
            y=similar(x,TensorKit.scalartype(ref))
            copy!(y,x)
        end
        if TensorKit.storagetype(ref) <: Array
            return y
        end
        if isdefined(@__MODULE__, :Adapt)
            return Adapt.adapt(TensorKit.storagetype(ref).name.wrapper,y)
        end
    end
    return x
end

function _ipeps_stash_env_bot(env_bot)
    if isdefined(@__MODULE__, :IPESS_ENV_BOT_TEMP_CPU) && IPESS_ENV_BOT_TEMP_CPU[]
        env_bot_cpu=ipeps_to_cpu(env_bot);
        env_bot=nothing;
        if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
            ipeps_reclaim_device_memory!(aggressive=true);
        end
        return env_bot, env_bot_cpu
    end
    return env_bot, nothing
end

function _ipeps_restore_env_bot(env_bot, env_bot_cpu, ref)
    if !isnothing(env_bot_cpu)
        env_bot=ipeps_to_storage_like(env_bot_cpu,ref);
        env_bot_cpu=nothing;
        if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
            ipeps_reclaim_device_memory!(aggressive=true);
        end
    end
    return env_bot, env_bot_cpu
end

function _ipeps_stash_work_tensor(x)
    x_cpu=ipeps_to_cpu(x);
    x=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    return x, x_cpu
end

function _ipeps_restore_work_tensor(x, x_cpu, ref)
    if !isnothing(x_cpu)
        x=ipeps_to_storage_like(x_cpu,ref);
        x_cpu=nothing;
        if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
            ipeps_reclaim_device_memory!(aggressive=true);
        end
    end
    return x, x_cpu
end

function _ipeps_full_update_permute_neighbour_ind(A,ind1,ind2,total_ind)
    @assert ind1+1==ind2
    if total_ind==3
        if (ind1==1)&&(ind2==2)
            gate=_ipeps_swap_gate_like(A,1,2); @tensor B[:]:=A[1,2,-3]*gate[-1,-2,1,2]; order=(2,1,3,);
        elseif (ind1==2)&&(ind2==3)
            gate=_ipeps_swap_gate_like(A,2,3); @tensor B[:]:=A[-1,1,2]*gate[-2,-3,1,2]; order=(1,3,2,);
        else
            error("full update memory-saving permute does not cover ind1=$(ind1), ind2=$(ind2), total_ind=$(total_ind)")
        end
    elseif total_ind==4
        if (ind1==1)&&(ind2==2)
            gate=_ipeps_swap_gate_like(A,1,2); @tensor B[:]:=A[1,2,-3,-4]*gate[-1,-2,1,2]; order=(2,1,3,4,);
        elseif (ind1==2)&&(ind2==3)
            gate=_ipeps_swap_gate_like(A,2,3); @tensor B[:]:=A[-1,1,2,-4]*gate[-2,-3,1,2]; order=(1,3,2,4,);
        elseif (ind1==3)&&(ind2==4)
            gate=_ipeps_swap_gate_like(A,3,4); @tensor B[:]:=A[-1,-2,1,2]*gate[-3,-4,1,2]; order=(1,2,4,3,);
        else
            error("full update memory-saving permute does not cover ind1=$(ind1), ind2=$(ind2), total_ind=$(total_ind)")
        end
    elseif total_ind==5
        if (ind1==1)&&(ind2==2)
            gate=_ipeps_swap_gate_like(A,1,2); @tensor B[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; order=(2,1,3,4,5,);
        elseif (ind1==2)&&(ind2==3)
            gate=_ipeps_swap_gate_like(A,2,3); @tensor B[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; order=(1,3,2,4,5,);
        elseif (ind1==3)&&(ind2==4)
            gate=_ipeps_swap_gate_like(A,3,4); @tensor B[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; order=(1,2,4,3,5,);
        elseif (ind1==4)&&(ind2==5)
            gate=_ipeps_swap_gate_like(A,4,5); @tensor B[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2]; order=(1,2,3,5,4,);
        else
            error("full update memory-saving permute does not cover ind1=$(ind1), ind2=$(ind2), total_ind=$(total_ind)")
        end
    elseif total_ind==6
        if (ind1==1)&&(ind2==2)
            gate=_ipeps_swap_gate_like(A,1,2); @tensor B[:]:=A[1,2,-3,-4,-5,-6]*gate[-1,-2,1,2]; order=(2,1,3,4,5,6,);
        elseif (ind1==2)&&(ind2==3)
            gate=_ipeps_swap_gate_like(A,2,3); @tensor B[:]:=A[-1,1,2,-4,-5,-6]*gate[-2,-3,1,2]; order=(1,3,2,4,5,6,);
        elseif (ind1==3)&&(ind2==4)
            gate=_ipeps_swap_gate_like(A,3,4); @tensor B[:]:=A[-1,-2,1,2,-5,-6]*gate[-3,-4,1,2]; order=(1,2,4,3,5,6,);
        elseif (ind1==4)&&(ind2==5)
            gate=_ipeps_swap_gate_like(A,4,5); @tensor B[:]:=A[-1,-2,-3,1,2,-6]*gate[-4,-5,1,2]; order=(1,2,3,5,4,6,);
        elseif (ind1==5)&&(ind2==6)
            gate=_ipeps_swap_gate_like(A,5,6); @tensor B[:]:=A[-1,-2,-3,-4,1,2]*gate[-5,-6,1,2]; order=(1,2,3,4,6,5,);
        else
            error("full update memory-saving permute does not cover ind1=$(ind1), ind2=$(ind2), total_ind=$(total_ind)")
        end
    else
        error("full update memory-saving permute does not cover ind1=$(ind1), ind2=$(ind2), total_ind=$(total_ind)")
    end

    gate=nothing;
    A=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    C=permute(B,order);
    B=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    return C
end

function split_3Tesnsors(B1, B2, B3, T, op_LD_RD_RU)
    # """
    #          M1     R1
    #            \   /
    #             \ /....d1
    #              |                   B1 =  |M1, d1><D1, R1|=|M1, d1><|R1, D1   
    #              |D1

    #              |                T=|R2, D1><M3|
    #             / \

    #   M2\   /R2    M3\   /R3
    #      \ /....d2    \ /....d3
    #       |            |   
    #       |D2          |D3

    #       B2           B3

    # B2=|M2, d2><D2, R2|=|M2, d2><|R2, D2 
    # B3=|M3, d3><D3, R3|=|M3, d3><|R3, D3 
    # """

    @assert (length(codomain(B1))==1)&(length(domain(B1))==3)
    @assert (length(codomain(B2))==1)&(length(domain(B2))==3)
    @assert (length(codomain(B3))==1)&(length(domain(B3))==3)
    @assert (length(codomain(T))==2)&(length(domain(T))==1)

    B1=permute_neighbour_ind(B1,2,3,4);#M1, R1, d1,  D1
    uu,ss,vv=tsvd(permute(B1,(1,2,),(3,4,)));
    B1_res=uu; #M1, R1, new1
    B1_keep=ss*vv; #new1, d1,  D1
    B1_res=permute(B1_res,(1,),(2,3,));#(M1), (R1, new1)


    B2=permute_neighbour_ind(B2,3,4,4);#M2, d2, D2, R2
    B2=permute_neighbour_ind(B2,2,3,4);#M2, D2, d2, R2
    uu,ss,vv=tsvd(permute(B2,(1,2,),(3,4,)));
    B2_res=uu;#M2, D2, new2
    B2_keep=ss*vv; #new2, d2, R2
    B2_res=permute_neighbour_ind(B2_res,2,3,3);#M2, new2, D2
    B2_res=permute(B2_res,(1,),(2,3,));#(M2), (new2, D2)

    B3=B3;#M3, d3, R3, D3 
    uu,ss,vv=tsvd(permute(B3,(1,2,),(3,4,)));
    B3_keep=uu*ss; #M3, d3, new3,
    B3_res=vv;#new3, R3, D3
    B3_res=permute(B3_res,(1,),(2,3,)); #(new3), (R3, D3)

    ##################


    B1_B2_T_B3=build_triangle_from_4tensors(T,B1_keep,B2_keep,B3_keep);

    #d2',d3',d1', d2,d3,d1
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,5,6,6);#d2',d3',d1', d2,d1,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,4,5,6);#d2',d3',d1', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,2,3,6);#d2',d1',d3', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,1,2,6);#d1',d2',d3', d1,d2,d3
    Up_run=_ipeps_fu_to_storage_like(Up,op_LD_RD_RU);
    @tensor op_LD_RD_RU[:]:=Up_run[-1,1,2,3]*op_LD_RD_RU[1,2,3,4,5,6]*Up_run'[4,5,6,-2];

    @tensor B1_B2_T_B3_op[:]:=B1_B2_T_B3[-1,-2,1,-4]*op_LD_RD_RU[-3,1];# new2, new1, d123, new3
    B1_B2_T_B3_op=permute(B1_B2_T_B3_op,(1,2,),(3,4,));# (new2, new1), (d123, new3)


    return B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op
end


function partial_triangle_partial_B1(Big_triangle,env_bot, T,B1_keep,B2_keep,B3_keep)
    #B1_keep: (new1, d1),  (D1)
    #Big_triangle: (new2, new1), (d123, new3)
    # gate1=swap_gate(space(B2_keep,2),space(T,2));
    # gate1=permute(gate1,(2,1,),(3,4,));
    # gate2=swap_gate(space(B2_keep,1),space(T,2));
    # gate2=permute(gate2,(2,1,),(3,4,));
    gate3=swap_gate(space(B1_keep,2),space(B2_keep,1));
    gate3=permute(gate3,(2,1,),(3,4,));
    gate3=_ipeps_fu_to_storage_like(gate3,B1_keep);
    # gate4=swap_gate(space(B1_keep,1),space(B2_keep,1));
    # gate4=permute(gate4,(2,1,),(3,4,));


    #env_bot: new_ind,new2,new3,new1
    env_bot=_ipeps_fu_to_storage_like(env_bot,B1_keep);
    env_bot_new=permute(env_bot,(1,3,2,4,));#new_ind,new3,new2,new1
    env_bot,env_bot_cpu=_ipeps_stash_env_bot(env_bot);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B1 before env_bot_new fermi permute",
            "env_bot"=>env_bot,
            "env_bot_cpu"=>env_bot_cpu,
            "env_bot_new"=>env_bot_new; aggressive=true);
    end
    #apply gate4
    env_bot_new=_ipeps_full_update_permute_neighbour_ind(env_bot_new,3,4,4);#new_ind,new3,new1,new2
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B1 after env_bot_new fermi permute",
            "env_bot"=>env_bot,
            "env_bot_cpu"=>env_bot_cpu,
            "env_bot_new"=>env_bot_new; aggressive=true);
    end

    @tensor B2_T[:]:=B2_keep[-1,-2,1]*T[1,-3,-4];     #(new2, d2, R2),  (R2, D1, M3) => (new2, d2, D1, M3)
    B2_T=permute_neighbour_ind(B2_T,2,3,4);#(new2, D1, d2, M3)
    B2_T=permute_neighbour_ind(B2_T,1,2,4);#(D1, new2, d2, M3)
    @tensor B2_T_B3[:]:=B2_T[-1,-2,-3,1]*B3_keep[1,-4,-5];#(D1, new2, d2, M3) (M3, d3, new3) -> (D1, new2, d2, d3, new3)
    UU=unitary(fuse(space(B2_T_B3,3)*space(B2_T_B3,4)), space(B2_T_B3,3)*space(B2_T_B3,4));
    UU=_ipeps_fu_to_storage_like(UU,B2_T_B3);
    @tensor B2_T_B3[:]:=B2_T_B3[-1,-2,1,2,-4]*UU[-3,1,2];#(D1, new2, d2d3, new3)

    @tensor gate3_B2_T_B3[:]:=gate3[-1,-2,-3,1]*B2_T_B3[-4,1,-5,-7];#(new2,d1),(d1,new2)  (D1, new2, d2d3, new3) -> (new2,d1,d1,  D1, d2d3, new3)
    B2_T=nothing;
    B2_T_B3=nothing;
    gate3=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end
    if isdefined(@__MODULE__, :IPESS_MEMORY_INFO) && IPESS_MEMORY_INFO[]
        ipeps_print_tensor_memory("partial_triangle_partial_B1: gate3_B2_T_B3", gate3_B2_T_B3);
        ipeps_print_device_memory("CUDA memory before partial_triangle_partial_B1 env contraction:");
    end

    Big_triangle,Big_triangle_cpu=_ipeps_stash_work_tensor(Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B1 before env contraction, Big_triangle stashed",
            "Big_triangle"=>Big_triangle,
            "Big_triangle_cpu"=>Big_triangle_cpu,
            "env_bot_new"=>env_bot_new,
            "gate3_B2_T_B3"=>gate3_B2_T_B3; aggressive=true);
    end

    @tensor env_bot_new_gate3_B2_T_B3[:]:=env_bot_new[-1,1,-2,2]*gate3_B2_T_B3[2,-3,-4,-5,-6,1];#new_ind,new3,new1,new2     (new2,d1,d1,  D1, d2d3, new3) -> new_ind, new1,  d1,d1,  D1, d2d3, 
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    if isdefined(@__MODULE__, :IPESS_MEMORY_INFO) && IPESS_MEMORY_INFO[]
        ipeps_print_tensor_memory("partial_triangle_partial_B1: env_bot_new_gate3_B2_T_B3", env_bot_new_gate3_B2_T_B3);
        ipeps_print_device_memory("CUDA memory after partial_triangle_partial_B1 env contraction cleanup:");
    end
    env_bot_new,env_bot_new_cpu=_ipeps_stash_work_tensor(env_bot_new);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B1 before rho, env_bot_new stashed",
            "env_bot_new"=>env_bot_new,
            "env_bot_new_cpu"=>env_bot_new_cpu,
            "env_bot_new_gate3_B2_T_B3"=>env_bot_new_gate3_B2_T_B3,
            "gate3_B2_T_B3"=>gate3_B2_T_B3; aggressive=true);
    end

    #left side
    @tensor rho[:]:=env_bot_new_gate3_B2_T_B3'[1,-1,2,-2,-3,3]*env_bot_new_gate3_B2_T_B3[1,-4,2,-5,-6,3];#(new_ind,new1,  d1,d1,  D1, d2d3),     (new_ind,new1,  d1,d1,  D1, d2d3) -> (new1,  d1,  D1),     (new1,  d1,  D1)
    
    rho=permute(rho,(1,2,3,),(4,5,6,));
    @assert norm(rho-rho')/norm(rho)<1e-10;
    rho=rho/2+rho'/2;
    eu,ev=eigh(rho);
    eu=check_positive(eu);
    rho_inv=ev*my_pinv(eu)*ev';
    eu=nothing;
    ev=nothing;
    env_bot_new_gate3_B2_T_B3=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end
    

    #right side
    global Up
    Big_triangle,Big_triangle_cpu=_ipeps_restore_work_tensor(Big_triangle,Big_triangle_cpu,gate3_B2_T_B3);
    Up_run=_ipeps_fu_to_storage_like(Up,Big_triangle);
    @tensor Big_triangle[:]:=Big_triangle[-1,-2,1,-5]*Up_run'[-3,2,3,1]*UU[-4,2,3];#(new2, new1, d1, d2d3, new3)
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    @tensor gate3_B2_T_B3_Big_triangle[:]:=gate3_B2_T_B3'[-1,1,-2,-3,2,-4]*Big_triangle[-5,-6,1,2,-7];#(new2,d1,d1,  D1, d2d3, new3)     (new2, new1, d1, d2d3, new3) -> (new2, d1,  D1, new3)     (new2, new1, new3)
    gate3_B2_T_B3=nothing;
    Big_triangle=nothing;
    Up_run=nothing;
    UU=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot,env_bot_cpu=_ipeps_restore_env_bot(env_bot,env_bot_cpu,gate3_B2_T_B3_Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B1 before env_bot right contraction",
            "env_bot"=>env_bot,
            "env_bot_new"=>env_bot_new,
            "env_bot_new_cpu"=>env_bot_new_cpu,
            "gate3_B2_T_B3_Big_triangle"=>gate3_B2_T_B3_Big_triangle,
            "rho_inv"=>rho_inv; aggressive=true);
    elseif isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    if isdefined(@__MODULE__, :IPESS_MEMORY_INFO) && IPESS_MEMORY_INFO[]
        ipeps_print_tensor_memory("partial_triangle_partial_B1: env_bot before env_bot right contraction", env_bot);
        ipeps_print_tensor_memory("partial_triangle_partial_B1: gate3_B2_T_B3_Big_triangle before env_bot right contraction", gate3_B2_T_B3_Big_triangle);
        ipeps_print_device_memory("CUDA memory immediately before partial_triangle_partial_B1 env_bot right contraction:");
        flush(stdout);
    end
    #env_bot_new: new_ind,new3,new1,new2
    #env_bot: new_ind,new2,new3,new1
    @tensor env_bot_gate3_B2_T_B3_Big_triangle[:]:=env_bot[-1,2,3,1]*gate3_B2_T_B3_Big_triangle[-2,-3,-4,-5,2,1,3];#(new_ind,new2,new3,new1),  (new2, d1,  D1, new3 | new2, new1, new3) -> (new_ind,  new2, d1,  D1, new3 )
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    gate3_B2_T_B3_Big_triangle=nothing;
    env_bot=nothing;
    env_bot_new,env_bot_new_cpu=_ipeps_restore_work_tensor(env_bot_new,env_bot_new_cpu,env_bot_gate3_B2_T_B3_Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B1 before rightside contraction",
            "env_bot"=>env_bot,
            "env_bot_new"=>env_bot_new,
            "env_bot_gate3_B2_T_B3_Big_triangle"=>env_bot_gate3_B2_T_B3_Big_triangle,
            "rho_inv"=>rho_inv;
            aggressive=true);
    elseif isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    @tensor rightside[:]:=env_bot_new'[4,3,-1,2]*env_bot_gate3_B2_T_B3_Big_triangle[4,2,-2,-3,3];#(new_ind,new3,new1,new2), (new_ind,  new2, d1,  D1, new3 ) -> new1, d1, D1
    env_bot_new=nothing;
    env_bot_gate3_B2_T_B3_Big_triangle=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end

    # norm1=@tensor rho[1,2,3,4,5,6]*B1_keep'[3,1,2]*B1_keep[4,5,6];#(new1,  d1,  D1 | new1,  d1,  D1) (D1 | new1, d1)    (new1, d1 | D1) 
    # norm2=@tensor rightside[1,2,3]*B1_keep'[3,1,2]; #(new1, d1, D1)  (D1 | new1, d1) -
    # println([norm1,norm2])
    
    
    @tensor B1_updated[:]:=rho_inv[-1,-2,-3,1,2,3]*rightside[1,2,3];#(new1,  d1,  D1  |  new1,  d1,  D1)    (new1, d1, D1)  -> new1,  d1,  D1 

    B1_updated=permute(B1_updated,(1,2,),(3,));#(new1,  d1),  (D1) 
    return rho, rightside, B1_updated
end




function partial_triangle_partial_B2(Big_triangle,env_bot, T,B1_keep,B2_keep,B3_keep)  
    #B1_keep: (new1, d1), (D1)
    #B2_keep: (new2, d2), (R2)
    #Big_triangle: (new2, new1), (d123, new3)
    gate1=swap_gate(space(B2_keep,2),space(T,2));
    gate1=permute(gate1,(2,1,),(3,4,));
    gate1=_ipeps_fu_to_storage_like(gate1,T);
    gate2=swap_gate(space(B2_keep,1),space(T,2));
    gate2=permute(gate2,(2,1,),(3,4,));
    gate2=_ipeps_fu_to_storage_like(gate2,T);
    gate3=swap_gate(space(B1_keep,2),space(B2_keep,1));
    gate3=permute(gate3,(2,1,),(3,4,));
    gate3=_ipeps_fu_to_storage_like(gate3,B1_keep);
    gate4=swap_gate(space(B1_keep,1),space(B2_keep,1));
    gate4=permute(gate4,(2,1,),(3,4,));
    gate4=_ipeps_fu_to_storage_like(gate4,B1_keep);

    Big_triangle,Big_triangle_cpu=_ipeps_stash_work_tensor(Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B2 before left env, Big_triangle stashed",
            "Big_triangle"=>Big_triangle,
            "Big_triangle_cpu"=>Big_triangle_cpu,
            "env_bot"=>env_bot; aggressive=true);
    end



    @tensor gate4_gate3[:]:=gate4[-1,-2,-4,1]*gate3[1,-3,-5,-6];#new2,new1,d1,  new1,d1,new2
    @tensor gate4_gate3_B1[:]:=gate4_gate3[-1,-2,-3,1,2,-5]*B1_keep[1,2,-4];#(new2,new1,d1,  new1,d1,new2), (new1, d1),  (D1)  -> (new2,new1,d1,  D1, new2) 
    @tensor gate4_gate3_B1_gate2[:]:=gate4_gate3_B1[-1,-2,-3,1,2]*gate2[1,2,-4,-5];#(new2,new1,d1,  D1, new2)  -> (new2,new1,d1,  new2, D1)

    #env_bot: new_ind,new2,new3,new1
    env_bot=_ipeps_fu_to_storage_like(env_bot,gate4_gate3_B1_gate2);
    @tensor env_bot_gate4_gate3_B1_gate2[:]:=env_bot[-1,2,-2,1]*gate4_gate3_B1_gate2[2,1,-3,-4,-5];#(new_ind,new2,new3,new1), (new2,new1,d1,  new2, D1) ->(new_ind, new3, d1, new2, D1)
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot,env_bot_cpu=_ipeps_stash_env_bot(env_bot);
    gate2=nothing;
    gate3=nothing;
    gate4=nothing;
    gate4_gate3=nothing;
    gate4_gate3_B1=nothing;
    gate4_gate3_B1_gate2=nothing;

    @tensor gate1_T_B3[:]:=gate1[-1,-2,-5,1]*T[-6,1,2]*B3_keep[2,-3,-4];#(D1,d2,d2,D1) (R2, D1, M3), (M3, d3, new3) -> (D1,d2,d3, new3,   d2, R2)
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    gate1=nothing;

    @tensor leftside[:]:=env_bot_gate4_gate3_B1_gate2'[1,-1,2,-2,-3]*env_bot_gate4_gate3_B1_gate2[1,-4,2,-5,-6];#(new_ind, new3, d1, new2, D1) (new_ind, new3, d1, new2, D1) -> (, new3, , new2, D1) (, new3, , new2, D1)
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot_gate4_gate3_B1_gate2,env_bot_gate4_gate3_B1_gate2_cpu=_ipeps_stash_work_tensor(env_bot_gate4_gate3_B1_gate2);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B2 before rho, env contraction stashed",
            "env_bot_gate4_gate3_B1_gate2"=>env_bot_gate4_gate3_B1_gate2,
            "env_bot_gate4_gate3_B1_gate2_cpu"=>env_bot_gate4_gate3_B1_gate2_cpu,
            "leftside"=>leftside,
            "gate1_T_B3"=>gate1_T_B3; aggressive=true);
    end
    
    Uu=unitary(fuse(space(gate1_T_B3,5)*space(gate1_T_B3,6)), space(gate1_T_B3,5)*space(gate1_T_B3,6));
    Uu=_ipeps_fu_to_storage_like(Uu,gate1_T_B3);
    @tensor gate1_T_B3[:]:=gate1_T_B3[-1,-2,-3,-4,1,2]*Uu[-5,1,2];#(D1,d2,d3, new3,   d2, R2), ->(D1,d2,d3, new3,   d2R2)
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end

    @tensor double_gate1_T_B3[:]:=gate1_T_B3'[-1,1,2,-2,-3]*gate1_T_B3[-4,1,2,-5,-6];#(D1,d2,d3, new3,   d2R2),  (D1,d2,d3, new3,   d2R2) -> (D1, new3, d2R2,       D1, new3, d2R2)
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    
    @tensor rho[:]:=leftside[2,-1,1, 4,-3,3]*double_gate1_T_B3[1,2,-2, 3,4,-4];#(new3, new2, D1,| new3, new2, D1),  (D1, new3, d2R2, |    D1, new3, d2R2)-> (new2, d2R2,  new2, d2R2)
    leftside=nothing;
    double_gate1_T_B3=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    @tensor rho[:]:=rho[-1,1,-4,2]*Uu[1,-2,-3]*Uu'[-5,-6,2];#(new2, d2,R2,  new2, d2,R2)
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end

    rho=permute(rho,(1,2,3,),(4,5,6,));
    @assert norm(rho-rho')/norm(rho)<1e-10
    rho=rho/2+rho'/2;
    eu,ev=eigh(rho);
    eu=check_positive(eu);
    rho_inv=ev*my_pinv(eu)*ev';
    eu=nothing;
    ev=nothing;

    global Up
    Big_triangle,Big_triangle_cpu=_ipeps_restore_work_tensor(Big_triangle,Big_triangle_cpu,gate1_T_B3);
    U21=unitary(fuse(space(Big_triangle,1)*space(Big_triangle,2)), space(Big_triangle,1)*space(Big_triangle,2));
    U21=_ipeps_fu_to_storage_like(U21,Big_triangle);
    Up_run=_ipeps_fu_to_storage_like(Up,Big_triangle);
    @tensor Big_triangle[:]:=Big_triangle[1,2,3,-6]*U21[-1,1,2]*Up_run'[-3,-4,-5,3];#(new2new1,  d1,d2,d3, new3) 
    Up_run=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    @tensor rightside[:]:=gate1_T_B3'[-1,1,2,-2,-3]*Big_triangle[-4,-5,1,2,-6];#(D1,d2,d3, new3,   d2R2),  (new2new1,  d1,d2,d3, new3) -> (D1, new3,   d2R2 | new2new1,  d1, new3)
    rightside,rightside_cpu=_ipeps_stash_work_tensor(rightside);
    gate1_T_B3=nothing;
    Big_triangle=nothing;
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B2 before env_bot__ contraction, rightside stashed",
            "rightside"=>rightside,
            "rightside_cpu"=>rightside_cpu,
            "env_bot"=>env_bot,
            "env_bot_cpu"=>env_bot_cpu,
            "env_bot_gate4_gate3_B1_gate2"=>env_bot_gate4_gate3_B1_gate2,
            "env_bot_gate4_gate3_B1_gate2_cpu"=>env_bot_gate4_gate3_B1_gate2_cpu,
            "U21"=>U21; aggressive=true);
    end

    env_bot,env_bot_cpu=_ipeps_restore_env_bot(env_bot,env_bot_cpu,U21);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B2 before env_bot__ contraction",
            "env_bot"=>env_bot,
            "rightside"=>rightside,
            "rightside_cpu"=>rightside_cpu,
            "env_bot_gate4_gate3_B1_gate2"=>env_bot_gate4_gate3_B1_gate2,
            "env_bot_gate4_gate3_B1_gate2_cpu"=>env_bot_gate4_gate3_B1_gate2_cpu,
            "U21"=>U21; aggressive=true);
    end
    @tensor env_bot__[:]:=env_bot[-1,1,-3,2]*U21'[1,2,-2];#(new_ind,new2,new3,new1)->(new_ind,new2new1,new3)
    env_bot=nothing;
    U21=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    rightside,rightside_cpu=_ipeps_restore_work_tensor(rightside,rightside_cpu,env_bot__);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B2 before rightside env_bot__ contraction",
            "env_bot_gate4_gate3_B1_gate2"=>env_bot_gate4_gate3_B1_gate2,
            "env_bot_gate4_gate3_B1_gate2_cpu"=>env_bot_gate4_gate3_B1_gate2_cpu,
            "env_bot__"=>env_bot__,
            "rightside"=>rightside,
            "rightside_cpu"=>rightside_cpu,
            "rho_inv"=>rho_inv; aggressive=true);
    end
    rightside_perm=permute(rightside,((3,2,1,5,),(4,6,)));#d2R2,new3,D1,d1,new2new1,new3
    rightside=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot___perm=permute(env_bot__,((2,3,),(1,)));#new2new1,new3,new_ind
    env_bot__=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B2 before reordered rightside env contraction",
            "rightside_perm"=>rightside_perm,
            "env_bot___perm"=>env_bot___perm,
            "env_bot_gate4_gate3_B1_gate2"=>env_bot_gate4_gate3_B1_gate2,
            "env_bot_gate4_gate3_B1_gate2_cpu"=>env_bot_gate4_gate3_B1_gate2_cpu,
            "rho_inv"=>rho_inv; aggressive=true);
    end


    rightside_env_bot__=rightside_perm*env_bot___perm;#(d2R2,new3,D1,d1,new2new1,new3), (new2new1,new3,new_ind) -> (d2R2,new3,D1,d1,new_ind)
    rightside_perm=nothing;
    env_bot___perm=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    rightside_env_bot_perm=permute(rightside_env_bot__,((1,),(2,3,4,5,)));#d2R2, (new3,D1,d1,new_ind)
    rightside_env_bot__=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot_gate4_gate3_B1_gate2,env_bot_gate4_gate3_B1_gate2_cpu=_ipeps_restore_work_tensor(env_bot_gate4_gate3_B1_gate2,env_bot_gate4_gate3_B1_gate2_cpu,rightside_env_bot_perm);
    env_bot_gate4_gate3_B1_gate2_perm=permute(env_bot_gate4_gate3_B1_gate2,((4,),(2,5,3,1,)));#new2, (new3,D1,d1,new_ind)
    env_bot_gate4_gate3_B1_gate2=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B2 before final rightside compose",
            "rightside_env_bot_perm"=>rightside_env_bot_perm,
            "env_bot_gate4_gate3_B1_gate2_perm"=>env_bot_gate4_gate3_B1_gate2_perm,
            "env_bot_gate4_gate3_B1_gate2_cpu"=>env_bot_gate4_gate3_B1_gate2_cpu,
            "rho_inv"=>rho_inv; aggressive=true);
    end
    rightside=rightside_env_bot_perm*env_bot_gate4_gate3_B1_gate2_perm';#d2R2, new2
    rightside_env_bot_perm=nothing;
    env_bot_gate4_gate3_B1_gate2_perm=nothing;
    if isdefined(@__MODULE__, :IPESS_MEMORY_INFO) && IPESS_MEMORY_INFO[]
        ipeps_print_tensor_memory("partial_triangle_partial_B2: rightside", rightside);
        ipeps_print_device_memory("CUDA memory after partial_triangle_partial_B2 rightside contraction:");
    end
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end
    @tensor rightside[:]:=rightside[1,-3]*Uu[1,-1,-2];#d2,R2, new2
    rightside=permute(rightside,(3,1,2,));#new2,d2,R2

    @tensor B2_updated[:]:=rho_inv[-1,-2,-3,1,2,3]*rightside[1,2,3];#(new2, d2,R2,  new2, d2,R2),    (new2,d2,R2)  -> new2, d2,R2 
    B2_updated=permute(B2_updated,(1,2,),(3,));#(new2, d2),  (R2) 


    # norm1=@tensor rho[1,2,3,4,5,6]*B2_keep'[3,1,2]*B2_keep[4,5,6];#(new2, d2,R2,  new2, d2,R2)  (R2|new2, d2),     (new2, d2|R2) 
    # norm2=@tensor rightside[1,2,3]*B2_keep'[3,1,2]; #(new2,d2,R2)  (R2|new2, d2) -
    # println([norm1,norm2])

    return rho,rightside,B2_updated
end




function partial_triangle_partial_B3(Big_triangle,env_bot, T,B1_keep,B2_keep,B3_keep)
    @tensor B2_T[:]:=B2_keep[-1,-2,1]*T[1,-3,-4];     #(new2, d2, R2),  (R2, D1, M3) => (new2, d2, D1, M3)
    B2_T=permute_neighbour_ind(B2_T,2,3,4);#(new2, D1, d2, M3)
    B2_T=permute_neighbour_ind(B2_T,1,2,4);#(D1, new2, d2, M3)
    @tensor B1_B2_T[:]:=B1_keep[-1,-2,1]*B2_T[1,-3,-4,-5];#(new1, d1,  D1), (D1, new2, d2, M3) => (new1, d1, new2, d2, M3)
    B2_T=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end

    B1_B2_T=_ipeps_full_update_permute_neighbour_ind(B1_B2_T,2,3,5);# new1, new2, d1, d2, M3
    B1_B2_T=_ipeps_full_update_permute_neighbour_ind(B1_B2_T,1,2,5);# new2, new1, d1, d2, M3

    #env_bot: new_ind,new2,new3,new1
    env_bot=_ipeps_fu_to_storage_like(env_bot,B1_B2_T);
    @tensor env_bot_B1_B2_T[:]:=env_bot[-1,2,-2,1]*B1_B2_T[2,1,-3,-4,-5];#(new_ind,new2,new3,new1), (new2, new1, d1, d2, M3) -> (new_ind, new3, d1, d2, M3)
    B1_B2_T=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot,env_bot_cpu=_ipeps_stash_env_bot(env_bot);

    Big_triangle=_ipeps_fu_to_storage_like(Big_triangle,env_bot_B1_B2_T);
    Up_run=_ipeps_fu_to_storage_like(Up,Big_triangle);
    @tensor Big_triangle[:]:=Big_triangle[-1,-2,1,-6]*Up_run'[-3,-4,-5,1];#(new2,new1,  d1,d2,d3, new3) 
    Up_run=nothing;
    Big_triangle,Big_triangle_cpu=_ipeps_stash_work_tensor(Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B3 before left env, Big_triangle stashed",
            "Big_triangle"=>Big_triangle,
            "Big_triangle_cpu"=>Big_triangle_cpu,
            "env_bot_B1_B2_T"=>env_bot_B1_B2_T; aggressive=true);
    end
    
    Id=unitary(space(B3_keep,2),space(B3_keep,2));
    Id=_ipeps_fu_to_storage_like(Id,B3_keep);
    @tensor leftside[:]:=env_bot_B1_B2_T'[1,-1,2,3,-2]*env_bot_B1_B2_T[1,-3,2,3,-4];#(new_ind, new3, d1, d2, M3), (new_ind, new3, d1, d2, M3) ->(new3, M3,  new3,  M3) 
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot_B1_B2_T,env_bot_B1_B2_T_cpu=_ipeps_stash_work_tensor(env_bot_B1_B2_T);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B3 before rho, env_bot_B1_B2_T stashed",
            "env_bot_B1_B2_T"=>env_bot_B1_B2_T,
            "env_bot_B1_B2_T_cpu"=>env_bot_B1_B2_T_cpu,
            "leftside"=>leftside,
            "Id"=>Id; aggressive=true);
    end
    @tensor rho[:]:=leftside[-1,-2,-4,-5]*Id[-3,-6];#(new3, M3,d3,    new3,  M3,d3) 
    leftside=nothing;
    Id=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end


    rho=permute(rho,(1,2,3,),(4,5,6,));
    @assert (norm(rho-rho')/norm(rho))<1e-10
    rho=rho/2+rho'/2;
    eu,ev=eigh(rho);
    eu=check_positive(eu);
    rho_inv=ev*my_pinv(eu)*ev';
    eu=nothing;
    ev=nothing;

    Big_triangle,Big_triangle_cpu=_ipeps_restore_work_tensor(Big_triangle,Big_triangle_cpu,rho_inv);
    env_bot,env_bot_cpu=_ipeps_restore_env_bot(env_bot,env_bot_cpu,Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B3 before env_bot Big_triangle contraction",
            "env_bot"=>env_bot,
            "Big_triangle"=>Big_triangle,
            "env_bot_B1_B2_T"=>env_bot_B1_B2_T,
            "env_bot_B1_B2_T_cpu"=>env_bot_B1_B2_T_cpu,
            "rho_inv"=>rho_inv; aggressive=true);
    end
    @tensor env_bot_Big_triangle[:]:=env_bot[-1,2,3,1]*Big_triangle[2,1,-2,-3,-4,3];#(new_ind,new2,new3,new1), (new2,new1,  d1,d2,d3, new3) -> (new_ind,  d1,d2,d3)
    env_bot=nothing;
    Big_triangle=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot_B1_B2_T,env_bot_B1_B2_T_cpu=_ipeps_restore_work_tensor(env_bot_B1_B2_T,env_bot_B1_B2_T_cpu,env_bot_Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B3 before rightside contraction",
            "env_bot_B1_B2_T"=>env_bot_B1_B2_T,
            "env_bot_Big_triangle"=>env_bot_Big_triangle,
            "rho_inv"=>rho_inv; aggressive=true);
    end
    env_bot_Big_triangle_perm=permute(env_bot_Big_triangle,((4,),(1,2,3)));#d3, (new_ind,d1,d2)
    env_bot_Big_triangle=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot_B1_B2_T_perm=permute(env_bot_B1_B2_T,((2,5),(1,3,4)));#(new3,M3), (new_ind,d1,d2)
    env_bot_B1_B2_T=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_B3 before rightside compose",
            "env_bot_Big_triangle_perm"=>env_bot_Big_triangle_perm,
            "env_bot_B1_B2_T_perm"=>env_bot_B1_B2_T_perm,
            "rho_inv"=>rho_inv; aggressive=true);
    end
    rightside_tmp=env_bot_Big_triangle_perm*env_bot_B1_B2_T_perm';#d3, (new3,M3)
    env_bot_Big_triangle_perm=nothing;
    env_bot_B1_B2_T_perm=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    rightside=permute(rightside_tmp,((2,3),(1,)));#(new3,M3), d3
    rightside_tmp=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    
    @tensor B3_updated[:]:=rho_inv[-1,-2,-3,1,2,3]*rightside[1,2,3];#(new3, M3,d3,    new3,  M3,d3) ,    (new3, M3,d3)  -> (new3, M3,d3)
    B3_updated=permute(B3_updated,(2,3,),(1,));#(M3, d3), (new3)

    # norm1=@tensor rho[1,2,3,4,5,6]*B3_keep'[1,2,3]*B3_keep[5,6,4];#(new3, M3,d3,    new3,  M3,d3)   (new3)(M3, d3),    (M3, d3)(new3)
    # norm2=@tensor rightside[1,2,3]*B3_keep'[1,2,3]; #(new3, M3,d3)   (new3)(M3, d3)
    # println([norm1,norm2])

    return rho,rightside,B3_updated
end

function partial_triangle_partial_T(Big_triangle,env_bot, T,B1_keep,B2_keep,B3_keep)
    #T:(R2, D1), (M3)
    #B1_keep: (new1, d1), (D1)
    #B2_keep: (new2, d2), (R2)
    #B3_keep: (M3, d3), (new3)
    #Big_triangle: (new2, new1), (d123, new3)
    gate1=swap_gate(space(B2_keep,2),space(T,2));
    gate1=permute(gate1,(2,1,),(3,4,));
    gate1=_ipeps_fu_to_storage_like(gate1,T);
    gate2=swap_gate(space(B2_keep,1),space(T,2));
    gate2=permute(gate2,(2,1,),(3,4,));
    gate2=_ipeps_fu_to_storage_like(gate2,T);
    gate3=swap_gate(space(B1_keep,2),space(B2_keep,1));
    gate3=permute(gate3,(2,1,),(3,4,));
    gate3=_ipeps_fu_to_storage_like(gate3,B1_keep);
    gate4=swap_gate(space(B1_keep,1),space(B2_keep,1));
    gate4=permute(gate4,(2,1,),(3,4,));
    gate4=_ipeps_fu_to_storage_like(gate4,B1_keep);




    @tensor gate4_gate3[:]:=gate4[-1,-2,-4,1]*gate3[1,-3,-5,-6];#new2,new1,d1,  new1,d1,new2
    @tensor gate4_gate3_B1[:]:=gate4_gate3[-1,-2,-3,1,2,-5]*B1_keep[1,2,-4];#(new2,new1,d1,  new1,d1,new2), (new1, d1),  (D1)  -> (new2,new1,d1,  D1, new2) 
    @tensor gate4_gate3_B1_gate2[:]:=gate4_gate3_B1[-1,-2,-3,1,2]*gate2[1,2,-4,-5];#(new2,new1,d1,  D1, new2)  -> (new2,new1,d1,  new2, D1)
    @tensor gate1_B2[:]:=gate1[-2,-3,1,-5]*B2_keep[-1,1,-4];#   new2,D1,d2,    R2, D1
    @tensor gate4_gate3_B1_gate2_gate1_B2[:]:=gate4_gate3_B1_gate2[-1,-2,-3,1,2]*gate1_B2[1,2,-4,-5,-6];#(new2,new1,d1,  new2, D1), (new2,D1,d2,  R2, D1) ->(new2,new1,d1,      d2, R2, D1)  
    gate1=nothing;
    gate2=nothing;
    gate3=nothing;
    gate4=nothing;
    gate4_gate3=nothing;
    gate4_gate3_B1=nothing;
    gate4_gate3_B1_gate2=nothing;
    gate1_B2=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end

    env_bot=_ipeps_fu_to_storage_like(env_bot,gate4_gate3_B1_gate2_gate1_B2);
    @tensor env_bot_new[:]:=env_bot[-1,1,-2,2]*gate4_gate3_B1_gate2_gate1_B2[1,2,-3,-4,-5,-6];#(new_ind,new2,new3,new1), (new2,new1,d1,    d2, R2, D1) -> (new_ind,new3,      d1, d2, R2, D1)
    gate4_gate3_B1_gate2_gate1_B2=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot,env_bot_cpu=_ipeps_stash_env_bot(env_bot);
    Big_triangle,Big_triangle_cpu=_ipeps_stash_work_tensor(Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_T before left env, Big_triangle stashed",
            "Big_triangle"=>Big_triangle,
            "Big_triangle_cpu"=>Big_triangle_cpu,
            "env_bot_new"=>env_bot_new; aggressive=true);
    end
    @tensor leftside[:]:=env_bot_new'[1,-1,2,3,-2,-3]*env_bot_new[1,-4,2,3,-5,-6];#(new_ind,new3,   d1, d2, R2, D1),(new_ind,new3,   d1, d2, R2, D1) -> (new3, R2, D1,    new3, R2, D1)
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot_new,env_bot_new_cpu=_ipeps_stash_work_tensor(env_bot_new);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_T before rho, env_bot_new stashed",
            "env_bot_new"=>env_bot_new,
            "env_bot_new_cpu"=>env_bot_new_cpu,
            "leftside"=>leftside,
            "B3_keep"=>B3_keep; aggressive=true);
    end
    @tensor rho[:]:=leftside[1,-1,-2,2,-4,-5]*B3_keep'[1,-3,3]*B3_keep[-6,3,2];#(new3, R2, D1,    new3, R2, D1), (new3)(M3, d3)', (M3, d3)(new3) ->(R2,D1,M3,  R2,D1,M3)
    leftside=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end

    rho=permute(rho,(1,2,3,),(4,5,6,));
    @assert (norm(rho-rho')/norm(rho))<1e-10
    rho=rho/2+rho'/2;
    eu,ev=eigh(rho);
    eu=check_positive(eu);
    rho_inv=ev*my_pinv(eu)*ev';
    eu=nothing;
    ev=nothing;

    global Up
    Big_triangle,Big_triangle_cpu=_ipeps_restore_work_tensor(Big_triangle,Big_triangle_cpu,rho_inv);
    Up_run=_ipeps_fu_to_storage_like(Up,Big_triangle);
    @tensor Big_triangle[:]:=Big_triangle[-1,-2,1,-6]*Up_run'[-3,-4,-5,1];#(new2,new1,  d1,d2,d3, new3) 
    Up_run=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot,env_bot_cpu=_ipeps_restore_env_bot(env_bot,env_bot_cpu,Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_T before env_bot Big_triangle contraction",
            "env_bot"=>env_bot,
            "Big_triangle"=>Big_triangle,
            "env_bot_new"=>env_bot_new,
            "env_bot_new_cpu"=>env_bot_new_cpu,
            "rho_inv"=>rho_inv; aggressive=true);
    end
    @tensor env_bot_Big_triangle[:]:=env_bot[-1,2,3,1]*Big_triangle[2,1,-2,-3,-4,3];#(new_ind,new2,new3,new1), (new2,new1,  d1,d2,d3, new3) -> (new_ind,  d1,d2,d3)
    env_bot=nothing;
    Big_triangle=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot_new,env_bot_new_cpu=_ipeps_restore_work_tensor(env_bot_new,env_bot_new_cpu,env_bot_Big_triangle);
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_T before rightside contraction",
            "env_bot_new"=>env_bot_new,
            "env_bot_Big_triangle"=>env_bot_Big_triangle,
            "rho_inv"=>rho_inv; aggressive=true);
    end
    env_bot_Big_triangle_perm=permute(env_bot_Big_triangle,((4,),(1,2,3)));#d3, (new_ind,d1,d2)
    env_bot_Big_triangle=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    env_bot_new_perm=permute(env_bot_new,((2,5,6),(1,3,4)));#(new3,R2,D1), (new_ind,d1,d2)
    env_bot_new=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("partial_triangle_partial_T before rightside compose",
            "env_bot_new_perm"=>env_bot_new_perm,
            "env_bot_Big_triangle_perm"=>env_bot_Big_triangle_perm,
            "rho_inv"=>rho_inv; aggressive=true);
    end
    rightside_tmp=env_bot_Big_triangle_perm*env_bot_new_perm';#d3, (new3,R2,D1)
    env_bot_Big_triangle_perm=nothing;
    env_bot_new_perm=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    rightside=permute(rightside_tmp,((2,3,4),(1,)));#(new3,R2,D1), d3
    rightside_tmp=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    @tensor rightside[:]:=rightside[1,-1,-2,2]*B3_keep'[1,-3,2];# (new3, R2, D1, d3),  (new3)(M3, d3) -> R2, D1, M3






    @tensor T_updated[:]:=rho_inv[-1,-2,-3,1,2,3]*rightside[1,2,3];#(R2,D1,M3,  R2,D1,M3),    R2, D1, M3  -> R2, D1, M3
    T_updated=permute(T_updated,(1,2,),(3,));#(R2, D1), (M3)


    # norm1=@tensor rho[1,2,3,4,5,6]*T'[3,1,2]*T[4,5,6];#(R2,D1,M3,  R2,D1,M3)   (M3)(R2, D1),     (R2, D1)(M3)
    # norm2=@tensor rightside[1,2,3]*T'[3,1,2]; #(R2, D1, M3)  (M3)(R2, D1)
    # println([norm1,norm2])

    return rho,rightside,T_updated


end


function sweep_iteration(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new)
    # jldsave("test1.jld2";B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new)
    ####################################
    T1_left,T1_right,T1_new=partial_triangle_partial_B1(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    T1_left=nothing;
    T1_right=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end
    T2_left,T2_right,T2_new=partial_triangle_partial_B2(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    T2_left=nothing;
    T2_right=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end
    T3_left,T3_right,T3_new=partial_triangle_partial_B3(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    T3_left=nothing;
    T3_right=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end
    B_left,B_right,B_new=partial_triangle_partial_T(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    B_left=nothing;
    B_right=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end

    #T1_new: (new1, d1),  (D1) 
    #T2_new: (new2, d2),  (R2) 
    #T3_new: (M3, d3), (new3)
    #B_new: (R2, D1), (M3)

    #set the gauge:
    @tensor T2_B[:]:=T2_new[-1,-2,1]*B_new[1,-3,-4];#(new2, d2,    D1,M3)
    u,s,v=tsvd(permute(T2_B,(1,2,),(3,4,));trunc=truncdim(dim(space(T2_new,3))));
    T2_new=u*sqrt(s);
    B_new=sqrt(s)*v;
    B_new=permute(B_new,(1,2,),(3,));
    T2_B=nothing;
    u=nothing; s=nothing; v=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end

    @tensor B_T3[:]:=B_new[-1,-2,1]*T3_new[1,-3,-4];#(R2,D1,  d3,new3)
    u,s,v=tsvd(permute(B_T3,(1,2,),(3,4,));trunc=truncdim(dim(space(T3_new,1))));
    B_new=u*sqrt(s);
    T3_new=sqrt(s)*v;
    T3_new=permute(T3_new,(1,2,),(3,));
    B_T3=nothing;
    u=nothing; s=nothing; v=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end

    B_new=permute_neighbour_ind(B_new,1,2,3);#(D1, R2, M3)
    @tensor T1_B[:]:=T1_new[-1,-2,1]*B_new[1,-3,-4]; #(new1, d1,  R2, M3)
    u,s,v=tsvd(permute(T1_B,(1,2,),(3,4,));trunc=truncdim(dim(space(T1_new,3))));
    T1_new=u*sqrt(s);
    B_new=sqrt(s)*v;
    B_new=permute_neighbour_ind(B_new,1,2,3);
    B_new=permute(B_new,(1,2,),(3,));
    T1_B=nothing;
    u=nothing; s=nothing; v=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end


    return B_new,T1_new,T2_new,T3_new
end



function sweep_optimizations(n_sweep,B1_B2_T_B3_op,env_top,env_bot, B_new,T1_new,T2_new,T3_new)
    return sweep_optimizations(n_sweep,B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new)
end

function sweep_optimizations(n_sweep,B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new)
    
    env_bot_cpu=ipeps_to_cpu(env_bot);
    env_bot=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!(aggressive=true);
    end
    if isdefined(@__MODULE__, :ipeps_memory_checkpoint)
        ipeps_memory_checkpoint("sweep_optimizations after env_bot moved to CPU",
            "env_bot"=>env_bot,
            "env_bot_cpu"=>env_bot_cpu,
            "B1_B2_T_B3_op"=>B1_B2_T_B3_op; aggressive=true);
    end
    ov_history=zeros(n_sweep);
    for ci=1:n_sweep
        B_new,T1_new,T2_new,T3_new=sweep_iteration(B1_B2_T_B3_op,env_bot_cpu, B_new,T1_new,T2_new,T3_new);
        big_T_compressed_opt=build_triangle_from_4tensors(B_new,T1_new,T2_new,T3_new);

        env_bot_overlap=ipeps_to_storage_like(env_bot_cpu,big_T_compressed_opt);
        B1_B2_T_B3_op_overlap=ipeps_to_storage_like(B1_B2_T_B3_op,env_bot_overlap);
        ov12=get_overlap_env(env_bot_overlap,big_T_compressed_opt',B1_B2_T_B3_op_overlap);
        ov11=get_overlap_env(env_bot_overlap,B1_B2_T_B3_op_overlap',B1_B2_T_B3_op_overlap);
        ov22=get_overlap_env(env_bot_overlap,big_T_compressed_opt',big_T_compressed_opt);
        ov=ov12/sqrt(ov11*ov22);
        print(string(norm(ov))*" , ")
        ov_history[ci]=norm(ov);
        env_bot_overlap=nothing;
        B1_B2_T_B3_op_overlap=nothing;
        ov12=nothing;
        ov11=nothing;
        ov22=nothing;
        if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
            ipeps_reclaim_device_memory!();
        end
        if ((ci>4)&& (abs(ov_history[ci]/ov_history[ci-1]-1)<1e-7))|(ci==n_sweep);
            print("\n")
            big_T_compressed_opt=nothing;
            env_bot_cpu=nothing;
            if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
                ipeps_reclaim_device_memory!();
            end
            return B_new,T1_new,T2_new,T3_new,nothing, norm(ov)
        end
        big_T_compressed_opt=nothing;
        if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
            ipeps_reclaim_device_memory!();
        end
    end

    env_bot_cpu=nothing;
    if isdefined(@__MODULE__, :ipeps_reclaim_device_memory!)
        ipeps_reclaim_device_memory!();
    end
    
end
