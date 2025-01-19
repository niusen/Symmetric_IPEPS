
function pi_rotate_mps(mps_set)
    mps_set_new=deepcopy(mps_set);
    Lx=length(mps_set_new);

    cx=1;
    mps_set_new[cx]=permute(mps_set_new[cx],(2,1,));
    for cx=2:Lx-1
        mps_set_new[cx]=permute(mps_set_new[cx],(3,1,2));
    end
    cx=Lx;
    mps_set_new[cx]=permute(mps_set_new[cx],(1,2,));

    return mps_set_new[end:-1:1]
end

function pi_rotate_mpo(mpo_set)
    mpo_set_new=deepcopy(mpo_set);
    Lx=length(mpo_set_new);

    cx=1;
    mpo_set_new[cx]=permute(mpo_set_new[cx],(2,3,1,));
    for cx=2:Lx-1
        mpo_set_new[cx]=permute(mpo_set_new[cx],(3,4,1,2));
    end
    cx=Lx;
    mpo_set_new[cx]=permute(mpo_set_new[cx],(3,1,2,));

    return mpo_set_new[end:-1:1]
end

function simple_truncate_to_moddle(mpo_set, mps_set,chi)
    # Random.seed!(888);
    #without canonical form, so should be less accurate
    Lx=length(mpo_set);
    mps_origin=deepcopy(mps_set);

  
    mps_set=deepcopy(mps_set);
    trun_errs=Vector{Float64}(undef,0);

    ######################
    #from right to left
    cx=Lx;
    @tensor T_tem[:]:= mpo_set[cx][-1,1,-3]*mps_set[cx][-2,1]
    u,s,v=tsvd(permute(T_tem,(1,2,),(3,)); trunc=truncdim(chi));

    trun_err=1-dot(s,s)/dot(T_tem,T_tem);
    push!(trun_errs,trun_err);
    u=u*s;
    mps_set[cx]=permute(v,(1,2,));
    @tensor A[:]:=mpo_set[cx-1][-1,2,3,-4]*mps_set[cx-1][-2,1,2]*u[3,1,-3];
    mps_set[cx-1]=A;
    for cx=Lx-1:-1:3
        T=permute(mps_set[cx],(1,2,),(3,4,))
        u,s,v=tsvd(T; trunc=truncdim(chi));
        # println("aaa")
        # println(norm(u*s*v-T)/norm(T))

        trun_err=1-dot(s,s)/dot(T,T);
        push!(trun_errs,trun_err);
        u=u*s;
        mps_set[cx]=permute(v,(1,2,3,));

        @tensor A[:]:=mpo_set[cx-1][-1,2,3,-4]*mps_set[cx-1][-2,1,2]*u[3,1,-3];
        mps_set[cx-1]=A;
    end
    cx=2;
    u,s,v=tsvd(permute(mps_set[cx],(1,2,),(3,4,)); trunc=truncdim(chi));

    trun_err=1-dot(s,s)/dot(mps_set[cx],mps_set[cx]);
    push!(trun_errs,trun_err);
    u=u*s;
    mps_set[cx]=permute(v,(1,2,3,));
    @tensor A[:]:=mpo_set[cx-1][2,3,-4]*mps_set[cx-1][1,2]*u[3,1,-3];
    mps_set[cx-1]=A;


    ######################################
    #from left to right
    cx=1;
    u_trun,s_trun,v_trun=tsvd(permute(mps_set[cx],(2,),(1,)); trunc=truncdim(chi));

    trun_err=1-dot(s_trun,s_trun)/dot(mps_set[cx],mps_set[cx]);
    v_trun=s_trun*v_trun;
    mps_set[cx]=permute(u_trun,(2,1,));
    @tensor A[:]:=v_trun[-1,1]*mps_set[cx+1][1,-2,-3];
    mps_set[cx+1]=A;
    push!(trun_errs,trun_err);
    for cx=2:Lx-2
        T=permute(mps_set[cx],(1,3,),(2,));
        u_trun,s_trun,v_trun=tsvd(T; trunc=truncdim(chi)); 
   
        trun_err=1-dot(s_trun,s_trun)/dot(T,T);
        v_trun=s_trun*v_trun;
        mps_set[cx]=permute(u_trun,(1,3,2,));
        @tensor A[:]:=v_trun[-1,1]*mps_set[cx+1][1,-2,-3];
        mps_set[cx+1]=A;
        push!(trun_errs,trun_err);
    end
    cx=Lx-1;
    T=permute(mps_set[cx],(1,3,),(2,));
    u_trun,s_trun,v_trun=tsvd(T; trunc=truncdim(chi)); 
   
    trun_err=1-dot(s_trun,s_trun)/dot(T,T);
    v_trun=s_trun*v_trun;
    mps_set[cx]=permute(u_trun,(1,3,2,));
    @tensor A[:]:=v_trun[-1,1]*mps_set[cx+1][1,-2];
    mps_set[cx+1]=A;
    push!(trun_errs,trun_err);


    #normalization
    A=mps_set[1];
    norm_A=norm(A);
    A=A/norm_A;
    mps_set[1]=A;

    return mps_set,trun_errs,norm_A
end


function contract_whole_disk(psi_single::Matrix{TensorMap},chi::Int)
    Lx,Ly=size(psi_single);#original cluster size without adding trivial boundary

    ppy=Int(round(Ly/2));

    mpo_mps_fun=simple_truncate_to_moddle;


    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """
    

    ########################################
    #construct top and bot environment
    log_norm_coe=0;
    trun_history=Vector{Float64}(undef,0);
    #mps_bot_set=Matrix{TensorMap}(undef,Lx,Ly);
    #mps_top_set=Matrix{TensorMap}(undef,Lx,Ly);

    mps_bot=psi_single[:,1];

    for cy=2:ppy
        mpo=psi_single[:,cy];
        mps_bot,trun_errs,norm_coe=mpo_mps_fun(mpo, mps_bot,chi);
        trun_history=vcat(trun_history,trun_errs);
        log_norm_coe=log_norm_coe+log(norm_coe);

    end


    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top[cx]=permute(mps_top[cx],(2,1,3,));
        end
        return mps_top
    end

    mps_top=psi_single[:,Ly];
    mps_top=pi_rotate_mps(mps_top);

    for cy=Ly-1:-1:ppy+1
        mpo=pi_rotate_mpo(psi_single[:,cy]);
        mps_top,trun_errs,norm_coe=mpo_mps_fun(mpo, mps_top,chi);

        trun_history=vcat(trun_history,trun_errs);
        log_norm_coe=log_norm_coe+log(norm_coe);
    end
    mps_top=treat_mps_top(mps_top);

    ########################################

    @tensor vl[:]:=mps_top[1][-1,1]*mps_bot[1][-2,1];
    for cx=2:Lx-1
        @tensor vl[:]:=vl[1,2]*mps_top[cx][1,-1,3]*mps_bot[cx][2,-2,3];
    end
    Norm=@tensor vl[2,3]*mps_top[Lx][2,1]*mps_bot[Lx][3,1];

    norm_coe=exp(log_norm_coe);



    return norm_coe*Norm,trun_history
end

# function contract_whole_disk(psi_single::Matrix{TensorMap},chi::Int)
#     Lx,Ly=size(psi_single);#original cluster size without adding trivial boundary

#     ppy=Int(round(Ly/2));

#     mpo_mps_fun=simple_truncate_to_moddle;


#     """coordinate
#         (1,2),(2,2)
#         (1,1),(2,1)
#     """
    

#     ########################################
#     #construct top and bot environment

#     trun_history=Vector{Float64}(undef,0);
#     mps_bot_set=Matrix{TensorMap}(undef,Lx,Ly);
#     mps_top_set=Matrix{TensorMap}(undef,Lx,Ly);

#     mps_bot=psi_single[:,1];
#     ############
#     mps_bot_set[:,1]=mps_bot;
#     ############
#     for cy=2:ppy
#         mpo=psi_single[:,cy];
#         mps_bot,trun_errs,norm_coe=mpo_mps_fun(mpo, mps_bot,chi);
#         mps_bot[1]=mps_bot[1]*norm_coe;
#         trun_history=vcat(trun_history,trun_errs);
#         ############
#         mps_bot_set[:,cy]=mps_bot;
#         ############
#     end


#     function treat_mps_top(mps_top)
#         #convert mps_top to normal order
#         mps_top=mps_top[end:-1:1];
#         for cx=2:Lx-1
#             mps_top[cx]=permute(mps_top[cx],(2,1,3,));
#         end
#         return mps_top
#     end

#     mps_top=psi_single[:,Ly];
#     mps_top=pi_rotate_mps(mps_top);
#     ############
#     mps_top_set[:,Ly]=mps_top;
#     ############
#     for cy=Ly-1:-1:ppy+1
#         mpo=pi_rotate_mpo(psi_single[:,cy]);
#         mps_top,trun_errs,norm_coe=mpo_mps_fun(mpo, mps_top,chi);
#         mps_top[1]=mps_top[1]*norm_coe;
#         trun_history=vcat(trun_history,trun_errs);
#         ############
#         mps_top_set[:,cy]=mps_top;
#         ############
#     end
#     mps_top=treat_mps_top(mps_top);

#     ########################################

#     @tensor vl[:]:=mps_top[1][-1,1]*mps_bot[1][-2,1];
#     for cx=2:Lx-1
#         @tensor vl[:]:=vl[1,2]*mps_top[cx][1,-1,3]*mps_bot[cx][2,-2,3];
#     end
#     Norm=@tensor vl[2,3]*mps_top[Lx][2,1]*mps_bot[Lx][3,1];


#     return Norm,trun_history, mps_top_set,mps_bot_set
# end



function contract_partial_disk(psi_single::Matrix{TensorMap},config_new::Vector{Int8},contract_history_::disk_contract_history, chi::Int)
    contract_history_=deepcopy(contract_history_);#warning: this deepcopy is necessary, otherwise may cause error is sweep is not accepted.
    Lx,Ly=size(psi_single);#original cluster size without adding trivial boundary
    config_new_=reshape(config_new,Lx,Ly);
    config_old_=reshape(contract_history_.config,Lx,Ly);
    ppy=Int(round(Ly/2));

    mpo_mps_fun=simple_truncate_to_moddle;

    #compare old and new config
    y_bot0=0;
    for cy=1:ppy
        if config_new_[:,cy]==config_old_[:,cy]
            y_bot0=y_bot0+1;
        else
            break;
        end
    end

    y_top0=Ly+1;
    for cy=Ly:-1:ppy+1
        if config_new_[:,cy]==config_old_[:,cy]
            y_top0=y_top0-1;
        else
            break;
        end
    end
    # @show y_bot0,y_top0
    

    ########################################
    #construct top and bot environment

    trun_history=Vector{Float64}(undef,0);
    mps_bot_set=contract_history_.mps_bot_set;
    mps_top_set=contract_history_.mps_top_set;

    if y_bot0==0
        mps_bot=psi_single[:,1];
        mps_bot_set[:,1]=mps_bot;
        y0=1;
    elseif y_bot0>0
        mps_bot=mps_bot_set[:,y_bot0];
        y0=y_bot0;
    end

    for cy=y0+1:ppy
        mpo=psi_single[:,cy];
        mps_bot,trun_errs,norm_coe=mpo_mps_fun(mpo, mps_bot,chi);
        mps_bot[1]=mps_bot[1]*norm_coe;
        mps_bot_set[:,cy]=mps_bot;
        trun_history=vcat(trun_history,trun_errs);
    end

    #######################
    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top[cx]=permute(mps_top[cx],(2,1,3,));
        end
        return mps_top
    end

    if y_top0==Ly+1
        mps_top=psi_single[:,Ly];
        mps_top=pi_rotate_mps(mps_top);
        mps_top_set[:,Ly]=mps_top;
        y1=Ly;
    elseif y_top0<Ly+1
        mps_top=mps_top_set[:,y_top0];
        y1=y_top0;
    end

    
    
    for cy=y1-1:-1:ppy+1
        mpo=pi_rotate_mpo(psi_single[:,cy]);
        mps_top,trun_errs,norm_coe=mpo_mps_fun(mpo, mps_top,chi);
        mps_top[1]=mps_top[1]*norm_coe;
        mps_top_set[:,cy]=mps_top;
        trun_history=vcat(trun_history,trun_errs);
    end
    mps_top=treat_mps_top(mps_top);

    ########################################

    @tensor vl[:]:=mps_top[1][-1,1]*mps_bot[1][-2,1];
    for cx=2:Lx-1
        @tensor vl[:]:=vl[1,2]*mps_top[cx][1,-1,3]*mps_bot[cx][2,-2,3];
    end
    Norm=@tensor vl[2,3]*mps_top[Lx][2,1]*mps_bot[Lx][3,1];





    return Norm,trun_history, disk_contract_history(config_new, mps_top_set,mps_bot_set)
end

function verify_contract_history(psi_single::Matrix{TensorMap},contract_history_::disk_contract_history, chi::Int)

    Lx,Ly=size(psi_single);#original cluster size without adding trivial boundary

    ppy=Int(round(Ly/2));

    mpo_mps_fun=simple_truncate_to_moddle;

    ########################################
    #construct top and bot environment

    mps_bot_set=Matrix{TensorMap}(undef,Lx,Ly);
    mps_top_set=Matrix{TensorMap}(undef,Lx,Ly);

    mps_bot=psi_single[:,1];
    mps_bot_set[:,1]=mps_bot;

    for cy=2:ppy
        mpo=psi_single[:,cy];
        mps_bot,trun_errs,norm_coe=mpo_mps_fun(mpo, mps_bot,chi);
        mps_bot[1]=mps_bot[1]*norm_coe;
        mps_bot_set[:,cy]=mps_bot;
    end

    #######################
    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top[cx]=permute(mps_top[cx],(2,1,3,));
        end
        return mps_top
    end


    mps_top=psi_single[:,Ly];
    mps_top=pi_rotate_mps(mps_top);
    mps_top_set[:,Ly]=mps_top;
    
    for cy=Ly-1:-1:ppy+1
        mpo=pi_rotate_mpo(psi_single[:,cy]);
        mps_top,trun_errs,norm_coe=mpo_mps_fun(mpo, mps_top,chi);
        mps_top[1]=mps_top[1]*norm_coe;
        mps_top_set[:,cy]=mps_top;
    end
    # mps_top=treat_mps_top(mps_top);

    ########################################

    mps_top_set_old=contract_history_.mps_top_set;
    mps_bot_set_old=contract_history_.mps_bot_set;

    for cx=1:Lx
        for cy=ppy+1:Ly
            @assert norm(mps_top_set[cx,cy]-mps_top_set_old[cx,cy])/norm(mps_top_set[cx,cy])<1e-10  string([cx,cy])
        end
    end

    for cx=1:Lx
        for cy=1:ppy
            @assert norm(mps_bot_set[cx,cy]-mps_bot_set_old[cx,cy])/norm(mps_bot_set[cx,cy])<1e-10  string([cx,cy])
        end
    end

    
end

########################################################
function exact_contraction(psi_single)
    Lx,Ly=size(psi_single);#original cluster size without adding trivial boundary

    ppy=Int(round(Ly/2));

    mps_top=psi_single[:,Ly];
    for cy=Ly-1:-1:ppy+1
        for cx=1:Lx
            if cx==1
                @tensor T[:]:=mps_top[cx][1,-2]*psi_single[cx,cy][-1,-3,1];
            elseif 1<cx<Lx
                @tensor T[:]:=mps_top[cx][-1,1,-4]*psi_single[cx,cy][-2,-3,-5,1];
            elseif cx==Lx
                @tensor T[:]:=mps_top[cx][-1,1]*psi_single[cx,cy][-2,-3,1];
            end
            mps_top[cx]=T;
        end
        U_set=Vector{TensorMap}(undef,Lx-1);
        for cx=1:Lx-1
            if cx==1
                U_set[cx]=unitary(fuse(space(mps_top[cx],2)*space(mps_top[cx],3)), space(mps_top[cx],2)*space(mps_top[cx],3));
            elseif cx>1
                U_set[cx]=unitary(fuse(space(mps_top[cx],4)*space(mps_top[cx],5)), space(mps_top[cx],4)*space(mps_top[cx],5));
            end
        end
        for cx=1:Lx
            if cx==1
                @tensor T[:]:=mps_top[cx][-1,1,2]*U_set[cx][-2,1,2];
            elseif 1<cx<Lx
                @tensor T[:]:=mps_top[cx][1,2,-2,3,4]*U_set[cx-1]'[1,2,-1]*U_set[cx][-3,3,4];
            elseif cx==Lx
                @tensor T[:]:=mps_top[cx][1,2,-2]*U_set[cx-1]'[1,2,-1];
            end
            mps_top[cx]=T;
        end
    end

    mps_bot=psi_single[:,1];
    for cy=2:ppy
        for cx=1:Lx
            if cx==1
                @tensor T[:]:=mps_bot[cx][-2,1]*psi_single[cx,cy][1,-1,-3];
            elseif 1<cx<Lx
                @tensor T[:]:=mps_bot[cx][-2,-4,1]*psi_single[cx,cy][-1,1,-3,-5];
            elseif cx==Lx
                @tensor T[:]:=mps_bot[cx][-2,1]*psi_single[cx,cy][-1,1,-3];
            end
            mps_bot[cx]=T;
        end
        U_set=Vector{TensorMap}(undef,Lx-1);
        for cx=1:Lx-1
            if cx==1
                U_set[cx]=unitary(fuse(space(mps_bot[cx],1)*space(mps_bot[cx],2)), space(mps_bot[cx],1)*space(mps_bot[cx],2));
            elseif cx>1
                U_set[cx]=unitary(fuse(space(mps_bot[cx],3)*space(mps_bot[cx],4)), space(mps_bot[cx],3)*space(mps_bot[cx],4));
            end
        end
        for cx=1:Lx
            if cx==1
                @tensor T[:]:=mps_bot[cx][1,2,-2]*U_set[cx][-1,1,2];
            elseif 1<cx<Lx
                @tensor T[:]:=mps_bot[cx][1,2,3,4,-3]*U_set[cx-1]'[1,2,-1]*U_set[cx][-2,3,4];
            elseif cx==Lx
                @tensor T[:]:=mps_bot[cx][1,2,-2]*U_set[cx-1]'[1,2,-1];
            end
            mps_bot[cx]=T;
        end
    end

    @tensor vl[:]:=mps_top[1][1,-1]*mps_bot[1][-2,1];
    for cx=2:Lx-1
        @tensor vl[:]:=vl[1,2]*mps_top[cx][1,3,-1]*mps_bot[cx][2,-2,3];
    end
    Norm=@tensor vl[1,2]*mps_top[Lx][1,3]*mps_bot[Lx][2,3];

    return Norm
end