
function impo_imps(mpo,A)
    unitary_mpo_A=unitary(fuse(space(mpo,1)⊗space(A,1)), space(mpo,1)⊗space(A,1));
    @tensor mpo_A[:]:=unitary_mpo_A[-1,2,1]*mpo[2,3,4,-3]*A[1,5,3]*unitary_mpo_A'[4,5,-2];
    return mpo_A
end

function HV_L_tensor(vl,A,mpo)
    if mpo==[]
        if numind(vl)==3
            @tensor vl[:]:=vl[-1,3,1]*A'[3,-2,2]*A[1,-3,2];
        elseif numind(vl)==2
            @tensor vl[:]:=vl[3,1]*A'[3,-2,2]*A[1,-3,2];
        end
    else
        if numind(vl)==5
            @tensor vl[:]:=vl[-1,7,5,3,1]*A'[7,-2,6]*mpo'[5,6,-3,4]*mpo[3,2,-4,4]*A[1,-5,2];
        elseif numind(vl)==4
            @tensor vl[:]:=vl[7,5,3,1]*A'[7,-1,6]*mpo'[5,6,-2,4]*mpo[3,2,-3,4]*A[1,-4,2];
        end
    end
    return vl
end

function HV_R_tensor(vr,A,mpo)
    if mpo==[]
        if numind(vr)==3
            @tensor vr[:]:=A'[-1,1,2]*A[-2,3,2]*vr[1,3,-3];
        elseif numind(vr)==2
            @tensor vr[:]:=A'[-1,1,2]*A[-2,3,2]*vr[1,3];
        end
    else
        if numind(vr)==5
            @tensor vr[:]:=A'[-1,7,6]*mpo'[-2,6,5,4]*mpo[-3,2,3,4]*A[-4,1,2]*vr[7,5,3,1,-5];
        elseif numind(vr)==4
            @tensor vr[:]:=A'[-1,7,6]*mpo'[-2,6,5,4]*mpo[-3,2,3,4]*A[-4,1,2]*vr[7,5,3,1];
        end
    end
    return vr
end

function left_eigenvector(A,mpo,type)

    if type=="A"
        HVfun1(x)=HV_L_tensor(x,A,[]);
        vl_init = permute(TensorMap(randn, space(A,1), space(A,1)), (1,2,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(HVfun1, vl_init, 1,:LM,Arnoldi());
        @assert maximum(abs.(eu)) == abs(eu[1])
        eu=eu[1];
        ev=ev[1];
        return ev,eu
    elseif type=="mpo_A"
        HVfun2(x)=HV_L_tensor(x,A,mpo);
        vl_init = permute(TensorMap(randn, space(A,1)*space(mpo,1),space(mpo,1)*space(A,1)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(HVfun2, vl_init, 1,:LM,Arnoldi());
        @assert maximum(abs.(eu)) == abs(eu[1])
        eu=eu[1];
        ev=ev[1];
        return ev,eu
    end
end

function right_eigenvector(A,mpo,type)

    if type=="A"
        HVfun1(x)=HV_R_tensor(x,A,[]);
        vr_init = permute(TensorMap(randn, space(A,2), space(A,2)), (1,2,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(HVfun1, vr_init, 1,:LM,Arnoldi());
        @assert maximum(abs.(eu)) == abs(eu[1])
        eu=eu[1];
        ev=ev[1];
        return ev,eu
    elseif type=="mpo_A"
        HVfun2(x)=HV_R_tensor(x,A,mpo);
        vr_init = permute(TensorMap(randn, space(A,2)*space(mpo,3),space(mpo,3)*space(A,2)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(HVfun2, vr_init, 1,:LM,Arnoldi());
        @assert maximum(abs.(eu)) == abs(eu[1])
        eu=eu[1];
        ev=ev[1];
        return ev,eu
    end
end



function left_eigenvalue(A1,A2,n)
    function HV_L_A1A2(vl,A1,A2)
        if numind(vl)==3
            @tensor vl[:]:=vl[-1,3,1]*A1'[3,-2,2]*A2[1,-3,2];
        elseif numind(vl)==2
            @tensor vl[:]:=vl[3,1]*A1'[3,-2,2]*A2[1,-3,2];
        end
    return vl
    end
    HVA1A2fun(x)=HV_L_A1A2(x,A1,A2);
    vl_init = permute(TensorMap(randn, space(A1,1), space(A2,1)), (1,2,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
    eu,ev=eigsolve(HVA1A2fun, vl_init, 1,:LM,Arnoldi());
    @assert maximum(abs.(eu)) == abs(eu[1])
    eu=eu[1:n];
    return eu
end

function itebd_ite(mpo,A,W,type)
    if type=="A"
        mpo_A=impo_imps(mpo,A);
        vl,_=left_eigenvector(mpo_A,[],"A");
        eu,ev=eig(permute(vl,(1,),(2,)));
        unitary_L=sqrt(eu)*ev';
        unitary_R=ev*pinv(sqrt(eu));
        @tensor AL[:]:=unitary_L[-1,1]*mpo_A[1,2,-3]*unitary_R[2,-2];
    elseif type=="mpo_A"
        vl,_=left_eigenvector(A,mpo,"mpo_A");
        eu,ev=eig(permute(vl,(2,1,),(3,4,)));
        unitary_L=sqrt(eu)*ev';
        unitary_R=ev*pinv(sqrt(eu));
        @tensor AL[:]:=unitary_L[-1,2,1]*mpo[2,3,4,-3]*A[1,5,3]*unitary_R[4,5,-2];
    end

    vr,_=right_eigenvector(AL,[],"A");

    U,S,_=tsvd(permute(vr,(1,),(2,)),trunc=truncdim(W));
    #println(diag(convert(Array,S)))
    @tensor A_trun[:]:=U[1,-1]*AL[1,2,-3]*U'[-2,2];

    return A_trun
end


function left_right_normalize(A)
    #left normalize
    vl,_=left_eigenvector(A,[],"A");
    eu,ev=eig(permute(vl,(1,),(2,)));
    unitary_L=sqrt(eu)*ev';
    unitary_R=ev*pinv(sqrt(eu));
    @tensor AL[:]:=unitary_L[-1,1]*A[1,2,-3]*unitary_R[2,-2];
    #transform right fixed point to be diagonal
    vr,eu=right_eigenvector(AL,[],"A");
    U,S,_=tsvd(permute(vr,(1,),(2,)));
    @tensor AL[:]:=U[1,-1]*AL[1,2,-3]*U'[-2,2];
    AL=AL/sqrt(eu);
    return AL
end


function ITEBD_boundary_groundstate(O1,O2,W,A_init,method)
    # ITEBD_boundary_groundstate(O1,O2,D,inv_tol):
        #use power method to obtain dominant boundary imps of nonhermitian MPO
        if method=="OO"
            @tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
            U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
            @tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];
        end
        #Initial state
        if A_init==[]
            mps_virtual=SU₂Space(0=>1,1/2=>1,1=>1);mps_phy=space(O1,2);
            A_init=permute(TensorMap(randn, mps_virtual*mps_virtual', mps_phy),(1,2,3,),());
        else 
        end
        AL=left_right_normalize(A_init);
        
        for cp=1:30
            AL_old=AL;
            @time if method=="O_O"
                A_trun_intermed=itebd_ite(O1,AL,3*W,"mpo_A");
                A_trun=itebd_ite(O2,A_trun_intermed,W,"mpo_A");
                AL=left_right_normalize(A_trun);
            elseif method=="OO"
                A_trun=itebd_ite(OO,AL,W,"mpo_A");#this "mpo_A" option is slightly faster when bond dimension is large
            end
            AL=left_right_normalize(A_trun);
    
            # vv,ee=right_eigenvector(AL,[],"A");
            # println(ee)
            # u,s,v=tsvd(permute(vv,(1,),(2,)));
            # println(sort(diag(convert(Array,s)),rev=true))
            
            #dominant eigenvalue of transfer matrix
            E=left_eigenvalue(impo_imps(O2,impo_imps(O1,AL)),AL,1)[1]
            ov=abs(left_eigenvalue(AL,AL_old,1)[1]);
            println("E="*string(E)*", "*"ov="*string(ov))
            if abs(ov-1)<1e-8
                break
            end
        
        end
        return AL,A_init
    end


    