# function Truncations(uM,sM,vM,bond_dim,trun_tol)  
#     sM=truncate_multiplet(sM,bond_dim,1e-5,trun_tol);
#     uM_new,sM_new,vM_new=delet_zero_block(uM,sM,vM);
#     @assert (norm(uM_new*sM_new*vM_new-uM*sM*vM)/norm(uM*sM*vM))<1e-14
#     uM=uM_new;
#     sM=sM_new;
#     vM=vM_new;
#     sM=sM/norm(sM)
#     return uM,sM,vM
# end

function hosvd_rotation_symmetric(A,trun_tol,bond_dim)
    uM,sM,vM = tsvd(A, (1,2,),(3,4,5,6,); trunc=truncdim(bond_dim));

    #uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);

    @tensor S[:]:=A[1,2,3,4,5,6]*uM'[-1,1,2]*uM'[-2,3,4]*uM'[-3,5,6];
    S=S/norm(S);
    return S, uM, sM

end

function hosvd(A,trun_tol,bond_dim)
    A=A/norm(A);
    
    uMa,sMa,vMa = tsvd(A, (1,2,),(3,4,5,6,); trunc=truncdim(bond_dim));
    uMa,sMa,vMa=Truncations(uMa,sMa,vMa,bond_dim,trun_tol);

    uMb,sMb,vMb = tsvd(A, (3,4,),(1,2,5,6,); trunc=truncdim(bond_dim));
    uMb,sMb,vMb=Truncations(uMb,sMb,vMb,bond_dim,trun_tol);

    uMc,sMc,vMc = tsvd(A, (5,6,),(1,2,3,4,); trunc=truncdim(bond_dim));
    uMc,sMc,vMc=Truncations(uMc,sMc,vMc,bond_dim,trun_tol);

    @tensor S[:]:=A[1,2,3,4,5,6]*uMa'[-1,1,2]*uMb'[-2,3,4]*uMc'[-3,5,6];
    S=S/norm(S);

    #lambda tensor is attached to the simplex tensor

    return S, uMa,uMb,uMc, sMa,sMb,sMc

end




function trotter_gate(H,dt)
        eu,ev=eigh(H);
        @assert norm(ev*eu*ev'-H)/norm(H)<1e-14 
        gate=ev*exp(-dt*eu)*ev';
        gate_half=ev*exp(-dt*eu/2)*ev';
    return gate, gate_half
end


function Tri_T_dn(T_d, B_a, B_b, B_c, lambda_u_a, lambda_u_b, lambda_u_c, gate,trun_tol,bond_dim,symmetric_hosvd)
    println(space(B_c))
    println(space(lambda_u_c))
    @tensor B_c_new[:]:=B_c[-1,1,-3]*lambda_u_c[1,-2];
    @tensor B_b_new[:]:=B_b[-1,1,-3]*lambda_u_b[1,-2];
    @tensor B_a_new[:]:=B_a[-1,1,-3]*lambda_u_a[1,-2];

    @tensor A[:]:=T_d[1,2,3]*B_a_new[1,-1,-2]*B_b_new[2,-3,-4]*B_c_new[3,-5,-6];

    @tensor A[:]:=gate[1,2,3,-2,-4,-6]*A[-1,1,-3,2,-5,3];

    if symmetric_hosvd
        S_trun, U, lambda=hosvd_rotation_symmetric(A,trun_tol,bond_dim)

        unit=unitary(space(U,3)',space(U,3))
        @tensor U[:]:=U[-1,-2,1]*unit[1,-3]
        @tensor S_trun[:]:=S_trun[1,2,3]*unit'[-1,1]*unit'[-2,2]*unit'[-3,3]

        B_c=deepcopy(U);
        B_b=deepcopy(U);
        B_a=deepcopy(U);
        lambda_d_c=deepcopy(lambda)
        lambda_d_b=deepcopy(lambda)
        lambda_d_a=deepcopy(lambda)
    else
        S_trun, B_a,B_b,B_c, lambda_d_a,lambda_d_b,lambda_d_c=hosvd(A,trun_tol,bond_dim)

        # unita=unitary(space(B_a,3)',space(B_a,3))
        # @tensor B_a[:]:=B_a[-1,-2,1]*unita[1,-3]
        # unitb=unitary(space(B_b,3)',space(B_b,3))
        # @tensor B_b[:]:=B_b[-1,-2,1]*unitb[1,-3]
        # unitc=unitary(space(B_c,3)',space(B_c,3))
        # @tensor B_c[:]:=B_c[-1,-2,1]*unitc[1,-3]

        #@tensor S_trun[:]:=S_trun[1,2,3]*unita'[-1,1]*unitb'[-2,2]*unitc'[-3,3]
    end

    lambda_u_c_inv=pinv(lambda_u_c)
    lambda_u_b_inv=pinv(lambda_u_b)
    lambda_u_a_inv=pinv(lambda_u_a)

    B_c=permute(B_c,(3,1,2,),())
    B_b=permute(B_b,(3,1,2,),())
    B_a=permute(B_a,(3,1,2,),())

    @tensor B_c[:]:=B_c[-1,1,-3]*lambda_u_c_inv[1,-2]
    @tensor B_b[:]:=B_b[-1,1,-3]*lambda_u_b_inv[1,-2]
    @tensor B_a[:]:=B_a[-1,1,-3]*lambda_u_a_inv[1,-2]

    verify_D_max(S_trun,S_trun,B_a,B_b,B_c, bond_dim);

    lambda_d_a=lambda_d_a/norm(lambda_d_a);
    lambda_d_b=lambda_d_b/norm(lambda_d_b);
    lambda_d_c=lambda_d_c/norm(lambda_d_c);
    S_trun=S_trun/norm(S_trun);
    return B_a, B_b, B_c, lambda_d_a, lambda_d_b, lambda_d_c, S_trun
end

function Tri_T_up(T_u, B_a, B_b, B_c, lambda_d_a, lambda_d_b, lambda_d_c, gate, trun_tol,bond_dim,symmetric_hosvd)
    println(space(B_c))
    println(space(lambda_d_c))
    @tensor B_c_new[:]:=B_c[1,-2,-3]*lambda_d_c[1,-1]
    @tensor B_b_new[:]:=B_b[1,-2,-3]*lambda_d_b[1,-1]
    @tensor B_a_new[:]:=B_a[1,-2,-3]*lambda_d_a[1,-1]

    @tensor A[:]:=T_u[1,2,3]*B_a_new[-1,1,-2]*B_b_new[-3,2,-4]*B_c_new[-5,3,-6]
    @tensor A[:]:=gate[1,2,3,-2,-4,-6]*A[-1,1,-3,2,-5,3]
    if symmetric_hosvd
        S_trun, U, lambda=hosvd_rotation_symmetric(A,trun_tol,bond_dim)
        
        unit=unitary(space(U,3)',space(U,3))
        @tensor U[:]:=U[-1,-2,1]*unit[1,-3]
        @tensor S_trun[:]:=S_trun[1,2,3]*unit'[-1,1]*unit'[-2,2]*unit'[-3,3]

        B_c=deepcopy(U)
        B_b=deepcopy(U)
        B_a=deepcopy(U)

        lambda_u_c=deepcopy(lambda)
        lambda_u_b=deepcopy(lambda)
        lambda_u_a=deepcopy(lambda)
    else
        S_trun, B_a,B_b,B_c, lambda_u_a,lambda_u_b,lambda_u_c=hosvd(A,trun_tol,bond_dim)
    
        # unita=unitary(space(B_a,3)',space(B_a,3))
        # @tensor B_a[:]:=B_a[-1,-2,1]*unita[1,-3]

        # unitb=unitary(space(B_b,3)',space(B_b,3))
        # @tensor B_b[:]:=B_b[-1,-2,1]*unitb[1,-3]

        # unitc=unitary(space(B_c,3)',space(B_c,3))
        # @tensor B_c[:]:=B_c[-1,-2,1]*unitc[1,-3]

        # @tensor S_trun[:]:=S_trun[1,2,3]*unita'[-1,1]*unitb'[-2,2]*unitc'[-3,3]

    end
    
    
    lambda_d_c_inv=pinv(lambda_d_c)
    lambda_d_b_inv=pinv(lambda_d_b)
    lambda_d_a_inv=pinv(lambda_d_a)

    B_c=permute(B_c,(1,3,2,),())
    B_b=permute(B_b,(1,3,2,),())
    B_a=permute(B_a,(1,3,2,),())

    @tensor B_c[:]:=B_c[1,-2,-3]*lambda_d_c_inv[1,-1]
    @tensor B_b[:]:=B_b[1,-2,-3]*lambda_d_b_inv[1,-1]
    @tensor B_a[:]:=B_a[1,-2,-3]*lambda_d_a_inv[1,-1]

    verify_D_max(S_trun,S_trun,B_a,B_b,B_c, bond_dim);

    lambda_u_a=lambda_u_a/norm(lambda_u_a);
    lambda_u_b=lambda_u_b/norm(lambda_u_b);
    lambda_u_c=lambda_u_c/norm(lambda_u_c);
    S_trun=S_trun/norm(S_trun);
    return B_a, B_b, B_c, lambda_u_a, lambda_u_b, lambda_u_c, S_trun
end

function verify_D_max(T_u,T_d,B_a,B_b,B_c, bond_dim)
    @assert dim(space(T_u,1))<=bond_dim;
    @assert dim(space(T_u,2))<=bond_dim;
    @assert dim(space(T_u,3))<=bond_dim;
    @assert dim(space(T_d,1))<=bond_dim;
    @assert dim(space(T_d,2))<=bond_dim;
    @assert dim(space(T_d,3))<=bond_dim;
end

function itebd_step(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, trun_tol, gate, posit, bond_dim,symmetric_hosvd)
    
    if posit=="dn"
        B_a, B_b, B_c, lambda_d_a, lambda_d_b, lambda_d_c, T_d=Tri_T_dn(T_d, B_a, B_b, B_c, lambda_u_a, lambda_u_b, lambda_u_c, gate, trun_tol, bond_dim,symmetric_hosvd);
        verify_D_max(T_u,T_d,B_a,B_b,B_c, bond_dim);
    elseif posit=="up"
        B_a, B_b, B_c, lambda_u_a, lambda_u_b, lambda_u_c, T_u=Tri_T_up(T_u, B_a, B_b, B_c, lambda_d_a, lambda_d_b, lambda_d_c, gate, trun_tol, bond_dim,symmetric_hosvd);
        verify_D_max(T_u,T_d,B_a,B_b,B_c, bond_dim);
    end
    return T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c
end

function itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H, trun_tol, tau, dt, bond_dim, symmetric_hosvd)
    gate, gate_half=trotter_gate(H, dt)
    
    T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd_step(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, trun_tol, gate_half, "dn", bond_dim,symmetric_hosvd)
    for cs=1:Int(round(tau/dt))
        T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd_step(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, trun_tol, gate, "up", bond_dim,symmetric_hosvd)
        # println(diag(convert(Array,lambda_u_c)))
        # println(diag(convert(Array,lambda_u_b)))
        T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd_step(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, trun_tol, gate, "dn", bond_dim,symmetric_hosvd)
        # println(diag(convert(Array,lambda_u_c)))
        # println(diag(convert(Array,lambda_u_b)))
    end
    T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd_step(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, trun_tol, gate_half, "up", bond_dim,symmetric_hosvd)

    return T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c
end