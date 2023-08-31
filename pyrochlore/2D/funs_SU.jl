using LinearAlgebra
using TensorKit

function Truncations(uM,sM,vM,bond_dim,trun_tol)  
    sM=truncate_multiplet(sM,bond_dim,1e-5,trun_tol);
    uM_new,sM_new,vM_new=delet_zero_block(uM,sM,vM);
    @assert (norm(uM_new*sM_new*vM_new-uM*sM*vM)/norm(uM*sM*vM))<1e-14
    uM=uM_new;
    sM=sM_new;
    vM=vM_new;
    sM=sM/norm(sM)
    return uM,sM,vM
end

function hosvd_rotation_symmetric(A,trun_tol,bond_dim)
    uM,sM,vM = tsvd(A, (1,2,),(3,4,5,6,7,8,); trunc=truncdim(bond_dim));

    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);

    @tensor S[:]:=A[1,2,3,4,5,6,7,8]*uM'[-1,1,2]*uM'[-2,3,4]*uM'[-3,5,6]*uM'[-4,7,8];
    S=S/norm(S);
    return S, uM, sM

end

function hosvd(A,trun_tol,bond_dim)
    uMa,sMa,vMa = tsvd(A, (1,2,),(3,4,5,6,7,8,); trunc=truncdim(bond_dim));
    uMa,sMa,vMa=Truncations(uMa,sMa,vMa,bond_dim,trun_tol);

    uMb,sMb,vMb = tsvd(A, (3,4,),(1,2,5,6,7,8,); trunc=truncdim(bond_dim));
    uMb,sMb,vMb=Truncations(uMb,sMb,vMb,bond_dim,trun_tol);

    uMc,sMc,vMc = tsvd(A, (5,6,),(1,2,3,4,7,8,); trunc=truncdim(bond_dim));
    uMc,sMc,vMc=Truncations(uMc,sMc,vMc,bond_dim,trun_tol);

    uMd,sMd,vMd = tsvd(A, (7,8,),(1,2,3,4,5,6,); trunc=truncdim(bond_dim));
    uMd,sMd,vMd=Truncations(uMc,sMc,vMc,bond_dim,trun_tol);

    @tensor S[:]:=A[1,2,3,4,5,6,7,8]*uMa'[-1,1,2]*uMb'[-2,3,4]*uMc'[-3,5,6]*uMd'[-4,7,8];
    S=S/norm(S);

    # sma=diag(convert(Array,sMa))
    # sma=sort(sma,rev=true)
    # sma=sma/sma[1]
    # println(sma)
    # smb=diag(convert(Array,sMb))
    # smb=sort(smb,rev=true)
    # smb=smb/smb[1]
    # println(smb)

    return S, uMa,uMb,uMc,uMd, sMa,sMb,sMc,sMd

end




function trotter_gate(H,dt)
        eu,ev=eigh(H);
        @assert norm(ev*eu*ev'-H)/norm(H)<1e-14 
        gate=ev*exp(-dt*eu)*ev';
        gate_half=ev*exp(-dt*eu/2)*ev';
    return gate, gate_half
end


function Tri_T_dn(T_d, B_a, B_b, B_c, B_d, lambda_u_a, lambda_u_b, lambda_u_c, lambda_u_d, gate,U_phy_2,trun_tol,bond_dim,symmetric_hosvd)
    @tensor B_d_new[:]:=B_d[-1,1,-3]*lambda_u_d[-2,1];
    @tensor B_c_new[:]:=B_c[-1,1,-3]*lambda_u_c[-2,1];
    @tensor B_b_new[:]:=B_b[-1,1,-3]*lambda_u_b[-2,1];
    @tensor B_a_new[:]:=B_a[-1,1,-3]*lambda_u_a[-2,1];

    #@tensor A[:]:=T_d[5,6,7,8]*B_a_new[5,-1,1]*B_b_new[6,-2,2]*B_c_new[7,-3,3]*B_d_new[8,-4,4]*U_phy_2[-5,1,2]*U_phy_2[-6,3,4];
    @tensor A[:]:=T_d[5,6,7,8]*B_a_new[5,-1,-5]*B_b_new[6,-2,-6]*B_c_new[7,-3,-7]*B_d_new[8,-4,-8];

    #@tensor A[:]:=gate[1,2,-5,-6]*A[-1,-2,-3,-4,1,2];
    @tensor A[:]:=gate[1,2,3,4,-2,-4,-6,-8]*A[-1,-3,-5,-7,1,2,3,4];

    if symmetric_hosvd
        S_trun, U, lambda=hosvd_rotation_symmetric(A,trun_tol,bond_dim)

        unit=unitary(space(U,3)',space(U,3))
        @tensor U[:]:=U[-1,-2,1]*unit[1,-3]
        @tensor S_trun[:]:=S_trun[1,2,3,4]*unit'[-1,1]*unit'[-2,2]*unit'[-3,3]*unit'[-4,4]

        B_d=deepcopy(U);
        B_c=deepcopy(U);
        B_b=deepcopy(U);
        B_a=deepcopy(U);
        lambda_d_d=deepcopy(lambda)
        lambda_d_c=deepcopy(lambda)
        lambda_d_b=deepcopy(lambda)
        lambda_d_a=deepcopy(lambda)
    else
        S_trun, B_a,B_b,B_c,B_d, lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d=hosvd(A,trun_tol,bond_dim)

        unita=unitary(space(B_a,3)',space(B_a,3))
        @tensor B_a[:]:=B_a[-1,-2,1]*unita[1,-3]
        unitb=unitary(space(B_b,3)',space(B_b,3))
        @tensor B_b[:]:=B_b[-1,-2,1]*unitb[1,-3]
        unitc=unitary(space(B_c,3)',space(B_c,3))
        @tensor B_c[:]:=B_c[-1,-2,1]*unitc[1,-3]
        unitd=unitary(space(B_d,3)',space(B_d,3))
        @tensor B_d[:]:=B_d[-1,-2,1]*unitd[1,-3]

        @tensor S_trun[:]:=S_trun[1,2,3,4]*unita'[-1,1]*unitb'[-2,2]*unitc'[-3,3]*unitc'[-4,4]
    end

    lambda_u_d_inv=pinv(lambda_u_d)
    lambda_u_c_inv=pinv(lambda_u_c)
    lambda_u_b_inv=pinv(lambda_u_b)
    lambda_u_a_inv=pinv(lambda_u_a)

    B_d=permute(B_d,(3,1,2,),())
    B_c=permute(B_c,(3,1,2,),())
    B_b=permute(B_b,(3,1,2,),())
    B_a=permute(B_a,(3,1,2,),())

    @tensor B_d[:]:=B_d[-1,1,-3]*lambda_u_d_inv[-2,1]
    @tensor B_c[:]:=B_c[-1,1,-3]*lambda_u_c_inv[-2,1]
    @tensor B_b[:]:=B_b[-1,1,-3]*lambda_u_b_inv[-2,1]
    @tensor B_a[:]:=B_a[-1,1,-3]*lambda_u_a_inv[-2,1]
    return B_a, B_b, B_c, B_d, lambda_d_a, lambda_d_b, lambda_d_c, lambda_d_d, S_trun
end

function Tri_T_up(T_u, B_a, B_b, B_c, B_d, lambda_d_a, lambda_d_b, lambda_d_c, lambda_d_d, gate,U_phy_2, trun_tol,bond_dim,symmetric_hosvd)
    @tensor B_d_new[:]:=B_d[1,-2,-3]*lambda_d_d[-1,1]
    @tensor B_c_new[:]:=B_c[1,-2,-3]*lambda_d_c[-1,1]
    @tensor B_b_new[:]:=B_b[1,-2,-3]*lambda_d_b[-1,1]
    @tensor B_a_new[:]:=B_a[1,-2,-3]*lambda_d_a[-1,1]

    #@tensor A[:]:=T_u[5,6,7,8]*B_c_new[-1,5,1]*B_d_new[-2,6,2]*B_a_new[-3,7,3]*B_b_new[-4,8,4]*U_phy_2[-5,1,2]*U_phy_2[-6,3,4];
    @tensor A[:]:=T_u[5,6,7,8]*B_c_new[-1,5,-5]*B_d_new[-2,6,-6]*B_a_new[-3,7,-7]*B_b_new[-4,8,-8];

    #@tensor A[:]:=gate[1,2,-5,-6]*A[-1,-2,-3,-4,1,2]
    @tensor A[:]:=gate[1,2,3,4,-2,-4,-6,-8]*A[-1,-3,-5,-7,1,2,3,4];

    if symmetric_hosvd
        S_trun, U, lambda=hosvd_rotation_symmetric(A,trun_tol,bond_dim)
        
        unit=unitary(space(U,3)',space(U,3))
        @tensor U[:]:=U[-1,-2,1]*unit[1,-3]
        @tensor S_trun[:]:=S_trun[1,2,3,4]*unit'[-1,1]*unit'[-2,2]*unit'[-3,3]*unit'[-4,4]

        B_d=deepcopy(U)
        B_c=deepcopy(U)
        B_b=deepcopy(U)
        B_a=deepcopy(U)

        lambda_u_d=deepcopy(lambda)
        lambda_u_c=deepcopy(lambda)
        lambda_u_b=deepcopy(lambda)
        lambda_u_a=deepcopy(lambda)
    else
        S_trun, B_a,B_b,B_c,B_d, lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d=hosvd(A,trun_tol,bond_dim)
    
        unita=unitary(space(B_a,3)',space(B_a,3))
        @tensor B_a[:]:=B_a[-1,-2,1]*unita[1,-3]

        unitb=unitary(space(B_b,3)',space(B_b,3))
        @tensor B_b[:]:=B_b[-1,-2,1]*unitb[1,-3]

        unitc=unitary(space(B_c,3)',space(B_c,3))
        @tensor B_c[:]:=B_c[-1,-2,1]*unitc[1,-3]

        unitd=unitary(space(B_d,3)',space(B_d,3))
        @tensor B_d[:]:=B_d[-1,-2,1]*unitd[1,-3]

        @tensor S_trun[:]:=S_trun[1,2,3,4]*unita'[-1,1]*unitb'[-2,2]*unitc'[-3,3]*unitc'[-4,4]

    end
    
    lambda_d_d_inv=pinv(lambda_d_d)
    lambda_d_c_inv=pinv(lambda_d_c)
    lambda_d_b_inv=pinv(lambda_d_b)
    lambda_d_a_inv=pinv(lambda_d_a)

    B_d=permute(B_d,(1,3,2,),())
    B_c=permute(B_c,(1,3,2,),())
    B_b=permute(B_b,(1,3,2,),())
    B_a=permute(B_a,(1,3,2,),())

    @tensor B_d[:]:=B_d[1,-2,-3]*lambda_d_d_inv[-1,1]
    @tensor B_c[:]:=B_c[1,-2,-3]*lambda_d_c_inv[-1,1]
    @tensor B_b[:]:=B_b[1,-2,-3]*lambda_d_b_inv[-1,1]
    @tensor B_a[:]:=B_a[1,-2,-3]*lambda_d_a_inv[-1,1]

    return B_a, B_b, B_c, B_d, lambda_u_a, lambda_u_b, lambda_u_c, lambda_u_d, S_trun
end

function itebd_step(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, trun_tol, gate,U_phy_2, posit, bond_dim,symmetric_hosvd)
    # println("one step")
    # println(space(T_u))
    # println(space(T_d))
    if posit=="dn"
        B_a, B_b, B_c, B_d, lambda_d_a, lambda_d_b, lambda_d_c, lambda_d_d, T_d=Tri_T_dn(T_d, B_a, B_b, B_c, B_d, lambda_u_a, lambda_u_b, lambda_u_c, lambda_u_d, gate,U_phy_2, trun_tol, bond_dim,symmetric_hosvd)
    
    elseif posit=="up"
        B_a, B_b, B_c, B_d, lambda_u_a, lambda_u_b, lambda_u_c, lambda_u_d, T_u=Tri_T_up(T_u, B_a, B_b, B_c, B_d, lambda_d_a, lambda_d_b, lambda_d_c, lambda_d_d, gate,U_phy_2, trun_tol, bond_dim,symmetric_hosvd)
    end
    return T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d
end

function itebd(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, H,U_phy_2, trun_tol, tau, dt, bond_dim, symmetric_hosvd)
    gate, gate_half=trotter_gate(H, dt)

    T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d=itebd_step(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, trun_tol, gate_half,U_phy_2, "dn", bond_dim,symmetric_hosvd)
    for cs=1:Int(round(tau/dt))
        T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d=itebd_step(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, trun_tol, gate,U_phy_2, "up", bond_dim,symmetric_hosvd)
        # println(diag(convert(Array,lambda_u_c)))
        # println(diag(convert(Array,lambda_u_b)))
        T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d=itebd_step(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, trun_tol, gate,U_phy_2, "dn", bond_dim,symmetric_hosvd)
        # println(diag(convert(Array,lambda_u_c)))
        # println(diag(convert(Array,lambda_u_b)))
    end
    T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d=itebd_step(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, trun_tol, gate_half,U_phy_2, "up", bond_dim,symmetric_hosvd)

    return T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d
end