using LinearAlgebra
using TensorKit

function change_space(uM,sM,vM)
    #change space from V to V'
    Unitary=unitary(space(sM,1)',space(sM,1))
    uM=uM*Unitary';
    sM=Unitary*sM*Unitary';
    vM=Unitary*vM;
    return uM,sM,vM
end

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


function hosvd(A,trun_tol,bond_dim)
    uMa,sMa,vMa = tsvd(A, (1,2,),(3,4,5,6,); trunc=truncdim(bond_dim));
    uMa,sMa,vMa=Truncations(uMa,sMa,vMa,bond_dim,trun_tol);

    uMb,sMb,vMb = tsvd(A, (3,4,),(1,2,5,6,); trunc=truncdim(bond_dim));
    uMb,sMb,vMb=Truncations(uMb,sMb,vMb,bond_dim,trun_tol);

    uMc,sMc,vMc = tsvd(A, (5,6,),(1,2,3,4,); trunc=truncdim(bond_dim));
    uMc,sMc,vMc=Truncations(uMc,sMc,vMc,bond_dim,trun_tol);

    @tensor S[:]:=A[1,2,3,4,5,6]*uMa'[-1,1,2]*uMb'[-2,3,4]*uMc'[-3,5,6];
    S=S/norm(S);

    # sma=diag(convert(Array,sMa))
    # sma=sort(sma,rev=true)
    # sma=sma/sma[1]
    # println(sma)
    # smb=diag(convert(Array,sMb))
    # smb=sort(smb,rev=true)
    # smb=smb/smb[1]
    # println(smb)

    return S, uMa,uMb,uMc, sMa,sMb,sMc

end




function trotter_gate(H,dt)
        eu,ev=eigh(H);
        #@assert norm(ev*eu*ev'-H)/norm(H)<1e-14 
        gate=ev*exp(-dt*eu)*ev';
        gate_half=ev*exp(-dt*eu/2)*ev';
    return gate, gate_half
end


function Tri_T_dn(T_d, B_a, B_b, B_c, lambda_u_a, lambda_u_b, lambda_u_c, gate,trun_tol,bond_dim)
    @tensor B_c_new[:]:=B_c[-1,1,-3]*lambda_u_c[-2,1];
    @tensor B_b_new[:]:=B_b[-1,1,-3]*lambda_u_b[-2,1];
    @tensor B_a_new[:]:=B_a[-1,1,-3]*lambda_u_a[-2,1];

    @tensor A[:]:=T_d[1,2,3]*B_a_new[1,-1,-2]*B_b_new[2,-3,-4]*B_c_new[3,-5,-6];

    @tensor A[:]:=gate[1,2,3,-2,-4,-6]*A[-1,1,-3,2,-5,3];


    S_trun, B_a,B_b,B_c, lambda_d_a,lambda_d_b,lambda_d_c=hosvd(A,trun_tol,bond_dim)

    unita=unitary(space(B_a,3)',space(B_a,3))
    @tensor B_a[:]:=B_a[-1,-2,1]*unita[1,-3]
    unitb=unitary(space(B_b,3)',space(B_b,3))
    @tensor B_b[:]:=B_b[-1,-2,1]*unitb[1,-3]
    unitc=unitary(space(B_c,3)',space(B_c,3))
    @tensor B_c[:]:=B_c[-1,-2,1]*unitc[1,-3]

    @tensor S_trun[:]:=S_trun[1,2,3]*unita'[-1,1]*unitb'[-2,2]*unitc'[-3,3]


    lambda_u_c_inv=pinv(lambda_u_c)
    lambda_u_b_inv=pinv(lambda_u_b)
    lambda_u_a_inv=pinv(lambda_u_a)

    B_c=permute(B_c,(),(3,1,2,))
    B_b=permute(B_b,(),(3,1,2,))
    B_a=permute(B_a,(),(3,1,2,))

    @tensor B_c[:]:=B_c[-1,1,-3]*lambda_u_c_inv[-2,1]
    @tensor B_b[:]:=B_b[-1,1,-3]*lambda_u_b_inv[-2,1]
    @tensor B_a[:]:=B_a[-1,1,-3]*lambda_u_a_inv[-2,1]

    B_c=permute(B_c,(1,2,),(3,))
    B_b=permute(B_b,(1,2,),(3,))
    B_a=permute(B_a,(1,2,),(3,))
    S_trun=permute(S_trun,(),(1,2,3,));
    return S_trun, B_a, B_b, B_c, lambda_d_a, lambda_d_b, lambda_d_c
end

function Tri_T_up(T_u, B_a, B_b, B_c, lambda_d_a, lambda_d_b, lambda_d_c, gate, trun_tol,bond_dim)

    @tensor B_c_new[:]:=B_c[1,-2,-3]*lambda_d_c[-1,1]
    @tensor B_b_new[:]:=B_b[1,-2,-3]*lambda_d_b[-1,1]
    @tensor B_a_new[:]:=B_a[1,-2,-3]*lambda_d_a[-1,1]

    @tensor A[:]:=T_u[1,2,3]*B_a_new[-1,1,-2]*B_b_new[-3,2,-4]*B_c_new[-5,3,-6]
    @tensor A[:]:=gate[1,2,3,-2,-4,-6]*A[-1,1,-3,2,-5,3]

    S_trun, B_a,B_b,B_c, lambda_u_a,lambda_u_b,lambda_u_c=hosvd(A,trun_tol,bond_dim)

    unita=unitary(space(B_a,3)',space(B_a,3))
    @tensor B_a[:]:=B_a[-1,-2,1]*unita[1,-3]
    unitb=unitary(space(B_b,3)',space(B_b,3))
    @tensor B_b[:]:=B_b[-1,-2,1]*unitb[1,-3]
    unitc=unitary(space(B_c,3)',space(B_c,3))
    @tensor B_c[:]:=B_c[-1,-2,1]*unitc[1,-3]

    @tensor S_trun[:]:=S_trun[1,2,3]*unita'[-1,1]*unitb'[-2,2]*unitc'[-3,3]
    
    
    lambda_d_c_inv=pinv(lambda_d_c)
    lambda_d_b_inv=pinv(lambda_d_b)
    lambda_d_a_inv=pinv(lambda_d_a)

    B_c=permute(B_c,(1,3,2,),())
    B_b=permute(B_b,(1,3,2,),())
    B_a=permute(B_a,(1,3,2,),())

    @tensor B_c[:]:=B_c[1,-2,-3]*lambda_d_c_inv[-1,1]
    @tensor B_b[:]:=B_b[1,-2,-3]*lambda_d_b_inv[-1,1]
    @tensor B_a[:]:=B_a[1,-2,-3]*lambda_d_a_inv[-1,1]

    B_c=permute(B_c,(1,2,),(3,))
    B_b=permute(B_b,(1,2,),(3,))
    B_a=permute(B_a,(1,2,),(3,))
    S_trun=permute(S_trun,(),(1,2,3,));
    return S_trun, B_a, B_b, B_c, lambda_u_a, lambda_u_b, lambda_u_c
end

function triangle_update(T_d_envs, T_u_envs, bond_dim,trun_tol,gate,Cell_ind, T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set)
    for cc=1:9
        Cell=Cell_ind[cc];
        #be careful: input lambda and output lambda are different
        T_d_tem, B_a_tem, B_b_tem, B_c_tem, lambda_a_d_tem, lambda_b_d_tem, lambda_c_d_tem=Tri_T_dn(T_d_set[Cell[1]], B_a_set[Cell[1]], B_b_set[Cell[1]], B_c_set[Cell[1]], lambda_a_u_set[T_d_envs[Cell[1]][1]], lambda_b_u_set[T_d_envs[Cell[1]][2]], lambda_c_u_set[T_d_envs[Cell[1]][3]], gate,trun_tol,bond_dim);

        T_d_set[Cell[1]]=T_d_tem;
        B_a_set[Cell[1]]=B_a_tem;
        B_b_set[Cell[1]]=B_b_tem;
        B_c_set[Cell[1]]=B_c_tem;
        lambda_a_d_set[Cell[1]]=lambda_a_d_tem;
        lambda_b_d_set[Cell[1]]=lambda_b_d_tem;
        lambda_c_d_set[Cell[1]]=lambda_c_d_tem;
    end

    for cc=1:9
        Cell=Cell_ind[cc];
        #be careful: input lambda and output lambda are different
        T_u_tem, B_a_tem, B_b_tem, B_c_tem, lambda_a_u_tem, lambda_b_u_tem, lambda_c_u_tem=Tri_T_up(T_u_set[Cell[2]], B_a_set[T_u_envs[Cell[2]][1]], B_b_set[T_u_envs[Cell[2]][2]], B_c_set[T_u_envs[Cell[2]][3]], lambda_a_d_set[T_u_envs[Cell[2]][1]], lambda_b_d_set[T_u_envs[Cell[2]][2]], lambda_c_d_set[T_u_envs[Cell[2]][3]], gate,trun_tol,bond_dim);

        T_u_set[Cell[2]]=T_u_tem;
        B_a_set[T_u_envs[Cell[2]][1]]=B_a_tem;
        B_b_set[T_u_envs[Cell[2]][2]]=B_b_tem;
        B_c_set[T_u_envs[Cell[2]][3]]=B_c_tem;
        lambda_a_u_set[Cell[2]]=lambda_a_u_tem;
        lambda_b_u_set[Cell[2]]=lambda_b_u_tem;
        lambda_c_u_set[Cell[2]]=lambda_c_u_tem;
    end
end

function evol_J2_term1(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10)

    @assert space(lambda_a_dn2,1)'==space(Td1,1);
    @assert space(lambda_c_up3,1)'==space(Tu2,3);

    @assert space(lambda_c_dn6,1)'==space(Td1,3);
    @assert space(lambda_b_dn7,1)'==space(Td1,2);
    @assert space(lambda_b_up8,1)'==space(Tu2,2);
    @assert space(lambda_a_up9,1)'==space(Tu2,1);

    
    
    
    #absorb lambda
    @tensor Bc1[:]:=Bc1[-1,1,-3]*lambda_c_up1[-2,1];
    # println(space(Ba3))
    # println(space(lambda_a_dn10))
    @tensor Ba3[:]:=Ba3[1,-2,-3]*lambda_a_dn10[-1,1];


    #fuse legs
    U_Bc1=unitary(fuse(space(Bc1,2)⊗space(Bc1,3)),space(Bc1,2)⊗space(Bc1,3))
    U_Ba3=unitary(fuse(space(Ba3,1)⊗space(Ba3,3)),space(Ba3,1)⊗space(Ba3,3))
    @tensor Bc1Ba3[:]:=Bc1[-3,-1,1]*Ba3[-4,-6,2]*gate[1,2,-2,-5];
    @tensor Bc1Ba3[:]:=Bc1Ba3[1,2,-2,3,4,-4]*U_Bc1[-1,1,2]*U_Ba3[-3,3,4];

    @tensor T_top[:]:=Td1[-2,1,-1]*Bb2[1,2,-3]*Tu2[-5,2,-4];
    U_top=unitary(fuse(space(T_top,2)⊗space(T_top,3)⊗space(T_top,4)),space(T_top,2)⊗space(T_top,3)⊗space(T_top,4));
    @tensor T_top[:]:=T_top[-1,1,2,3,-3]*U_top[-2,1,2,3];

    #contract the whole tensor
    @tensor T[:]:=T_top[1,-2,2]*Bc1Ba3[-1,1,-3,2];

    #decompose
    uM,sM,vM = tsvd(T, (1,),(2,3,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    uM,sM,vM=change_space(uM,sM,vM);
    lambda_c_dn6=permute(sM,(2,),(1,));
    @tensor Bc1[:]:=uM[1,-1]*U_Bc1'[-2,-3,1];
    @tensor Bc1[:]:=Bc1[-1,1,-3]*pinv(lambda_c_up1)[-2,1];
    Bc1=permute(Bc1,(1,2,),(3,))

    T=sM*vM;
    uM,sM,vM = tsvd(T, (1,2,),(3,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    @tensor Ba3[:]:=vM[-2,1]*U_Ba3'[-1,-3,1];
    @tensor Ba3[:]:=Ba3[1,-2,-3]*pinv(lambda_a_dn10)[-1,1];
    Ba3=permute(Ba3,(1,2,),(3,))
    lambda_a_up9=sM;
    
    T=uM*sM;
    @tensor T[:]:=T[-1,1,-5]*U_top'[-2,-3,-4,1];
    uM,sM,vM = tsvd(T, (1,2,),(3,4,5,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    Td1=permute(uM*sM,(),(2,3,1,));
    lambda_b_dn7=sM;

    

    T=sM*vM;
    uM,sM,vM = tsvd(T, (1,2,),(3,4,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    uM,sM,vM=change_space(uM,sM,vM);
    Tu2=permute(sM*vM,(),(3,1,2,));
    lambda_b_up8=permute(sM,(2,),(1,));


    @tensor Bb2[:]:=uM[1,-3,-2]*pinv(lambda_b_dn7)[-1,1];
    Bb2=permute(Bb2,(1,2,),(3,));


    return Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10

end

function evol_J2_term2(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10)

    #permute index to accomadate
    Td1=permute(Td1,(),(3,1,2,));
    Tu2=permute(Tu2,(),(3,1,2,));

    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));
    

    Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
    evol_J2_term1(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10);

    #permute index back
    Td1=permute(Td1,(),(2,3,1,));
    Tu2=permute(Tu2,(),(2,3,1,));

    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));

    return Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10

end

function evol_J2_term3(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10)

    #permute index to accomadate
    Td1=permute(Td1,(),(2,3,1,));
    Tu2=permute(Tu2,(),(2,3,1,));

    Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
    evol_J2_term1(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10);
    
    #permute index back
    Td1=permute(Td1,(),(3,1,2,));
    Tu2=permute(Tu2,(),(3,1,2,));

    return Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10

end

function evol_J2_term4(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10)

    #permute index to accomadate
    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));

    Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
    evol_J2_term1(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10);
    
    #permute index back
    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));

    return Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10

end

function evol_J2_term5(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10)

    #permute index to accomadate
    Td1=permute(Td1,(),(3,1,2,));
    Tu2=permute(Tu2,(),(3,1,2,));

    Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
    evol_J2_term1(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10);
    
    #permute index back
    Td1=permute(Td1,(),(2,3,1,));
    Tu2=permute(Tu2,(),(2,3,1,));

    return Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10

end

function evol_J2_term6(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10)

    #permute index to accomadate
    Td1=permute(Td1,(),(2,3,1,));
    Tu2=permute(Tu2,(),(2,3,1,));

    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));


    Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
    evol_J2_term1(bond_dim,trun_tol,gate,Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10);
    
    #permute index back
    Td1=permute(Td1,(),(3,1,2,));
    Tu2=permute(Tu2,(),(3,1,2,));

    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));

    return Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10

end

function evol_J3_term1_α(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)

    @assert space(lambda_a_dn2,1)'==space(Td1,1);
    @assert space(lambda_c_up3,1)'==space(Tu2,3);
    @assert space(lambda_b_dn4,1)'==space(Td3,2);

    @assert space(lambda_c_dn6,1)'==space(Td1,3);
    @assert space(lambda_b_dn7,1)'==space(Td1,2);
    @assert space(lambda_b_up8,1)'==space(Tu2,2);
    @assert space(lambda_a_up9,1)'==space(Tu2,1);
    @assert space(lambda_a_dn10,1)'==space(Td3,1);
    @assert space(lambda_c_dn11,1)'==space(Td3,3);
    
    
    
    #absorb lambda
    @tensor Bc1[:]:=Bc1[-1,1,-3]*lambda_c_up1[-2,1];
    @tensor Bc4[:]:=Bc4[-1,1,-3]*lambda_c_up5[-2,1];


    #fuse legs
    U_Bc1=unitary(fuse(space(Bc1,2)⊗space(Bc1,3)),space(Bc1,2)⊗space(Bc1,3))
    U_Bc4=unitary(fuse(space(Bc4,2)⊗space(Bc4,3)),space(Bc4,2)⊗space(Bc4,3))
    @tensor Bc1Bc4[:]:=Bc1[-3,-1,1]*Bc4[-6,-4,2]*gate[1,2,-2,-5];
    @tensor Bc1Bc4[:]:=Bc1Bc4[1,2,-2,3,4,-4]*U_Bc1[-1,1,2]*U_Bc4[-3,3,4];

    @tensor T_top[:]:=Bb2[-1,1,-2]*Tu2[2,1,-3]*Ba3[-5,2,-4];
    U_top=unitary(fuse(space(T_top,2)⊗space(T_top,3)⊗space(T_top,4)),space(T_top,2)⊗space(T_top,3)⊗space(T_top,4));
    @tensor T_top[:]:=T_top[-1,1,2,3,-3]*U_top[-2,1,2,3];

    #contract the whole tensor
    @tensor T[:]:=T_top[2,-3,3]*Td1[-2,2,1]*Td3[3,-4,4]*Bc1Bc4[-1,1,-5,4];

    #decompose
    uM,sM,vM = tsvd(T, (1,),(2,3,4,5,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    uM,sM,vM=change_space(uM,sM,vM);
    lambda_c_dn6=permute(sM,(2,),(1,));
    @tensor Bc1[:]:=uM[1,-1]*U_Bc1'[-2,-3,1];
    @tensor Bc1[:]:=Bc1[-1,1,-3]*pinv(lambda_c_up1)[-2,1];
    Bc1=permute(Bc1,(1,2,),(3,))

    T=sM*vM;
    uM,sM,vM = tsvd(T, (1,2,),(3,4,5,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    Td1=uM*sM;
    Td1=permute(Td1,(),(2,3,1,));
    lambda_b_dn7=sM;

    T=sM*vM;
    uM,sM,vM = tsvd(T, (1,2,3,),(4,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    lambda_c_dn11=sM;
    @tensor Bc4[:]:=vM[-1,1]*U_Bc4'[-2,-3,1];
    @tensor Bc4[:]:=Bc4[-1,1,-3]*pinv(lambda_c_up5)[-2,1];
    Bc4=permute(Bc4,(1,2,),(3,));
    

    T=uM*sM;
    uM,sM,vM = tsvd(T, (1,2,),(3,4,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    uM,sM,vM=change_space(uM,sM,vM);
    Td3=sM*vM;
    Td3=permute(Td3,(),(1,2,3,));
    lambda_a_dn10=permute(sM,(2,),(1,));
    
    T=uM*sM;
    @tensor T[:]:=T[-1,1,-5]*U_top'[-2,-3,-4,1];
    uM,sM,vM = tsvd(T, (1,2,),(3,4,5,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    uM,sM,vM=change_space(uM,sM,vM);
    @tensor Bb2[:]:=uM[1,-3,-2]*pinv(lambda_b_dn7)[-1,1];
    Bb2=permute(Bb2,(1,2,),(3,));
    Bb2=Bb2/norm(Bb2);
    lambda_b_up8=permute(sM,(2,),(1,));

    T=sM*vM;
    uM,sM,vM = tsvd(T, (1,2,),(3,4,); trunc=truncdim(bond_dim));
    sM=sM/norm(sM);
    uM,sM,vM=Truncations(uM,sM,vM,bond_dim,trun_tol);
    @tensor Ba3[:]:=vM[-2,-3,1]*pinv(lambda_a_dn10)[-1,1];
    Ba3=permute(Ba3,(1,2,),(3,));
    lambda_a_up9=sM;

    Tu2=permute(uM*sM,(),(3,1,2,));

    
    

    # println(space(Bc1))
    # println(space(lambda_c_dn6))
    # println(space(Td1))
    # println(space(lambda_b_dn7))
    # println(space(Bc4))
    # println(space(lambda_c_dn11))
    # println(space(Td3))
    # println(space(lambda_a_dn10))
    # println(space(Bb2))
    # println(space(lambda_b_up8))
    # println(space(Ba3))
    # println(space(lambda_a_up9))
    # println(space(Tu2))

    return Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11

end

function evol_J3_term1_β(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)

    #permute index to accomadate
    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));
    Bc4=permute(Bc4,(2,1,),(3,));

    Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
    evol_J3_term1_α(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)
    
    #permute index back
    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));
    Bc4=permute(Bc4,(2,1,),(3,));


    
    return Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11
end

function evol_J3_term2_α(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)

    #permute index to accomadate
    Td1=permute(Td1,(),(3,1,2,));
    Tu2=permute(Tu2,(),(3,1,2,));
    Td3=permute(Td3,(),(3,1,2,));


    Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
    evol_J3_term1_α(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)
    
    #permute index back
    Td1=permute(Td1,(),(2,3,1,));
    Tu2=permute(Tu2,(),(2,3,1,));
    Td3=permute(Td3,(),(2,3,1,));


    return Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11
end

function evol_J3_term2_β(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)

    #permute index to accomadate
    Td1=permute(Td1,(),(3,1,2,));
    Tu2=permute(Tu2,(),(3,1,2,));
    Td3=permute(Td3,(),(3,1,2,));

    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));
    Bc4=permute(Bc4,(2,1,),(3,));


    Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
    evol_J3_term1_α(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)
    
    #permute index back
    Td1=permute(Td1,(),(2,3,1,));
    Tu2=permute(Tu2,(),(2,3,1,));
    Td3=permute(Td3,(),(2,3,1,));

    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));
    Bc4=permute(Bc4,(2,1,),(3,));

    
    return Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11
end

function evol_J3_term3_α(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)

    #permute index to accomadate
    Td1=permute(Td1,(),(2,3,1,));
    Tu2=permute(Tu2,(),(2,3,1,));
    Td3=permute(Td3,(),(2,3,1,));


    Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
    evol_J3_term1_α(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)
    
    #permute index back
    Td1=permute(Td1,(),(3,1,2,));
    Tu2=permute(Tu2,(),(3,1,2,));
    Td3=permute(Td3,(),(3,1,2,));

    
    return Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11
end

function evol_J3_term3_β(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)

    #permute index to accomadate
    Td1=permute(Td1,(),(2,3,1,));
    Tu2=permute(Tu2,(),(2,3,1,));
    Td3=permute(Td3,(),(2,3,1,));

    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));
    Bc4=permute(Bc4,(2,1,),(3,));


    Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
    evol_J3_term1_α(bond_dim,trun_tol,gate,Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11)
    
    #permute index back
    Td1=permute(Td1,(),(3,1,2,));
    Tu2=permute(Tu2,(),(3,1,2,));
    Td3=permute(Td3,(),(3,1,2,));

    Bc1=permute(Bc1,(2,1,),(3,));
    Bb2=permute(Bb2,(2,1,),(3,));
    Ba3=permute(Ba3,(2,1,),(3,));
    Bc4=permute(Bc4,(2,1,),(3,));

    
    return Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11
end
function J2_update(bond_dim,trun_tol,gate,Cell_ind, T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set)

    for cc=1:9
        Cell=Cell_ind[cc];
        

        Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
        evol_J2_term1(bond_dim,trun_tol,gate,T_d_set[Cell[1]],T_u_set[Cell[2]],B_c_set[Cell[1]], B_b_set[Cell[1]], B_a_set[Cell[3]],lambda_c_u_set[Cell[6]], lambda_a_d_set[Cell[1]], lambda_c_u_set[Cell[2]],lambda_c_d_set[Cell[1]], lambda_b_d_set[Cell[1]], lambda_b_u_set[Cell[2]], lambda_a_u_set[Cell[2]], lambda_a_d_set[Cell[3]]);

        T_d_set[Cell[1]]=Td1;
        T_u_set[Cell[2]]=Tu2;
        B_c_set[Cell[1]]=Bc1;
        B_b_set[Cell[1]]=Bb2;
        B_a_set[Cell[3]]=Ba3;
        lambda_c_u_set[Cell[6]]=lambda_c_up1;
        lambda_a_d_set[Cell[1]]=lambda_a_dn2;
        lambda_c_u_set[Cell[2]]=lambda_c_up3;
        lambda_c_d_set[Cell[1]]=lambda_c_dn6;
        lambda_b_d_set[Cell[1]]=lambda_b_dn7;
        lambda_b_u_set[Cell[2]]=lambda_b_up8;
        lambda_a_u_set[Cell[2]]=lambda_a_up9;
        lambda_a_d_set[Cell[3]]=lambda_a_dn10;

        Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
        evol_J2_term2(bond_dim,trun_tol,gate,T_u_set[Cell[2]],T_d_set[Cell[3]],B_b_set[Cell[1]], B_a_set[Cell[3]], B_c_set[Cell[3]],lambda_b_d_set[Cell[1]], lambda_c_u_set[Cell[2]], lambda_b_d_set[Cell[3]],lambda_b_u_set[Cell[2]], lambda_a_u_set[Cell[2]], lambda_a_d_set[Cell[3]], lambda_c_d_set[Cell[3]], lambda_c_u_set[Cell[4]]);
        
        T_u_set[Cell[2]]=Td1;
        T_d_set[Cell[3]]=Tu2;
        B_b_set[Cell[1]]=Bc1;
        B_a_set[Cell[3]]=Bb2;
        B_c_set[Cell[3]]=Ba3;
        lambda_b_d_set[Cell[1]]=lambda_c_up1;
        lambda_c_u_set[Cell[2]]=lambda_a_dn2;
        lambda_b_d_set[Cell[3]]=lambda_c_up3;
        lambda_b_u_set[Cell[2]]=lambda_c_dn6;
        lambda_a_u_set[Cell[2]]=lambda_b_dn7;
        lambda_a_d_set[Cell[3]]=lambda_b_up8;
        lambda_c_d_set[Cell[3]]=lambda_a_up9;
        lambda_c_u_set[Cell[4]]=lambda_a_dn10;

        Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
        evol_J2_term3(bond_dim,trun_tol,gate,T_d_set[Cell[3]],T_u_set[Cell[4]],B_a_set[Cell[3]], B_c_set[Cell[3]], B_b_set[Cell[5]],lambda_a_u_set[Cell[2]], lambda_b_d_set[Cell[3]], lambda_a_u_set[Cell[4]],lambda_a_d_set[Cell[3]], lambda_c_d_set[Cell[3]], lambda_c_u_set[Cell[4]], lambda_b_u_set[Cell[4]], lambda_b_d_set[Cell[5]]);
        
        T_d_set[Cell[3]]=Td1;
        T_u_set[Cell[4]]=Tu2;
        B_a_set[Cell[3]]=Bc1;
        B_c_set[Cell[3]]=Bb2;
        B_b_set[Cell[5]]=Ba3;
        lambda_a_u_set[Cell[2]]=lambda_c_up1;
        lambda_b_d_set[Cell[3]]=lambda_a_dn2;
        lambda_a_u_set[Cell[4]]=lambda_c_up3;
        lambda_a_d_set[Cell[3]]=lambda_c_dn6;
        lambda_c_d_set[Cell[3]]=lambda_b_dn7;
        lambda_c_u_set[Cell[4]]=lambda_b_up8;
        lambda_b_u_set[Cell[4]]=lambda_a_up9;
        lambda_b_d_set[Cell[5]]=lambda_a_dn10;


        Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
        evol_J2_term4(bond_dim,trun_tol,gate,T_u_set[Cell[4]],T_d_set[Cell[5]],B_c_set[Cell[3]], B_b_set[Cell[5]], B_a_set[Cell[5]],lambda_c_d_set[Cell[3]], lambda_a_u_set[Cell[4]], lambda_c_d_set[Cell[5]],lambda_c_u_set[Cell[4]], lambda_b_u_set[Cell[4]], lambda_b_d_set[Cell[5]], lambda_a_d_set[Cell[5]], lambda_a_u_set[Cell[6]]);
        
        T_u_set[Cell[4]]=Td1;
        T_d_set[Cell[5]]=Tu2;
        B_c_set[Cell[3]]=Bc1;
        B_b_set[Cell[5]]=Bb2;
        B_a_set[Cell[5]]=Ba3;
        lambda_c_d_set[Cell[3]]=lambda_c_up1;
        lambda_a_u_set[Cell[4]]=lambda_a_dn2;
        lambda_c_d_set[Cell[5]]=lambda_c_up3;
        lambda_c_u_set[Cell[4]]=lambda_c_dn6;
        lambda_b_u_set[Cell[4]]=lambda_b_dn7;
        lambda_b_d_set[Cell[5]]=lambda_b_up8;
        lambda_a_d_set[Cell[5]]=lambda_a_up9;
        lambda_a_u_set[Cell[6]]=lambda_a_dn10;

        Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
        evol_J2_term5(bond_dim,trun_tol,gate,T_d_set[Cell[5]],T_u_set[Cell[6]],B_b_set[Cell[5]], B_a_set[Cell[5]], B_c_set[Cell[1]],lambda_b_u_set[Cell[4]], lambda_c_d_set[Cell[5]], lambda_b_u_set[Cell[6]],lambda_b_d_set[Cell[5]], lambda_a_d_set[Cell[5]], lambda_a_u_set[Cell[6]], lambda_c_u_set[Cell[6]], lambda_c_d_set[Cell[1]]);
        
        T_d_set[Cell[5]]=Td1;
        T_u_set[Cell[6]]=Tu2;
        B_b_set[Cell[5]]=Bc1;
        B_a_set[Cell[5]]=Bb2;
        B_c_set[Cell[1]]=Ba3;
        lambda_b_u_set[Cell[4]]=lambda_c_up1;
        lambda_c_d_set[Cell[5]]=lambda_a_dn2;
        lambda_b_u_set[Cell[6]]=lambda_c_up3;
        lambda_b_d_set[Cell[5]]=lambda_c_dn6;
        lambda_a_d_set[Cell[5]]=lambda_b_dn7;
        lambda_a_u_set[Cell[6]]=lambda_b_up8;
        lambda_c_u_set[Cell[6]]=lambda_a_up9;
        lambda_c_d_set[Cell[1]]=lambda_a_dn10;

        Td1,Tu2,Bc1,Bb2,Ba3,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10=
        evol_J2_term6(bond_dim,trun_tol,gate,T_u_set[Cell[6]],T_d_set[Cell[1]],B_a_set[Cell[5]], B_c_set[Cell[1]], B_b_set[Cell[1]],lambda_a_d_set[Cell[5]], lambda_b_u_set[Cell[6]], lambda_a_d_set[Cell[1]],lambda_a_u_set[Cell[6]], lambda_c_u_set[Cell[6]], lambda_c_d_set[Cell[1]], lambda_b_d_set[Cell[1]], lambda_b_u_set[Cell[2]]);
        
        T_u_set[Cell[6]]=Td1;
        T_d_set[Cell[1]]=Tu2;
        B_a_set[Cell[5]]=Bc1;
        B_c_set[Cell[1]]=Bb2;
        B_b_set[Cell[1]]=Ba3;
        lambda_a_d_set[Cell[5]]=lambda_c_up1;
        lambda_b_u_set[Cell[6]]=lambda_a_dn2;
        lambda_a_d_set[Cell[1]]=lambda_c_up3;
        lambda_a_u_set[Cell[6]]=lambda_c_dn6;
        lambda_c_u_set[Cell[6]]=lambda_b_dn7;
        lambda_c_d_set[Cell[1]]=lambda_b_up8;
        lambda_b_d_set[Cell[1]]=lambda_a_up9;
        lambda_b_u_set[Cell[2]]=lambda_a_dn10;
    end
end
function J3_update(bond_dim,trun_tol,gate,Cell_ind, T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set)

    for cc=1:9
        Cell=Cell_ind[cc];
        
        Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
        evol_J3_term1_α(bond_dim,trun_tol,gate, T_d_set[Cell[1]], T_u_set[Cell[2]], T_d_set[Cell[3]], B_c_set[Cell[1]], B_b_set[Cell[1]], B_a_set[Cell[3]], B_c_set[Cell[3]], lambda_c_u_set[Cell[6]], lambda_a_d_set[Cell[1]], lambda_c_u_set[Cell[2]], lambda_b_d_set[Cell[3]], lambda_c_u_set[Cell[4]], lambda_c_d_set[Cell[1]], lambda_b_d_set[Cell[1]], lambda_b_u_set[Cell[2]], lambda_a_u_set[Cell[2]], lambda_a_d_set[Cell[3]], lambda_c_d_set[Cell[3]]);

        T_d_set[Cell[1]]=Td1;
        T_u_set[Cell[2]]=Tu2;
        T_d_set[Cell[3]]=Td3;
        B_c_set[Cell[1]]=Bc1;
        B_b_set[Cell[1]]=Bb2;
        B_a_set[Cell[3]]=Ba3;
        B_c_set[Cell[3]]=Bc4;
        lambda_c_u_set[Cell[6]]=lambda_c_up1;
        lambda_a_d_set[Cell[1]]=lambda_a_dn2;
        lambda_c_u_set[Cell[2]]=lambda_c_up3;
        lambda_b_d_set[Cell[3]]=lambda_b_dn4;
        lambda_c_u_set[Cell[4]]=lambda_c_up5;
        lambda_c_d_set[Cell[1]]=lambda_c_dn6;
        lambda_b_d_set[Cell[1]]=lambda_b_dn7;
        lambda_b_u_set[Cell[2]]=lambda_b_up8;
        lambda_a_u_set[Cell[2]]=lambda_a_up9;
        lambda_a_d_set[Cell[3]]=lambda_a_dn10;
        lambda_c_d_set[Cell[3]]=lambda_c_dn11;


        #################



        Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
        evol_J3_term1_β(bond_dim,trun_tol,gate, T_u_set[Cell[4]], T_d_set[Cell[5]], T_u_set[Cell[6]], B_c_set[Cell[3]], B_b_set[Cell[5]], B_a_set[Cell[5]], B_c_set[Cell[1]], lambda_c_d_set[Cell[3]], lambda_a_u_set[Cell[4]], lambda_c_d_set[Cell[5]], lambda_b_u_set[Cell[6]], lambda_c_d_set[Cell[1]], lambda_c_u_set[Cell[4]], lambda_b_u_set[Cell[4]], lambda_b_d_set[Cell[5]], lambda_a_d_set[Cell[5]], lambda_a_u_set[Cell[6]], lambda_c_u_set[Cell[6]]);

        T_u_set[Cell[4]]=Td1;
        T_d_set[Cell[5]]=Tu2;
        T_u_set[Cell[6]]=Td3;
        B_c_set[Cell[3]]=Bc1;
        B_b_set[Cell[5]]=Bb2;
        B_a_set[Cell[5]]=Ba3;
        B_c_set[Cell[1]]=Bc4;
        lambda_c_d_set[Cell[3]]=lambda_c_up1;
        lambda_a_u_set[Cell[4]]=lambda_a_dn2;
        lambda_c_d_set[Cell[5]]=lambda_c_up3;
        lambda_b_u_set[Cell[6]]=lambda_b_dn4;
        lambda_c_d_set[Cell[1]]=lambda_c_up5;
        lambda_c_u_set[Cell[4]]=lambda_c_dn6;
        lambda_b_u_set[Cell[4]]=lambda_b_dn7;
        lambda_b_d_set[Cell[5]]=lambda_b_up8;
        lambda_a_d_set[Cell[5]]=lambda_a_up9;
        lambda_a_u_set[Cell[6]]=lambda_a_dn10;
        lambda_c_u_set[Cell[6]]=lambda_c_dn11;


        #################

        Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
        evol_J3_term2_α(bond_dim,trun_tol,gate, T_d_set[Cell[5]], T_u_set[Cell[6]], T_d_set[Cell[1]], B_b_set[Cell[5]], B_a_set[Cell[5]], B_c_set[Cell[1]], B_b_set[Cell[1]], lambda_b_u_set[Cell[4]], lambda_c_d_set[Cell[5]], lambda_b_u_set[Cell[6]], lambda_a_d_set[Cell[1]], lambda_b_u_set[Cell[2]], lambda_b_d_set[Cell[5]], lambda_a_d_set[Cell[5]], lambda_a_u_set[Cell[6]], lambda_c_u_set[Cell[6]], lambda_c_d_set[Cell[1]], lambda_b_d_set[Cell[1]]);

        T_d_set[Cell[5]]=Td1;
        T_u_set[Cell[6]]=Tu2;
        T_d_set[Cell[1]]=Td3;
        B_b_set[Cell[5]]=Bc1;
        B_a_set[Cell[5]]=Bb2;
        B_c_set[Cell[1]]=Ba3;
        B_b_set[Cell[1]]=Bc4;
        lambda_b_u_set[Cell[4]]=lambda_c_up1;
        lambda_c_d_set[Cell[5]]=lambda_a_dn2;
        lambda_b_u_set[Cell[6]]=lambda_c_up3;
        lambda_a_d_set[Cell[1]]=lambda_b_dn4;
        lambda_b_u_set[Cell[2]]=lambda_c_up5;
        lambda_b_d_set[Cell[5]]=lambda_c_dn6;
        lambda_a_d_set[Cell[5]]=lambda_b_dn7;
        lambda_a_u_set[Cell[6]]=lambda_b_up8;
        lambda_c_u_set[Cell[6]]=lambda_a_up9;
        lambda_c_d_set[Cell[1]]=lambda_a_dn10;
        lambda_b_d_set[Cell[1]]=lambda_c_dn11;


        #################
        Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
        evol_J3_term2_β(bond_dim,trun_tol,gate, T_u_set[Cell[2]], T_d_set[Cell[3]], T_u_set[Cell[4]], B_b_set[Cell[1]], B_a_set[Cell[3]], B_c_set[Cell[3]], B_b_set[Cell[5]], lambda_b_d_set[Cell[1]], lambda_c_u_set[Cell[2]], lambda_b_d_set[Cell[3]], lambda_a_u_set[Cell[4]], lambda_b_d_set[Cell[5]], lambda_b_u_set[Cell[2]], lambda_a_u_set[Cell[2]], lambda_a_d_set[Cell[3]], lambda_c_d_set[Cell[3]], lambda_c_u_set[Cell[4]], lambda_b_u_set[Cell[4]]);

        T_u_set[Cell[2]]=Td1;
        T_d_set[Cell[3]]=Tu2;
        T_u_set[Cell[4]]=Td3;
        B_b_set[Cell[1]]=Bc1;
        B_a_set[Cell[3]]=Bb2;
        B_c_set[Cell[3]]=Ba3;
        B_b_set[Cell[5]]=Bc4;
        lambda_b_d_set[Cell[1]]=lambda_c_up1;
        lambda_c_u_set[Cell[2]]=lambda_a_dn2;
        lambda_b_d_set[Cell[3]]=lambda_c_up3;
        lambda_a_u_set[Cell[4]]=lambda_b_dn4;
        lambda_b_d_set[Cell[5]]=lambda_c_up5;
        lambda_b_u_set[Cell[2]]=lambda_c_dn6;
        lambda_a_u_set[Cell[2]]=lambda_b_dn7;
        lambda_a_d_set[Cell[3]]=lambda_b_up8;
        lambda_c_d_set[Cell[3]]=lambda_a_up9;
        lambda_c_u_set[Cell[4]]=lambda_a_dn10;
        lambda_b_u_set[Cell[4]]=lambda_c_dn11;



        #################
        Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
        evol_J3_term3_α(bond_dim,trun_tol,gate, T_d_set[Cell[3]], T_u_set[Cell[4]], T_d_set[Cell[5]], B_a_set[Cell[3]], B_c_set[Cell[3]], B_b_set[Cell[5]], B_a_set[Cell[5]], lambda_a_u_set[Cell[2]], lambda_b_d_set[Cell[3]], lambda_a_u_set[Cell[4]], lambda_c_d_set[Cell[5]], lambda_a_u_set[Cell[6]], lambda_a_d_set[Cell[3]], lambda_c_d_set[Cell[3]], lambda_c_u_set[Cell[4]], lambda_b_u_set[Cell[4]], lambda_b_d_set[Cell[5]], lambda_a_d_set[Cell[5]]);

        T_d_set[Cell[3]]=Td1;
        T_u_set[Cell[4]]=Tu2;
        T_d_set[Cell[5]]=Td3;
        B_a_set[Cell[3]]=Bc1;
        B_c_set[Cell[3]]=Bb2;
        B_b_set[Cell[5]]=Ba3;
        B_a_set[Cell[5]]=Bc4;
        lambda_a_u_set[Cell[2]]=lambda_c_up1;
        lambda_b_d_set[Cell[3]]=lambda_a_dn2;
        lambda_a_u_set[Cell[4]]=lambda_c_up3;
        lambda_c_d_set[Cell[5]]=lambda_b_dn4;
        lambda_a_u_set[Cell[6]]=lambda_c_up5;
        lambda_a_d_set[Cell[3]]=lambda_c_dn6;
        lambda_c_d_set[Cell[3]]=lambda_b_dn7;
        lambda_c_u_set[Cell[4]]=lambda_b_up8;
        lambda_b_u_set[Cell[4]]=lambda_a_up9;
        lambda_b_d_set[Cell[5]]=lambda_a_dn10;
        lambda_a_d_set[Cell[5]]=lambda_c_dn11;



        #################
        Td1,Tu2,Td3,Bc1,Bb2,Ba3,Bc4,lambda_c_up1,lambda_a_dn2,lambda_c_up3,lambda_b_dn4,lambda_c_up5,lambda_c_dn6,lambda_b_dn7,lambda_b_up8,lambda_a_up9,lambda_a_dn10,lambda_c_dn11=
        evol_J3_term3_β(bond_dim,trun_tol,gate, T_u_set[Cell[6]], T_d_set[Cell[1]], T_u_set[Cell[2]], B_a_set[Cell[5]], B_c_set[Cell[1]], B_b_set[Cell[1]], B_a_set[Cell[3]], lambda_a_d_set[Cell[5]], lambda_b_u_set[Cell[6]], lambda_a_d_set[Cell[1]], lambda_c_u_set[Cell[2]], lambda_a_d_set[Cell[3]], lambda_a_u_set[Cell[6]], lambda_c_u_set[Cell[6]], lambda_c_d_set[Cell[1]], lambda_b_d_set[Cell[1]], lambda_b_u_set[Cell[2]], lambda_a_u_set[Cell[2]]);

        T_u_set[Cell[6]]=Td1;
        T_d_set[Cell[1]]=Tu2;
        T_u_set[Cell[2]]=Td3;
        B_a_set[Cell[5]]=Bc1;
        B_c_set[Cell[1]]=Bb2;
        B_b_set[Cell[1]]=Ba3;
        B_a_set[Cell[3]]=Bc4;
        lambda_a_d_set[Cell[5]]=lambda_c_up1;
        lambda_b_u_set[Cell[6]]=lambda_a_dn2;
        lambda_a_d_set[Cell[1]]=lambda_c_up3;
        lambda_c_u_set[Cell[2]]=lambda_b_dn4;
        lambda_a_d_set[Cell[3]]=lambda_c_up5;
        lambda_a_u_set[Cell[6]]=lambda_c_dn6;
        lambda_c_u_set[Cell[6]]=lambda_b_dn7;
        lambda_c_d_set[Cell[1]]=lambda_b_up8;
        lambda_b_d_set[Cell[1]]=lambda_a_up9;
        lambda_b_u_set[Cell[2]]=lambda_a_dn10;
        lambda_a_u_set[Cell[2]]=lambda_c_dn11;


    end

end


function itebd_step(T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set, gate_triangle, gate_Heisenberg_J2, gate_half_Heisenberg_J3, Cell_ind, T_d_envs, T_u_envs, trun_tol, bond_dim)
    triangle_update(T_d_envs, T_u_envs, bond_dim,trun_tol,gate_triangle,Cell_ind, T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set)

    J2_update(bond_dim,trun_tol,gate_Heisenberg_J2,Cell_ind, T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set)
    J3_update(bond_dim,trun_tol,gate_half_Heisenberg_J3,Cell_ind, T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set)
    
    return T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set
end

function itebd(T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set, H_triangle, H_Heisenberg, J2, J3, trun_tol, tau, dt, bond_dim, Cell_ind, T_d_envs,T_u_envs)
    gate_triangle, gate_half_triangle=trotter_gate(H_triangle,dt);
    gate_Heisenberg_J2, gate_half_Heisenberg_J2=trotter_gate(H_Heisenberg*J2,dt);
    gate_Heisenberg_J3, gate_half_Heisenberg_J3=trotter_gate(H_Heisenberg*J3,dt);
    for cs=1:Int(round(tau/dt))
        println("dt step: "*string(cs));flush(stdout);
        T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set=itebd_step(T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set, gate_triangle, gate_Heisenberg_J2, gate_half_Heisenberg_J3, Cell_ind, T_d_envs, T_u_envs, trun_tol, bond_dim);
    end
    return T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set
end