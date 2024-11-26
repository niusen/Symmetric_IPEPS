"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

function torus_to_cylinder_xpbc(psi)
    psi=deepcopy(psi);
    Lx=size(psi,1);
    Ly=size(psi,2);

    for cx=1:Lx #convert to cylinder where x direction is PBC
        for cy=2:Ly-1
            Vy=space(psi[cx,Ly],4);
            A=psi[cx,cy];
            Uytop=@ignore_derivatives unitary(space(A,4)'*Vy, fuse(space(A,4)'*Vy));#the PBC link is below
            Uybot=@ignore_derivatives unitary(fuse(space(A,2)*Vy),space(A,2)*Vy);#the PBC link is below
            Iy=@ignore_derivatives unitary(Vy,Vy);

            @tensor A_new[:]:=A[-1,3,-3,1,-5]*Iy[4,2]*Uybot[-2,3,4]*Uytop[1,2,-4];
            psi[cx,cy]=A_new
        end

        Vy=space(psi[cx,Ly],4);
        Atop=psi[cx,Ly];
        Uybot=@ignore_derivatives unitary(fuse(space(Atop,2)*Vy),space(Atop,2)*Vy);#the PBC link is below
        @tensor Atop[:]:=Atop[-1,1,-3,2,-4]*Uybot[-2,1,2];
        Abot=psi[cx,1];
        Uytop=@ignore_derivatives unitary(space(Abot,4)'*Vy, fuse(space(Abot,4)'*Vy));#the PBC link is below
        @tensor Abot[:]:=Abot[-1,2,-2,1,-4]*Uytop[1,2,-3];
        psi[cx,Ly]=Atop;
        psi[cx,1]=Abot;

    end

    return psi

end

function cylinder_xpbc_to_disk(psi)
    psi=deepcopy(psi);
    Lx=size(psi,1);
    Ly=size(psi,2);

    for cy=2:Ly-1
        for cx=2:Lx-1 
            Vx=space(psi[1,cy],1);
            A=psi[cx,cy];
            Uxright=@ignore_derivatives unitary(fuse(space(A,3)*Vx),space(A,3)*Vx);#the PBC link is below
            Uxleft=@ignore_derivatives unitary(space(A,1)'*Vx,fuse(space(A,1)'*Vx));#the PBC link is below
            Ix=@ignore_derivatives unitary(Vx,Vx);
            @tensor A_new[:]:=A[1,-2,3,-4,-5]*Ix[4,2]*Uxright[-3,3,4]*Uxleft[1,2,-1];
            psi[cx,cy]=A_new
        end

        Vx=space(psi[1,cy],1);
        Aleft=psi[1,cy];
        Uxright=@ignore_derivatives unitary(fuse(space(Aleft,3)*Vx),space(Aleft,3)*Vx);#the PBC link is below
        @tensor Aleft[:]:=Aleft[2,-1,1,-3,-4]*Uxright[-2,1,2];
        Aright=psi[Lx,cy];
        Uxleft=@ignore_derivatives unitary(space(Aright,1)'*Vx, fuse(space(Aright,1)'*Vx));#the PBC link is below
        @tensor Aright[:]:=Aright[1,-2,2,-3,-4]*Uxleft[1,2,-1];
        psi[1,cy]=Aleft;
        psi[Lx,cy]=Aright;
    end

    ##################
    cy=1;
    for cx=2:Lx-1 
        Vx=space(psi[1,cy],1);
        A=psi[cx,cy];
        Uxright=@ignore_derivatives unitary(fuse(space(A,2)*Vx),space(A,2)*Vx);#the PBC link is below
        Uxleft=@ignore_derivatives unitary(space(A,1)'*Vx,fuse(space(A,1)'*Vx));#the PBC link is below
        Ix=@ignore_derivatives unitary(Vx,Vx);
        @tensor A_new[:]:=A[1,3,-4,-5]*Ix[4,2]*Uxright[-3,3,4]*Uxleft[1,2,-1];
        psi[cx,cy]=A_new
    end

    Vx=space(psi[1,cy],1);
    Aleft=psi[1,cy];
    Uxright=@ignore_derivatives unitary(fuse(space(Aleft,2)*Vx),space(Aleft,2)*Vx);#the PBC link is below
    @tensor Aleft[:]:=Aleft[2,1,-3,-4]*Uxright[-2,1,2];
    Aright=psi[Lx,cy];
    Uxleft=@ignore_derivatives unitary(space(Aright,1)'*Vx,fuse(space(Aright,1)'*Vx));#the PBC link is below
    @tensor Aright[:]:=Aright[1,2,-3,-4]*Uxleft[1,2,-1];
    psi[1,cy]=Aleft;
    psi[Lx,cy]=Aright;

    ##################
    cy=Ly;
    for cx=2:Lx-1 
        Vx=space(psi[1,cy],1);
        A=psi[cx,cy];
        Uxright=@ignore_derivatives unitary(fuse(space(A,3)*Vx),space(A,3)*Vx);#the PBC link is below
        Uxleft=@ignore_derivatives unitary(space(A,1)'*Vx,fuse(space(A,1)'*Vx));#the PBC link is below
        Ix=@ignore_derivatives unitary(Vx,Vx);
        @tensor A_new[:]:=A[1,-2,3,-5]*Ix[4,2]*Uxright[-3,3,4]*Uxleft[1,2,-1];
        psi[cx,cy]=A_new
    end

    Vx=space(psi[1,cy],1);
    Aleft=psi[1,cy];
    Uxright=@ignore_derivatives unitary(fuse(space(Aleft,3)*Vx),space(Aleft,3)*Vx);#the PBC link is below
    @tensor Aleft[:]:=Aleft[2,-1,1,-4]*Uxright[-2,1,2];
    Aright=psi[Lx,cy];
    Uxleft=@ignore_derivatives unitary(space(Aright,1)'*Vx,fuse(space(Aright,1)'*Vx));#the PBC link is below
    @tensor Aright[:]:=Aright[1,-2,2,-4]*Uxleft[1,2,-1];
    psi[1,cy]=Aleft;
    psi[Lx,cy]=Aright;

    return psi

end


function disk_to_torus(psi)
    psi=deepcopy(psi);
    Lx,Ly=size(psi);
    if sectortype(space(psi[1,1],1)) == Trivial
        Vtrivial=(ℂ^1);
    else
        if isa(space(psi[1],1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
            Vtrivial=Rep[SU₂](0=>1);
        elseif isa(space(psi[1],1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
            Vtrivial=Rep[U₁](0=>1);
        end
    end
    Vl=Vtrivial;
    Vd=Vl;
    Vr=Vl';
    Vu=Vl';

    #left
    ca=1;
    for cb=2:Ly-1
        T=psi[ca,cb];
        uu=@ignore_derivatives unitary(Vl*space(T,1),space(T,1));
        @tensor T[:]:=uu[-1,-2,1]*T[1,-3,-4,-5];
        psi=matrix_update(psi,ca,cb,T);
    end

    #right
    ca=Lx;
    for cb=2:Ly-1
        T=psi[ca,cb];
        uu=@ignore_derivatives unitary(Vr*space(T,3),space(T,3));
        @tensor T[:]:=uu[-3,-4,1]*T[-1,-2,1,-5];
        psi=matrix_update(psi,ca,cb,T);
    end

    #top
    for ca=2:Lx-1
        cb=Ly;
        T=psi[ca,cb];
        uu=@ignore_derivatives unitary(space(T,3)*Vu,space(T,3));
        @tensor T[:]:=uu[-3,-4,1]*T[-1,-2,1,-5];
        psi=matrix_update(psi,ca,cb,T);
    end

    #bot
    for ca=2:Lx-1
        cb=1;
        T=psi[ca,cb];
        uu=@ignore_derivatives unitary(space(T,1)*Vd,space(T,1));
        @tensor T[:]:=uu[-1,-2,1]*T[1,-3,-4,-5];
        psi=matrix_update(psi,ca,cb,T);
    end

    #left_top
    ca=1;
    cb=Ly;
    T=psi[ca,cb];
    uu=@ignore_derivatives unitary(Vl*Vu*space(T,1),space(T,1));
    @tensor T[:]:=uu[-1,-4,-2,1]*T[1,-3,-5];
    psi=matrix_update(psi,ca,cb,T);

    #left_bot
    ca=1;
    cb=1;
    T=psi[ca,cb];
    uu=@ignore_derivatives unitary(Vl*Vd*space(T,1),space(T,1));
    @tensor T[:]:=uu[-1,-2,-3,1]*T[1,-4,-5];
    psi=matrix_update(psi,ca,cb,T);

    #right_top
    ca=Lx;
    cb=Ly;
    T=psi[ca,cb];
    uu=@ignore_derivatives unitary(space(T,2)*Vr*Vu,space(T,2));
    @tensor T[:]:=uu[-2,-3,-4,1]*T[-1,1,-5];
    psi=matrix_update(psi,ca,cb,T);

    #right_bot
    ca=Lx;
    cb=1;
    T=psi[ca,cb];
    uu=@ignore_derivatives unitary(space(T,1)*Vd*Vr,space(T,1));
    @tensor T[:]:=uu[-1,-2,-3,1]*T[1,-4,-5];
    psi=matrix_update(psi,ca,cb,T);


    return psi
end



function remove_trivial_boundary_leg(psi)
    psi=deepcopy(psi);
    Lx,Ly=size(psi);

    #left
    ca=1;
    for cb=2:Ly-1
        T=psi[ca,cb];
        uu=@ignore_derivatives unitary(space(T,2),space(T,1)*space(T,2));
        @tensor T[:]:=uu[-1,1,2]*T[1,2,-2,-3,-4];
        psi=matrix_update(psi,ca,cb,T);
    end

    #right
    ca=Lx;
    for cb=2:Ly-1
        T=psi[ca,cb];
        uu=@ignore_derivatives unitary(space(T,4),space(T,3)*space(T,4));
        @tensor T[:]:=uu[-3,1,2]*T[-1,-2,1,2,-4];
        psi=matrix_update(psi,ca,cb,T);
    end

    #top
    for ca=2:Lx-1
        cb=Ly;
        T=psi[ca,cb];
        uu=@ignore_derivatives unitary(space(T,3),space(T,3)*space(T,4));
        @tensor T[:]:=uu[-3,1,2]*T[-1,-2,1,2,-4];
        psi=matrix_update(psi,ca,cb,T);
    end

    #bot
    for ca=2:Lx-1
        cb=1;
        T=psi[ca,cb];
        uu=@ignore_derivatives unitary(space(T,1),space(T,1)*space(T,2));
        @tensor T[:]:=uu[-1,1,2]*T[1,2,-2,-3,-4];
        psi=matrix_update(psi,ca,cb,T);
    end

    #left_top
    ca=1;
    cb=Ly;
    T=psi[ca,cb];
    uu=@ignore_derivatives unitary(space(T,2),space(T,1)*space(T,2)*space(T,4));
    @tensor T[:]:=uu[-1,1,2,3]*T[1,2,-2,3,-3];
    psi=matrix_update(psi,ca,cb,T);

    #left_bot
    ca=1;
    cb=1;
    T=psi[ca,cb];
    uu=@ignore_derivatives unitary(space(T,3),space(T,1)*space(T,2)*space(T,3));
    @tensor T[:]:=uu[-1,1,2,3]*T[1,2,3,-2,-3];
    psi=matrix_update(psi,ca,cb,T);

    #right_top
    ca=Lx;
    cb=Ly;
    T=psi[ca,cb];
    uu=@ignore_derivatives unitary(space(T,2),space(T,2)*space(T,3)*space(T,4));
    @tensor T[:]:=uu[-2,1,2,3]*T[-1,1,2,3,-3];
    psi=matrix_update(psi,ca,cb,T);

    #right_bot
    ca=Lx;
    cb=1;
    T=psi[ca,cb];
    uu=@ignore_derivatives unitary(space(T,1),space(T,1)*space(T,2)*space(T,3));
    @tensor T[:]:=uu[-1,1,2,3]*T[1,2,3,-2,-3];
    psi=matrix_update(psi,ca,cb,T);


    return psi
end