function generate_obc_from_iPEPS(A,Lx,Ly)

    
    P=zeros(1,3);P[1,1]=1;
    # P_L=TensorMap(P,Rep[SU₂](0=>1),space(A,1));
    # P_D=TensorMap(P,Rep[SU₂](0=>1),space(A,2));
    P_L=TensorMap(P,Rep[U₁](0=>1),space(A,1));
    P_D=TensorMap(P,Rep[U₁](0=>1),space(A,2));

    psi=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
    for cx=2:Lx-1
        for cy=2:Ly-1
            psi[cx,cy]=A;
        end
    end

    cx=1;
    for cy=2:Ly-1
        @tensor T[:]:=A[1,-2,-3,-4,-5]*P_L[-1,1];
        psi[cx,cy]=T;
    end

    cx=Lx;
    for cy=2:Ly-1
        @tensor T[:]:=A[-1,-2,1,-4,-5]*P_L'[1,-3];
        psi[cx,cy]=T;
    end

    cy=1;
    for cx=2:Lx-1
        @tensor T[:]:=A[-1,1,-3,-4,-5]*P_D[-2,1];
        psi[cx,cy]=T;
    end

    cy=Ly;
    for cx=2:Lx-1
        @tensor T[:]:=A[-1,-2,-3,1,-5]*P_D'[1,-4];
        psi[cx,cy]=T;
    end

    cx=1;
    cy=1;
    @tensor T[:]:=A[1,2,-3,-4,-5]*P_L[-1,1]*P_D[-2,2];
    psi[cx,cy]=T;

    cx=Lx;
    cy=1;
    @tensor T[:]:=A[-1,2,1,-4,-5]*P_L'[1,-3]*P_D[-2,2];
    psi[cx,cy]=T;

    cx=1;
    cy=Ly;
    @tensor T[:]:=A[1,-2,-3,2,-5]*P_L[-1,1]*P_D'[2,-4];
    psi[cx,cy]=T;

    cx=Lx;
    cy=Ly;
    @tensor T[:]:=A[-1,-2,1,2,-5]*P_L'[1,-3]*P_D'[2,-4];
    psi[cx,cy]=T;

    return psi
end

function apply_sampling_projector(fPEPS,config)
    fPEPS=deepcopy(fPEPS);
    Lx,Ly=size(fPEPS);
    Vp=U₁Space(1/2=>1,-1/2=>1);
    Vup=U₁Space(1/2=>1);
    Vdn=U₁Space(-1/2=>1);
    Pup=TensorMap([1,0],Vup',Vp');
    Pdn=TensorMap([0,1],Vdn',Vp');

    for cx=1:Lx
        for cy=1:Ly
            T=fPEPS[cx,cy];
            if config[cx,cy]==1
                @tensor T[:]:=T[-1,-2,-3,-4,1]*Pup[-5,1];
            elseif config[cx,cy]==-1
                @tensor T[:]:=T[-1,-2,-3,-4,1]*Pdn[-5,1];
            end
            fPEPS[cx,cy]=T;
        end
    end
    return fPEPS
end