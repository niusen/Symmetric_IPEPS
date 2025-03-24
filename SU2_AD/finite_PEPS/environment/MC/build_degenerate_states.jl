function build_projector(Vv)
    if Vv==U₁Space(0=>1,1/2=>1,-1/2=>1)
        op=zeros(3,3);
        op[1,1]=1;
        op[2,2]=-1;
        op[3,3]=-1;
    elseif Vv==Rep[SU₂](0=>1, 1/2=>1)
        op=zeros(3,3);
        op[1,1]=1;
        op[2,2]=-1;
        op[3,3]=-1;
    elseif Vv==ℂ^3
        op=zeros(3,3);
        op[1,1]=1;
        op[2,2]=-1;
        op[3,3]=-1;
    end
    op=TensorMap(op,Vv,Vv);
    return op
end

function construct_4_states(psi,Vv)
    op=build_projector(Vv);
    psi_00=deepcopy(psi);
    psi_0pi=deepcopy(psi);
    psi_pi0=deepcopy(psi);
    psi_pipi=deepcopy(psi);
    Lx,Ly=size(psi);

    cx=1;
    for cy=1:Ly 
        T=psi_pi0[cx,cy];
        @tensor T[:]:=T[1,-2,-3,-4,-5]*op[-1,1];
        psi_pi0[cx,cy]=T;

        T=psi_pipi[cx,cy];
        @tensor T[:]:=T[1,-2,-3,-4,-5]*op[-1,1];
        psi_pipi[cx,cy]=T;
    end

    cy=1;
    for cx=1:Lx 
        T=psi_0pi[cx,cy];
        @tensor T[:]:=T[-1,1,-3,-4,-5]*op[-2,1];
        psi_0pi[cx,cy]=T;

        T=psi_pipi[cx,cy];
        @tensor T[:]:=T[-1,1,-3,-4,-5]*op[-2,1];
        psi_pipi[cx,cy]=T;
    end

    return psi_00,psi_0pi,psi_pi0,psi_pipi 
end