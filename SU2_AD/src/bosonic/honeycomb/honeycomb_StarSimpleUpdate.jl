# Four-site star SU. The HOSVD is the kagome construction: three unfoldings
# of ONE tensor, followed by the three projectors together. No core re-HOSVD.

function honeycomb_star_geometry(Lx,Ly,center)
    x,y=center;
    legs=iseven(x+y) ? (1,2,3) : (1,3,4);
    neighbor(leg)=leg==1 ? (mod1(x-1,Lx),y) : leg==3 ? (mod1(x+1,Lx),y) :
        leg==2 ? (x,mod1(y-1,Ly)) : (x,mod1(y+1,Ly));
    leaves=map(neighbor,legs);inner=map(l->(3,4,1,2)[l],legs);
    length(unique((center,leaves...)))==4 || throw(ArgumentError("a star needs four distinct cell entries; use a 4x4 cell"));
    outer=map(leaves,inner) do site,leg
        cyclic=iseven(sum(site)) ? (1,2,3) : (1,3,4);
        k=findfirst(==(leg),cyclic);
        (cyclic[mod1(k+1,3)],cyclic[mod1(k+2,3)])
    end;
    return (;center,legs,leaves,inner,outer)
end

"""Four disjoint star-covering layers, checked on the actual periodic graph."""
function honeycomb_star_groups(Lx,Ly)
    Lx>=4 && Ly>=4 && Lx%4==0 && Ly%4==0 ||
        throw(ArgumentError("star HOSVD currently requires Lx,Ly divisible by 4; use method=:two_site for the old kernel"));
    groups=[Tuple{Int,Int}[] for _=1:4];
    for x=1:Lx,y=1:Ly
        color=isodd(x) ? xor(div(x-1,2)%2,div(y-1,2)%2) :
            2+xor(div(x-2,2)%2,(y-1)%2);
        push!(groups[color+1],(x,y));
    end
    for group in groups
        occupied=Set{Tuple{Int,Int}}();
        for center in group
            g=honeycomb_star_geometry(Lx,Ly,center);
            for site in (center,g.leaves...)
                site in occupied && error("overlapping stars in one HOSVD layer");
                push!(occupied,site);
            end
        end
        length(occupied)==Lx*Ly || error("star layer does not cover the whole cell");
    end
    return groups
end

function _star_mode_product(A,M,leg)
    n=numind(A);rest=Tuple(k for k=1:n if k!=leg);order=(leg,rest...);
    B=M*permute(A,((leg,),rest));
    return permute(B,Tuple(invperm(collect(order))))
end

function _star_weight(lx,ly,site,leg)
    x,y=site;nx,ny=size(lx);
    L=leg==1 ? lx[x,y] : leg==2 ? ly[x,y] :
        leg==3 ? lx[mod1(x+1,nx),y] : ly[x,mod1(y+1,ny)];
    # Stored horizontal weights point into L, vertical weights into U.
    return leg in (1,4) ? L : permute(L,((2,),(1,)))
end

function _star_strip(A,order)
    unused=only(setdiff(1:5,order));V=space(A,unused);
    dim(V)==1 || throw(ArgumentError("unused brick-wall leg must have dimension one"));
    e=TensorMap(ones(eltype(A),1),V',one(V*V));
    M=permute(A,(order...,unused));
    @tensor B[:]:=M[-1,-2,-3,-4,a]*e[a];
    return B
end

function _star_embed(B,reference,order)
    unused=only(setdiff(1:5,order));V=space(reference,unused);
    e=TensorMap(ones(eltype(B),1),V,one(V*V));
    @tensor M[:]:=B[-1,-2,-3,-4]*e[-5];
    return permute(M,Tuple(invperm(collect((order...,unused)))))
end

"""h_center=J/2 sum_i(P_center,i-shift); every bond occurs in two stars."""
function honeycomb_star_hamiltonian(Vp;J=1.,convention=:traceless)
    convention in (:permutation,:traceless) || throw(ArgumentError("invalid Hamiltonian convention"));
    dim(Vp)==4 || throw(ArgumentError("physical dimension must be four"));
    V=Vp*Vp*Vp*Vp;I4=unitary(V,V);H=0.0*I4;
    for i=2:4
        order=collect(1:4);order[1],order[i]=order[i],order[1];
        H+=permute(I4,(Tuple(order),(5,6,7,8)))-(convention==:traceless ? I4/4 : 0*I4);
    end
    # SU(4) recoupling can leave roundoff in theoretically zero blocks.
    # Make every block exactly Hermitian before TensorKit's blockwise eigh.
    return (J/4)*(H+H')
end

function _star_prepare(T,lx,ly,geometry)
    g=geometry;center=T[g.center...];
    A0=_star_strip(center,(g.legs...,5));
    residuals=Any[];keeps=Any[];weights=Any[];orders=Any[];
    for i=1:3
        site=g.leaves[i];outer=g.outer[i];order=(outer...,5,g.inner[i]);
        A=_star_strip(T[site...],order);
        W=map(leg->_star_weight(lx,ly,site,leg),outer);
        # Absorb BEFORE the exact outer-leg reduction, undo AFTER rebuilding.
        A=_star_mode_product(_star_mode_product(A,W[1],1),W[2],2);
        # Remove exactly zero auxiliary singular values only. This is an
        # exact support reduction, with no finite-D or numerical cutoff.
        zero_cutoff=nextfloat(real(zero(eltype(A))));
        q,s,v=tsvd(permute(A,((1,2),(3,4)));trunc=truncbelow(zero_cutoff));
        _require_honeycomb_lambda(s,"star outer-leg exact support reduction");
        push!(residuals,permute(q,(1,2,3)));
        push!(keeps,permute(s*v,(1,2,3)));
        push!(weights,W);push!(orders,order);
    end
    K1,K2,K3=keeps;
    @tensor theta[:]:=A0[a,b,c,-1]*K1[-2,-3,a]*K2[-4,-5,b]*K3[-6,-7,c];
    return theta,(;residuals,weights,orders)
end

function _star_apply_gate(theta,gate)
    isnothing(gate) && return theta;
    @tensor updated[:]:=theta[a,-2,b,-4,c,-6,d]*gate[-1,-3,-5,-7,a,b,c,d];
    return updated
end

function _star_mode_svd(theta,left,Dmax,trun_tol;audit=nothing,mode=0)
    right=Tuple(k for k=1:7 if !(k in left));
    isnothing(audit) || audit((stage=:mode_full,mode=mode,matrix=permute(theta,(left,right))));
    U,S,_=isnothing(Dmax) ? tsvd(permute(theta,(left,right))) :
        tsvd(permute(theta,(left,right));trunc=truncdim(Dmax));
    info=_require_honeycomb_lambda(S,"star HOSVD mode $left spectrum");
    isnothing(audit) || audit((stage=:dimension_cut,mode=mode,S=S,cutoff=trun_tol*info.max_abs));
    if trun_tol>0
        # Same role as kagome's Truncations: remove numerical zero singular
        # blocks and normalize S. This is an S-only cleanup, not a core SVD.
        a,s,_=tsvd(S;trunc=truncbelow(trun_tol*info.max_abs));
        _require_honeycomb_lambda(s,"filtered star HOSVD mode spectrum");
        U=U*a;S=s;
    end
    isnothing(audit) || audit((stage=:tolerance_cut,mode=mode,S=S,cutoff=trun_tol*info.max_abs));
    return U,S/norm(S)
end

"""Kagome-style simultaneous HOSVD, with an untruncated central physical leg.
theta=(p0,r1,p1,r2,p2,r3,p3). Each U comes from the SAME normalized theta.
"""
function honeycomb_star_hosvd(theta,Dmax;trun_tol=1e-10,audit=nothing)
    n=norm(theta);isfinite(n) && n>0 || error("zero/nonfinite star wavefunction before HOSVD");
    theta=theta/n;
    U1,S1=_star_mode_svd(theta,(2,3),Dmax,trun_tol;audit,mode=1);
    U2,S2=_star_mode_svd(theta,(4,5),Dmax,trun_tol;audit,mode=2);
    U3,S3=_star_mode_svd(theta,(6,7),Dmax,trun_tol;audit,mode=3);
    @tensor core[:]:=theta[-1,a,b,c,d,e,f]*U1'[-2,a,b]*U2'[-3,c,d]*U3'[-4,e,f];
    retained=norm(core)^2;
    retained>0 && isfinite(retained) || error("zero/nonfinite projected HOSVD core");
    # As in kagome hosvd(), the core keeps the mode weights and is normalized.
    core=core/norm(core);
    return core,(U1,U2,U3),(S1,S2,S3),retained
end

function _star_reassemble(T,lx,ly,g,prepared,core,U,S)
    roots=map(s->sqrt(s),S);C=core;
    for i=1:3;C=_star_mode_product(C,pinv(roots[i]),i+1);end
    center=_star_embed(permute(C,(2,3,4,1)),T[g.center...],(g.legs...,5));
    leaves=Any[];
    for i=1:3
        Q=prepared.residuals[i];K=U[i]*roots[i];
        @tensor B[:]:=Q[-1,-2,a]*K[a,-3,-4];
        W=prepared.weights[i];
        B=_star_mode_product(_star_mode_product(B,pinv(W[1]),1),pinv(W[2]),2);
        push!(leaves,_star_embed(B,T[g.leaves[i]...],prepared.orders[i]));
    end
    # No state mutation until every new tensor and weight has been validated.
    for A in (center,leaves...)
        n=norm(A);isfinite(n) && n>0 || error("zero/nonfinite reconstructed star site");
    end
    for L in roots;_require_honeycomb_lambda(L,"new star weight");end
    for i=1:3
        space(center,g.legs[i])'==space(leaves[i],g.inner[i]) ||
            throw(SpaceMismatch("reconstructed star bond dual spaces do not match"));
        space(roots[i],1)==space(center,g.legs[i]) ||
            throw(SpaceMismatch("new star weight does not match its center leg"));
    end
    return center,leaves,roots
end

function _star_store_weight!(lx,ly,site,leg,R)
    x,y=site;nx,ny=size(lx);
    stored=leg in (1,4) ? R : permute(R,((2,),(1,)));
    if leg==1;lx[x,y]=stored;
    elseif leg==2;ly[x,y]=stored;
    elseif leg==3;lx[mod1(x+1,nx),y]=stored;
    else;ly[x,mod1(y+1,ny)]=stored;
    end
end

function honeycomb_star_update!(T,lx,ly,gate,center,Dmax;trun_tol=1e-10,audit=nothing)
    g=honeycomb_star_geometry(size(T)...,center);
    isnothing(audit) || audit((stage=:input,T=T,lx=lx,ly=ly,geometry=g));
    theta,prepared=_star_prepare(T,lx,ly,g);
    isnothing(audit) || audit((stage=:before_gate,theta=theta,geometry=g));
    theta=_star_apply_gate(theta,gate);
    core,U,S,retained=honeycomb_star_hosvd(theta,Dmax;trun_tol,audit);
    C,B,R=_star_reassemble(T,lx,ly,g,prepared,core,U,S);
    T[g.center...]=C/norm(C);
    for i=1:3
        T[g.leaves[i]...]=B[i]/norm(B[i]);
        _star_store_weight!(lx,ly,g.center,g.legs[i],R[i]);
    end
    isnothing(audit) || audit((stage=:output,T=T,lx=lx,ly=ly,geometry=g,retained=retained));
    return (center=center,retained_norm_squared=retained,
        dimensions=map(r->dim(space(r,1)),R))
end

function itebd_honeycomb_star(parameters,T,lx,ly,tau,dt,Dmax;
        convention=:traceless,rotate_order=true,verbose=true,step_offset=0,trun_tol=1e-10,observer=nothing,audit=nothing)
    dt>0 && tau>=0 && Dmax>=1 && trun_tol>=0 || throw(ArgumentError("invalid star SU controls"));
    steps=round(Int,tau/dt);
    isapprox(steps*dt,tau;atol=1e-12,rtol=1e-10) || throw(ArgumentError("tau must be an integer multiple of dt"));
    groups=honeycomb_star_groups(size(T)...);_check_honeycomb_state(T,lx,ly);
    if iszero(parameters["J"])
        gate=nothing;half=nothing;
    else
        H=honeycomb_star_hamiltonian(space(T[1],5);J=parameters["J"],convention);
        gate,half=trotter_gate(H,dt);
    end
    for ct=1:steps
        offset=rotate_order ? mod(ct+step_offset-1,4) : 0;
        a,b,c,d=ntuple(k->mod1(k+offset,4),4);
        for (layer,(group,G)) in enumerate(((a,half),(b,half),(c,half),(d,gate),(c,half),(b,half),(a,half)))
            for center in groups[group]
                try
                    local_audit=isnothing(audit) ? nothing : event->audit(merge(event,
                        (sweep=ct+step_offset,layer=layer,group=group,center=center)));
                    report=honeycomb_star_update!(T,lx,ly,G,center,Dmax;trun_tol,audit=local_audit);
                    isnothing(observer) || observer(merge(report,(sweep=ct+step_offset,layer=layer,group=group)));
                catch err
                    error("honeycomb star HOSVD failed at sweep=$(ct+step_offset), layer=$layer, center=$center: "*sprint(showerror,err,catch_backtrace()));
                end
            end
        end
        if verbose && (ct==1 || ct%20==0 || ct==steps)
            println("honeycomb star HOSVD: iteration $ct/$steps, dt=$dt, Dmax=$Dmax");flush(stdout);
        end
    end
    return T,lx,ly
end
