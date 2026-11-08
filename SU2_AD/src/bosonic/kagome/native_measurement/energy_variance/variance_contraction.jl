tuple_grid(a)=ntuple(x->ntuple(y->a[x,y],size(a,2)),size(a,1))
function set_dimensions(Lx,Ly)
    Core.eval(C,:(Lx=$Lx));Core.eval(C,:(Ly=$Ly))
end

function lift_edge(T,J)
    for U in (J,N.conj_tensor(J))
        if space(T,2)==space(U,2)'
            @tensor value[:] := T[-1,1,-3]*U[-2,1]
            return value
        end
    end
    error("mixed-edge warm-start spaces do not match")
end

function base_context(f,env,L)
    g=f.state.geometry
    qk=Matrix{Any}(undef,L,L);rk=similar(qk);uq=similar(qk);ur=similar(qk)
    qb=similar(qk);rb=similar(qk);cs=similar(qk);ts=similar(qk)
    fusers=[N.factor_fusers(f,r)[1] for r in eachindex(g.representatives)]
    for x in 1:L,y in 1:L
        r=N.owner(g,x,y);qk[x,y]=f.Q[r];rk[x,y]=f.R[r];uq[x,y]=fusers[r]
        cs[x,y]=env.Cset[mod1(x,g.Lx)][mod1(y,g.Ly)]
        ts[x,y]=env.Tset[mod1(x,g.Lx)][mod1(y,g.Ly)]
    end
    for x in 1:L,y in 1:L
        ur[x,y]=(N.conj_tensor(uq[x,mod1(y+1,L)][3]),
                 N.conj_tensor(uq[mod1(x+1,L),y][1]),N.conj_tensor(uq[x,y][2]))
        r=N.owner(g,x,y)
        # References to the small periodic cell: no L^2 copies of Q/R data.
        qb[x,y]=env.Q[mod1(x,g.Lx)][mod1(y,g.Ly)]
        rb[x,y]=env.R[mod1(x,g.Lx)][mod1(y,g.Ly)]
    end
    ops=[N.physical_operators(q) for q in f.Q]
    (;f,L,ketQ=qk,ketR=rk,Uq=uq,Ur=ur,Q=qb,R=rb,Cset=cs,Tset=ts,ops)
end

function central_context(f,env,L,x,y,orientation,edge;shift=1.)
    ctx=base_context(f,env,L);g=f.state.geometry
    changedQ=Tuple{Int,Int}[];changedR=Tuple{Int,Int}[]
    embeddings=Dict{Tuple{Int,Int,Int},Any}()
    if orientation==:up
        r=N.owner(g,x,y);op=ctx.ops[r].pairs[edge]
        P=codomain(op);shifted=op+shift*unitary(P,P)
        @tensor acted[:] := shifted[-4,-5,-6,1,2,3]*f.Q[r][-1,-2,-3,1,2,3,-7]
        ctx.ketQ[x,y]=acted;push!(changedQ,(x,y))
    else
        # Single-layer Q axes are (B1down,B3down,B2down,p1,p2,p3,aux).
        # Double-Q network axes are (B1down,B2down,B3down): N.double_Q
        # swaps the last two virtual axes. Preserve BOTH old conventions.
        specs=((x,y,2,3,3),(x,y+1,3,2,1),(x+1,y,1,1,2)) # x,y,physical,single-Q axis,Rleg
        a,b=PAIRS[edge];xa,ya,ia,qa,ra=specs[a];xb,yb,ib,qb,rb=specs[b]
        A=ctx.ketQ[xa,ya];B=ctx.ketQ[xb,yb]
        ops=operator_factors(A,B,ia,ib;shift)
        ctx.ketQ[xa,ya],Ua=fuse_operator(A,ops.A,qa)
        ctx.ketQ[xb,yb],Ub=fuse_operator(B,ops.B,qb)
        ctx.ketR[x,y]=route_operator(ctx.ketR[x,y],Ua,Ub,ra,rb,ops.W)
        push!(changedQ,(xa,ya),(xb,yb));push!(changedR,(x,y))
        embeddings[(xa,ya,(1,3,2)[qa])]=identity_embedding(Ua,ops.identity_index)
        # Identity lives in the first vacuum degeneracy on either dual space.
        bi=sectortype(ops.W)==U1Irrep ? first(axes(ops.W',U1Irrep(0))) : 1
        embeddings[(xb,yb,(1,3,2)[qb])]=identity_embedding(Ub,bi)
    end
    for (xx,yy) in changedQ
        r=N.owner(g,xx,yy)
        ctx.Uq[xx,yy]=ntuple(k->begin
            if !haskey(embeddings,(xx,yy,k));return ctx.Uq[xx,yy][k];end
            axis=(1,3,2)[k]
            V=space(f.Q[r],axis)'⊗space(ctx.ketQ[xx,yy],axis)
            F=unitary(fuse(V),V)
            E=embeddings[(xx,yy,k)]
            J=F*(unitary(space(f.Q[r],axis)',space(f.Q[r],axis)')⊗E)*ctx.Uq[xx,yy][k]'
            # The internal Q/R edge is not an external PEPS boundary leg.
            if k==1
                xr=mod1(xx-1,L)
                ctx.Tset[xx,yy]=C.set_Tset(ctx.Tset[xx,yy],lift_edge(C.get_Tset(ctx.Tset[xx,yy],2),J),2)
                ctx.Tset[xr,yy]=C.set_Tset(ctx.Tset[xr,yy],lift_edge(C.get_Tset(ctx.Tset[xr,yy],4),J),4)
            elseif k==3
                yr=mod1(yy-1,L)
                ctx.Tset[xx,yy]=C.set_Tset(ctx.Tset[xx,yy],lift_edge(C.get_Tset(ctx.Tset[xx,yy],3),J),3)
                ctx.Tset[xx,yr]=C.set_Tset(ctx.Tset[xx,yr],lift_edge(C.get_Tset(ctx.Tset[xx,yr],1),J),1)
            end
            F
        end,3)
    end
    # Only at most three local mixed factors differ from the periodic network.
    allR=Set(changedR)
    for (xx,yy) in changedQ
        push!(allR,(mod1(xx-1,L),yy),(xx,mod1(yy-1,L)),(xx,yy))
        r=N.owner(g,xx,yy)
        ctx.Q[xx,yy]=mixed_Q(f.Q[r],ctx.ketQ[xx,yy],ctx.Uq[xx,yy])
    end
    for (xx,yy) in allR
        ctx.Ur[xx,yy]=(N.conj_tensor(ctx.Uq[xx,mod1(yy+1,L)][3]),
                       N.conj_tensor(ctx.Uq[mod1(xx+1,L),yy][1]),N.conj_tensor(ctx.Uq[xx,yy][2]))
        r=N.owner(g,xx,yy)
        ctx.R[xx,yy]=mixed_R(f.R[r],ctx.ketR[xx,yy],ctx.Ur[xx,yy])
    end
    ctx,changedQ,changedR
end

function move_origin(j,d,L)
    # convert_cell_posit uses cx-dx (NOT -cx-dx) for direction 3.
    # Both reverse directions therefore use origin=j+1, traversed downward.
    d in (1,2) ? j-1 : j+1
end
# Retained as the small-cell numerical oracle for the measurement-only array path.
function large_cell_sweeps_reference(ctx,positions;chi,sweeps=2,callback=nothing)
    L=ctx.L;set_dimensions(L,L)
    cfg=C.LS_CTMRG_settings(CTM_trun_tol=1e-12,grad_checkpoint=false,use_sub_checkpoint=false)
    Q=C.LayerGrid(tuple_grid(ctx.Q));R=C.LayerGrid(tuple_grid(ctx.R))
    cs,ts=tuple_grid(ctx.Cset),tuple_grid(ctx.Tset)
    xmin,xmax=extrema(first.(positions));ymin,ymax=extrema(last.(positions))
    function absorb(j,d)
        cs,ts=C.ctm_update_single_cx(chi,move_origin(j,d,L),L,cs,ts,Q,R,d,cfg,L,L)
        isnothing(callback) || callback((direction=d,absorbed_coordinate=j))
    end
    # Warm up immediately outside the enlarged central bond(s).
    absorb(xmax,3);absorb(ymax,4)
    for sweep in 1:sweeps
        for j in xmin-1:L-1;absorb(j,1);end
        for j in xmax+1:-1:2;absorb(j,3);end
        for j in ymin-1:L-1;absorb(j,2);end
        for j in ymax+1:-1:2;absorb(j,4);end
    end
    merge(ctx,(Cset=cs,Tset=ts,Qgrid=Q,Rgrid=R))
end

function bond_expectation(ctx,x,y,orientation,edge)
    g=(Lx=ctx.L,Ly=ctx.L);small=ctx.f.state.geometry
    positions=((x,y),(x+1,y),(x,y+1),(x+1,y+1));modes=(:LU,:RU,:LD,:RD)
    env=(Cset=ctx.Cset isa AbstractMatrix ? tuple_grid(ctx.Cset) : ctx.Cset,
         Tset=ctx.Tset isa AbstractMatrix ? tuple_grid(ctx.Tset) : ctx.Tset)
    corners=Any[N.enlarged_corner(env,ctx.Q[p...],ctx.R[p...],modes[i],x-1,y-1,g)
             for (i,p) in enumerate(positions)]
    normal=N.close_corners(corners)
    abs(normal)>1e-200 || error("zero mixed overlap in local measurement")
    function inserted(i,op;charged=false)
        xx,yy=positions[i];r=N.owner(small,xx,yy)
        q=mixed_Q(ctx.f.Q[r],ctx.ketQ[xx,yy],ctx.Uq[xx,yy];operator=op,charged)
        N.enlarged_corner(env,q,ctx.R[xx,yy],modes[i],x-1,y-1,g;charged)
    end
    if orientation==:up
        r=N.owner(small,x,y);op=ctx.ops[r].pairs[edge]
        cs=copy(corners);cs[1]=inserted(1,op)
        return N.close_corners(cs)/normal
    end
    specs=((1,2),(3,3),(2,1));a,b=PAIRS[edge];ca,ia=specs[a];cb,ib=specs[b]
    ra=N.owner(small,positions[ca]...);rb=N.owner(small,positions[cb]...)
    oa=ctx.ops[ra];ob=ctx.ops[rb]
    total=0.0+0.0im
    for (op1,op2,charged,weight) in ((oa.z[ia],ob.z[ib],false,1.),
            (oa.plus[ia],ob.minus[ib],true,.5),(oa.minus[ia],ob.plus[ib],true,.5))
        cs=copy(corners);cs[ca]=inserted(ca,op1;charged);cs[cb]=inserted(cb,op2;charged)
        total+=weight*N.close_corners(cs,charged ? Tuple(sort([ca,cb])) : ())/normal
    end
    total
end
