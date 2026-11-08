# Measurement does not differentiate through the CTM environment. Do not use
# initial_tuple_cell/fill_tuple for an L-by-L measurement window: replacing Int
# placeholders by TensorMaps changes a large nested tuple's TYPE on every fill,
# producing many expensive Julia specializations. Arrays keep one storage type.
# Tensor contractions, all projectors on the same cut, normalizations, and the
# final-update order are exactly those of C.ctm_update_single_cx.
grid_value(a,x,y)=a isa AbstractMatrix ? a[x,y] : a[x][y]
measurement_grid(a,L)=[Any[grid_value(a,x,y) for y in 1:L] for x in 1:L]

function variance_column_update!(cs,ts,Q,R,chi,cx,d,cfg,L)
    P=Matrix{Any}(undef,L,L);Pi=similar(P)
    edges=Vector{Any}(undef,L);uppers=similar(edges);lowers=similar(edges)
    for cy in 1:L
        M,rmup,rmlow=C.get_M([cx,cy],d,Q,R,cs,ts,L,L)
        u,s,v=C.native_svd(M,chi,cfg.CTM_trun_tol);s=s/norm(s)
        si=complex(C.safe_inv_sqrt(s,cfg.CTM_trun_tol))
        pi=rmlow*v'*si
        p=permute(si*u'*rmup,((2,3),(1,)))
        a=C.convert_cell_posit(cx,cy,0,2,d,L,L);P[a[1],a[2]]=p
        a=C.convert_cell_posit(cx,cy,0,1,d,L,L);Pi[a[1],a[2]]=pi
    end
    # Finish every absorption before changing cs/ts, as in the tuple oracle.
    for cy in 1:L
        pos(a,b)=C.convert_cell_posit(cx,cy,a,b,d,L,L)
        c(a,b,k)=C.get_Cset(cs[pos(a,b)[1]][pos(a,b)[2]],mod1(k,4))
        t(a,b,k)=C.get_Tset(ts[pos(a,b)[1]][pos(a,b)[2]],mod1(k,4))
        a=pos(1,2);pa=pos(0,2)
        e=C.factor_edge(t(0,2,d-1),Pi[pa[1],pa[2]],P[pa[1],pa[2]],Q[a[1]][a[2]],R[a[1]][a[2]],d)
        cc=c(0,0,d);tt=t(1,0,d);a=pos(0,0);pi=Pi[a[1],a[2]]
        @tensor upper[:] := cc[1,2]*tt[2,3,-2]*pi[1,3,-1]
        cc=c(0,3,d-1);tt=t(1,3,d-2);a=pos(0,3);p=P[a[1],a[2]]
        @tensor lower[:] := cc[1,2]*tt[-1,3,1]*p[2,3,-2]
        edges[cy]=e/norm(e);uppers[cy]=upper/norm(upper);lowers[cy]=lower/norm(lower)
    end
    for cy in 1:L
        # Cset_struc/Tset_struc are immutable. Replace array entries, never
        # mutate the shared periodic environment or its TensorMap contents.
        a=C.convert_cell_posit(cx,cy,1,0,d,L,L)
        cs[a[1]][a[2]]=C.set_Cset(cs[a[1]][a[2]],uppers[cy],mod1(d,4))
        a=C.convert_cell_posit(cx,cy,1,2,d,L,L)
        ts[a[1]][a[2]]=C.set_Tset(ts[a[1]][a[2]],edges[cy],mod1(d-1,4))
        a=C.convert_cell_posit(cx,cy,1,3,d,L,L)
        cs[a[1]][a[2]]=C.set_Cset(cs[a[1]][a[2]],lowers[cy],mod1(d-1,4))
    end
    nothing
end

function large_cell_sweeps(ctx,positions;chi,sweeps=2,callback=nothing,storage=:array)
    storage==:tuple && return large_cell_sweeps_reference(ctx,positions;chi,sweeps,callback)
    storage==:array || error("variance sweep storage must be :array or :tuple")
    L=ctx.L;set_dimensions(L,L)
    cfg=C.LS_CTMRG_settings(CTM_trun_tol=1e-12,grad_checkpoint=false,use_sub_checkpoint=false)
    Q=C.LayerGrid(tuple_grid(ctx.Q));R=C.LayerGrid(tuple_grid(ctx.R))
    cs=measurement_grid(ctx.Cset,L);ts=measurement_grid(ctx.Tset,L)
    xmin,xmax=extrema(first.(positions));ymin,ymax=extrema(last.(positions))
    function absorb(j,d)
        variance_column_update!(cs,ts,Q,R,chi,move_origin(j,d,L),d,cfg,L)
        isnothing(callback) || callback((direction=d,absorbed_coordinate=j))
    end
    absorb(xmax,3);absorb(ymax,4)
    for sweep in 1:sweeps
        for j in xmin-1:L-1;absorb(j,1);end
        for j in xmax+1:-1:2;absorb(j,3);end
        for j in ymin-1:L-1;absorb(j,2);end
        for j in ymax+1:-1:2;absorb(j,4);end
    end
    merge(ctx,(Cset=cs,Tset=ts,Qgrid=Q,Rgrid=R))
end
