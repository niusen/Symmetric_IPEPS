# Specialized factor contractions around the existing CTMRG projector code.
@eval CTM begin
struct LayerGrid
    cells::Tuple
end
Base.getindex(a::LayerGrid,i)=a.cells[i]
function factor_corner(q,r,e1,e2,e3,mode,d;charged=false)
    if mode==:LU && d==1 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[4,5,3]*r[-2,-4,5]
        return value
    end
    if mode==:LU && d==1 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[4,5,3,-5]*r[-2,-4,5]
        return value
    end
    if mode==:LU && d==2 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[-2,5,4]*r[-4,3,5]
        return value
    end
    if mode==:LU && d==2 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[-2,5,4,-5]*r[-4,3,5]
        return value
    end
    if mode==:LU && d==3 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[-4,5,-2]*r[3,4,5]
        return value
    end
    if mode==:LU && d==3 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[-4,5,-2,-5]*r[3,4,5]
        return value
    end
    if mode==:LU && d==4 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[3,5,-4]*r[4,-2,5]
        return value
    end
    if mode==:LU && d==4 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[3,5,-4,-5]*r[4,-2,5]
        return value
    end
    if mode==:RU && d==1 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[-2,5,3]*r[-4,4,5]
        return value
    end
    if mode==:RU && d==1 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[-2,5,3,-5]*r[-4,4,5]
        return value
    end
    if mode==:RU && d==2 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[-4,5,-2]*r[4,3,5]
        return value
    end
    if mode==:RU && d==2 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[-4,5,-2,-5]*r[4,3,5]
        return value
    end
    if mode==:RU && d==3 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[4,5,-4]*r[3,-2,5]
        return value
    end
    if mode==:RU && d==3 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[4,5,-4,-5]*r[3,-2,5]
        return value
    end
    if mode==:RU && d==4 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[3,5,4]*r[-2,-4,5]
        return value
    end
    if mode==:RU && d==4 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[3,5,4,-5]*r[-2,-4,5]
        return value
    end
    if mode==:LD && d==1 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[3,5,-2]*r[4,-4,5]
        return value
    end
    if mode==:LD && d==1 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[3,5,-2,-5]*r[4,-4,5]
        return value
    end
    if mode==:LD && d==2 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[4,5,3]*r[-4,-2,5]
        return value
    end
    if mode==:LD && d==2 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[4,5,3,-5]*r[-4,-2,5]
        return value
    end
    if mode==:LD && d==3 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[-4,5,4]*r[-2,3,5]
        return value
    end
    if mode==:LD && d==3 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[-4,5,4,-5]*r[-2,3,5]
        return value
    end
    if mode==:LD && d==4 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[-2,5,-4]*r[3,4,5]
        return value
    end
    if mode==:LD && d==4 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[-2,5,-4,-5]*r[3,4,5]
        return value
    end
    if mode==:RD && d==1 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[-2,5,-4]*r[3,4,5]
        return value
    end
    if mode==:RD && d==1 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[-2,5,-4,-5]*r[3,4,5]
        return value
    end
    if mode==:RD && d==2 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[3,5,-2]*r[4,-4,5]
        return value
    end
    if mode==:RD && d==2 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[3,5,-2,-5]*r[4,-4,5]
        return value
    end
    if mode==:RD && d==3 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[4,5,3]*r[-4,-2,5]
        return value
    end
    if mode==:RD && d==3 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[4,5,3,-5]*r[-4,-2,5]
        return value
    end
    if mode==:RD && d==4 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[-4,5,4]*r[-2,3,5]
        return value
    end
    if mode==:RD && d==4 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[-4,5,4,-5]*r[-2,3,5]
        return value
    end
    error("unknown corner/rotation")
end
function factor_edge(t,pi,p,q,r,d)
    if d==1
        @tensor order=(4,1,3,5,2,6) value[:] := t[4,3,1]*pi[4,5,-1]*p[1,2,-3]*q[3,6,2]*r[5,-2,6]
        return value
    end
    if d==2
        @tensor order=(4,1,3,5,2,6) value[:] := t[4,3,1]*pi[4,5,-1]*p[1,2,-3]*q[5,6,3]*r[-2,2,6]
        return value
    end
    if d==3
        @tensor order=(4,1,3,5,2,6) value[:] := t[4,3,1]*pi[4,5,-1]*p[1,2,-3]*q[-2,6,5]*r[2,3,6]
        return value
    end
    if d==4
        @tensor order=(4,1,3,5,2,6) value[:] := t[4,3,1]*pi[4,5,-1]*p[1,2,-3]*q[2,6,-2]*r[3,5,6]
        return value
    end
    error("invalid direction")
end

function split_corner(coord,d,Q::LayerGrid,R::LayerGrid,C,T,Lx,Ly,mode)
    pos(a,b)=convert_cell_posit(coord[1],coord[2],a,b,d,Lx,Ly)
    c(a,b,k)=get_Cset(C[pos(a,b)[1]][pos(a,b)[2]],mod1(k,4))
    t(a,b,k)=get_Tset(T[pos(a,b)[1]][pos(a,b)[2]],mod1(k,4))
    if mode==:LU
        a=pos(1,1);e=(c(0,0,d),t(1,0,d),t(0,1,d-1))
    elseif mode==:LD
        a=pos(1,2);e=(t(0,2,d-1),c(0,3,d-1),t(1,3,d-2))
    elseif mode==:RU
        a=pos(2,1);e=(t(2,0,d),c(3,0,d+1),t(3,1,d+1))
    else
        a=pos(2,2);e=(t(3,2,d+1),t(2,3,d-2),c(3,3,d-2))
    end
    factor_corner(Q[a[1]][a[2]],R[a[1]][a[2]],e...,mode,d)
end
build_corner_MMup(coord,d,Q::LayerGrid,R::LayerGrid,C,T,Lx,Ly)=split_corner(coord,d,Q,R,C,T,Lx,Ly,:LU)
build_corner_MMlow(coord,d,Q::LayerGrid,R::LayerGrid,C,T,Lx,Ly)=split_corner(coord,d,Q,R,C,T,Lx,Ly,:LD)
build_corner_MMup_reflect(coord,d,Q::LayerGrid,R::LayerGrid,C,T,Lx,Ly)=split_corner(coord,d,Q,R,C,T,Lx,Ly,:RU)
build_corner_MMlow_reflect(coord,d,Q::LayerGrid,R::LayerGrid,C,T,Lx,Ly)=split_corner(coord,d,Q,R,C,T,Lx,Ly,:RD)

function native_svd(M,chi,tol)
    if isdefined(TensorKit,:tsvd)
        return TensorKit.tsvd(M;trunc=TensorKit.truncdim(chi)&TensorKit.truncbelow(tol*norm(M)))
    end
    u,s,v,_=TensorKit.svd_trunc(M;trunc=TensorKit.truncrank(chi)&TensorKit.trunctol(;atol=tol*norm(M)))
    return u,s,v
end
function safe_inv_sqrt(S,tol)
    result=copy(S)
    largest=maximum(maximum(abs,b isa Diagonal ? b.diag : b) for (_,b) in blocks(S))
    for (_,b) in blocks(result)
        values=b isa Diagonal ? b.diag : b
        values .= ifelse.(real.(values).>tol*largest,
            inv.(sqrt.(max.(real.(values),eps(Float64)))),zero(eltype(values)))
    end
    result
end
function ctm_update_single_cx(chi,cx,cy_max,C,T,Q::LayerGrid,R::LayerGrid,d,cfg,Lx,Ly)
    P=initial_tuple_cell(Lx,Ly);Pi=initial_tuple_cell(Lx,Ly)
    m1=initial_tuple_cell(Lx,Ly);m5=initial_tuple_cell(Lx,Ly);m7=initial_tuple_cell(Lx,Ly)
    for cy in 1:cy_max
        coord=[cx,cy]
        M,rmup,rmlow=get_M(coord,d,Q,R,C,T,Lx,Ly)
        u,s,v=native_svd(M,chi,cfg.CTM_trun_tol);s=s/norm(s)
        si=complex(safe_inv_sqrt(s,cfg.CTM_trun_tol))
        pi=rmlow*v'*si
        p=permute(si*u'*rmup,((2,3),(1,)))
        a=convert_cell_posit(cx,cy,0,2,d,Lx,Ly);P=fill_tuple(P,p,a[1],a[2])
        a=convert_cell_posit(cx,cy,0,1,d,Lx,Ly);Pi=fill_tuple(Pi,pi,a[1],a[2])
    end
    for cy in 1:cy_max
        pos(a,b)=convert_cell_posit(cx,cy,a,b,d,Lx,Ly)
        c(a,b,k)=get_Cset(C[pos(a,b)[1]][pos(a,b)[2]],mod1(k,4))
        t(a,b,k)=get_Tset(T[pos(a,b)[1]][pos(a,b)[2]],mod1(k,4))
        a=pos(1,2);pa=pos(0,2)
        e=factor_edge(t(0,2,d-1),Pi[pa[1]][pa[2]],P[pa[1]][pa[2]],Q[a[1]][a[2]],R[a[1]][a[2]],d)
        cc=c(0,0,d);tt=t(1,0,d);a=pos(0,0);pi=Pi[a[1]][a[2]]
        @tensor upper[:] := cc[1,2]*tt[2,3,-2]*pi[1,3,-1]
        cc=c(0,3,d-1);tt=t(1,3,d-2);a=pos(0,3);p=P[a[1]][a[2]]
        @tensor lower[:] := cc[1,2]*tt[-1,3,1]*p[2,3,-2]
        a=pos(1,2);m5=fill_tuple(m5,e/norm(e),a[1],a[2])
        a=pos(1,0);m1=fill_tuple(m1,upper/norm(upper),a[1],a[2])
        a=pos(1,3);m7=fill_tuple(m7,lower/norm(lower),a[1],a[2])
    end
    for cy in 1:cy_max
        C,T=final_CTM_update(C,T,m1,m5,m7,[cx,cy],d,Lx,Ly)
    end
    C,T
end
end

function native_ctmrg(f::NativeFactors;chi=40,iterations=50,tolerance=1e-6,verbose=true)
    chi>0 && 2<=iterations<=50 && tolerance>0 || error("invalid CTMRG settings")
    g=f.state.geometry
    Core.eval(CTM,:(Lx=$(g.Lx)));Core.eval(CTM,:(Ly=$(g.Ly)))
    qs=Any[];rs=Any[];cs=Any[];ts=Any[]
    for r in eachindex(g.representatives)
        uq,ur=factor_fusers(f,r)
        q=double_Q(f.Q[r],uq);rr=double_R(f.R[r],ur)
        c,t=initial_environment(q,rr,uq,ur)
        push!(qs,q);push!(rs,rr);push!(cs,c);push!(ts,t)
    end
    grid(a)=ntuple(x->ntuple(y->a[owner(g,x,y)],g.Ly),g.Lx)
    Q=CTM.LayerGrid(grid(qs));R=CTM.LayerGrid(grid(rs));C=grid(cs);T=grid(ts)
    cfg=CTM.LS_CTMRG_settings(CTM_trun_tol=1e-12,grad_checkpoint=false,use_sub_checkpoint=false)
    previous=Dict();err=Inf;iter=0
    for iteration in 1:iterations
        for d in (3,4,1,2)
            C,T=CTM.CTM_ite_cell_continuous_update(C,T,Q,R,chi,d,cfg,g.Lx,g.Ly)
        end
        current=Dict();err=0.
        for x in 1:g.Lx,y in 1:g.Ly,k in 1:4
            values=svdvals(convert(Array,to_cpu(CTM.get_Cset(C[x][y],k))))
            values ./= maximum(values);key=(x,y,k);current[key]=values
            if haskey(previous,key)
                old=previous[key];n=max(length(old),length(values));dv=zeros(n)
                dv[1:length(old)].-=old;dv[1:length(values)].+=values
                err=max(err,norm(dv))
            else
                err=Inf
            end
        end
        previous=current;iter=iteration
        verbose && (println("Julia CTMRG ",iteration,": corner-spectrum error=",err);flush(stdout))
        err<tolerance && break
    end
    return (;Cset=C,Tset=T,Q,R),Dict("iterations"=>iter,"error"=>err,"converged"=>err<tolerance,
        "chi"=>chi,"actual_corner_dimensions"=>sort(unique(dim(space(CTM.get_Cset(C[x][y],k),1))
        for x in 1:g.Lx,y in 1:g.Ly,k in 1:4)))
end
