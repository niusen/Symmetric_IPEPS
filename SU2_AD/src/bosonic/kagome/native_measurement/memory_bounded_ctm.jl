# Exact memory-bounded contractions. Original generic methods remain the oracle.
# Ordinary double layers are stored as rank-3 Q/R. Like the triangle code,
# get_AA_direction below assembles a SINGLE site's AA on demand, never a cell.
# Production corner/edge absorption avoids even that AA when possible: it joins
# Q/R to separate pieces of the boundary before closing both connecting legs.
# Do NOT restore order=(1,2,3,4,5): that makes a chi^2*(D^2)^3 intermediate.
# Orders below were enumerated over all 5!/6! index sequences, including the
# initial boundary chi0=D^2. Shared indices between two tensors close together.
# No truncation, conjugation, physical axes or CTMRG projector changes are made.
@eval CTM begin
function local_double_layer(q,r;charged=false)
    if charged
        @tensor aa[:] := q[-1,1,-4,-5]*r[-2,-3,1]
    else
        @tensor aa[:] := q[-1,1,-4]*r[-2,-3,1]
    end
    aa
end
function get_AA_direction(Q::LayerGrid,R::LayerGrid,d,pos)
    rotate_AA_direction(local_double_layer(Q[pos[1]][pos[2]],R[pos[1]][pos[2]]),d)
end
function factor_corner(q::AbstractTensorMap,r::AbstractTensorMap,
    e1::AbstractTensorMap,e2::AbstractTensorMap,e3::AbstractTensorMap,mode::Symbol,d::Integer;charged=false)
    if mode==:LU && d==1 && charged==false
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[4,5,3]*r[-2,-4,5]
        return value
    end
    if mode==:LU && d==1 && charged==true
        @tensor order=(1,2,3,4,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[4,5,3,-5]*r[-2,-4,5]
        return value
    end
    if mode==:LU && d==2 && charged==false
        @tensor order=(1,3,4,2,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[-2,5,4]*r[-4,3,5]
        return value
    end
    if mode==:LU && d==2 && charged==true
        @tensor order=(1,3,4,2,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[-2,5,4,-5]*r[-4,3,5]
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
        @tensor order=(1,3,4,2,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[3,5,-4]*r[4,-2,5]
        return value
    end
    if mode==:LU && d==4 && charged==true
        @tensor order=(1,3,4,2,5) value[:] := e1[1,2]*e2[2,3,-3]*e3[-1,4,1]*q[3,5,-4,-5]*r[4,-2,5]
        return value
    end
    if mode==:RU && d==1 && charged==false
        @tensor order=(1,3,4,2,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[-2,5,3]*r[-4,4,5]
        return value
    end
    if mode==:RU && d==1 && charged==true
        @tensor order=(1,3,4,2,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[-2,5,3,-5]*r[-4,4,5]
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
        @tensor order=(1,3,4,2,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[4,5,-4]*r[3,-2,5]
        return value
    end
    if mode==:RU && d==3 && charged==true
        @tensor order=(1,3,4,2,5) value[:] := e1[-1,3,1]*e2[1,2]*e3[2,4,-3]*q[4,5,-4,-5]*r[3,-2,5]
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
        @tensor order=(1,3,4,2,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[3,5,-2]*r[4,-4,5]
        return value
    end
    if mode==:LD && d==1 && charged==true
        @tensor order=(1,3,4,2,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[3,5,-2,-5]*r[4,-4,5]
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
        @tensor order=(1,3,4,2,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[-4,5,4]*r[-2,3,5]
        return value
    end
    if mode==:LD && d==3 && charged==true
        @tensor order=(1,3,4,2,5) value[:] := e1[1,3,-1]*e2[2,1]*e3[-3,4,2]*q[-4,5,4,-5]*r[-2,3,5]
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
        @tensor order=(1,3,4,2,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[3,5,-2]*r[4,-4,5]
        return value
    end
    if mode==:RD && d==2 && charged==true
        @tensor order=(1,3,4,2,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[3,5,-2,-5]*r[4,-4,5]
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
        @tensor order=(1,3,4,2,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[-4,5,4]*r[-2,3,5]
        return value
    end
    if mode==:RD && d==4 && charged==true
        @tensor order=(1,3,4,2,5) value[:] := e1[-3,4,2]*e2[1,3,-1]*e3[2,1]*q[-4,5,4,-5]*r[-2,3,5]
        return value
    end
    error("unknown corner/rotation")
end
function factor_edge(t::AbstractTensorMap,pi::AbstractTensorMap,p::AbstractTensorMap,
    q::AbstractTensorMap,r::AbstractTensorMap,d::Integer)
    if d==1
        @tensor order=(1,2,3,4,5,6) value[:] := t[4,3,1]*pi[4,5,-1]*p[1,2,-3]*q[3,6,2]*r[5,-2,6]
        return value
    end
    if d==2
        @tensor order=(2,3,4,1,5,6) value[:] := t[4,3,1]*pi[4,5,-1]*p[1,2,-3]*q[5,6,3]*r[-2,2,6]
        return value
    end
    if d==3
        @tensor order=(1,2,3,4,5,6) value[:] := t[4,3,1]*pi[4,5,-1]*p[1,2,-3]*q[-2,6,5]*r[2,3,6]
        return value
    end
    if d==4
        @tensor order=(2,3,4,1,5,6) value[:] := t[4,3,1]*pi[4,5,-1]*p[1,2,-3]*q[2,6,-2]*r[3,5,6]
        return value
    end
    error("invalid direction")
end

# Hold two enlarged corners at a time rather than all four. This is the
# repository's original get_M algebra, with shorter temporary lifetimes.
function paired_half(coord,d,Q,R,C,T,Lx,Ly,upper)
    if upper
        a=build_corner_MMup(coord,d,Q,R,C,T,Lx,Ly)
        b=build_corner_MMup_reflect(coord,d,Q,R,C,T,Lx,Ly)
    else
        a=build_corner_MMlow(coord,d,Q,R,C,T,Lx,Ly)
        b=build_corner_MMlow_reflect(coord,d,Q,R,C,T,Lx,Ly)
    end
    @tensor half[:] := a[-1,-2,1,2]*b[1,2,-3,-4]
    permute(half,upper ? ((3,4),(1,2)) : ((1,2),(3,4)))
end
function get_M(coord,d,Q::LayerGrid,R::LayerGrid,C,T,Lx,Ly)
    up=paired_half(coord,d,Q,R,C,T,Lx,Ly,true)
    low=paired_half(coord,d,Q,R,C,T,Lx,Ly,false)
    up=up/norm(up);low=low/norm(low)
    M=up*low
    M/norm(M),up,low
end
end # CTM

function native_observables(f::NativeFactors,env::NamedTuple;h=.38)
    g=f.state.geometry;cells=Dict[];max_imag=0.
    ops=[physical_operators(Q) for Q in f.Q]
    for x in 1:g.Lx,y in 1:g.Ly
        positions=((x,y),(x+1,y),(x,y+1),(x+1,y+1));names=(:LU,:RU,:LD,:RD)
        owners=[owner(g,p...) for p in positions]
        corners=[enlarged_corner(env,env.Q[mod1(p[1],g.Lx)][mod1(p[2],g.Ly)],
            env.R[mod1(p[1],g.Lx)][mod1(p[2],g.Ly)],names[i],x-1,y-1,g)
            for (i,p) in enumerate(positions)]
        normal=close_corners(corners)
        function inserted(i,op;charged=false)
            r=owners[i];uq,ur=factor_fusers(f,r)
            q=double_Q(f.Q[r],uq;operator=op,charged)
            p=positions[i]
            enlarged_corner(env,q,env.R[mod1(p[1],g.Lx)][mod1(p[2],g.Ly)],names[i],x-1,y-1,g;charged)
        end
        function expectation(replacements;charged=())
            cs=Any[corners...] # charged insertions carry one extra 1D charge leg
            for (i,v) in replacements;cs[i]=v;end
            v=close_corners(cs,charged)/normal
            max_imag=max(max_imag,abs(imag(v)))
            real(v)
        end
        op=ops[owners[1]]
        # Stream operator insertions. Do not keep Sz/S+/S- corners for all
        # sites simultaneously: each such corner is chi^2*D^4 before blocks.
        mz=[expectation(Dict(1=>inserted(1,a))) for a in op.z]
        up=[expectation(Dict(1=>inserted(1,a))) for a in op.pairs]
        cup=expectation(Dict(1=>inserted(1,op.triple)))
        specs=((1,2),(3,3),(2,1)) # same Td / B2(R), B3(R+ey), B1(R+ex)
        function pair_term(ca,ba,cb,bb,kind)
            oa,ob=ops[owners[ca]],ops[owners[cb]]
            if kind==:zz
                aa=inserted(ca,oa.z[ba]);bbv=inserted(cb,ob.z[bb])
                return expectation(Dict(ca=>aa,cb=>bbv))
            end
            aop=kind==:pm ? oa.plus[ba] : oa.minus[ba]
            bop=kind==:pm ? ob.minus[bb] : ob.plus[bb]
            aa=inserted(ca,aop;charged=true);bbv=inserted(cb,bop;charged=true)
            expectation(Dict(ca=>aa,cb=>bbv);charged=Tuple(sort([ca,cb])))
        end
        down=Float64[]
        for (a,b) in ((1,2),(2,3),(3,1))
            ca,ba=specs[a];cb,bb=specs[b]
            push!(down,pair_term(ca,ba,cb,bb,:zz)+
                .5pair_term(ca,ba,cb,bb,:pm)+.5pair_term(ca,ba,cb,bb,:mp))
        end
        cdown=expectation(Dict(i=>inserted(i,ops[owners[i]].z[b]) for (i,b) in specs))
        push!(cells,Dict("x"=>x,"y"=>y,"representative"=>"$(owners[1]),1","Sz"=>mz,
            "E_up"=>sum(up),"E_down"=>sum(down),"bonds_up"=>up,"bonds_down"=>down,
            "C_up"=>cup,"C_down"=>cdown))
    end
    mz=sum(sum(c["Sz"]) for c in cells)/(3length(cells))
    exchange=sum(c["E_up"]+c["E_down"] for c in cells)/(3length(cells))
    Dict("energy_per_spin"=>exchange+h*mz,"exchange_energy_per_spin"=>exchange,
        "field_energy_per_spin"=>h*mz,"mean_Sz"=>mz,"m_normalized"=>-2mz,
        "contraction_backend"=>"memory_bounded_QR_v1",
        "distance_to_one_ninth"=>-2mz-1/9,"cells"=>cells,"max_observable_imaginary_part"=>max_imag)
end
