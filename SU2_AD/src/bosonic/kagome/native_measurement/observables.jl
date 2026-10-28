# Physical up/down order is inherited from the existing SU2 -> U1 restriction.
const SZ=ComplexF64[.5 0;0 -.5]
const SP=ComplexF64[0 1;0 0]
function embedded_operator(site,op)
    values=[Matrix{ComplexF64}(I,2,2) for _ in 1:3];values[site]=op
    kron(values[3],values[2],values[1]) # first physical index is fastest in Julia
end
function physical_operators(Q)
    P=space(Q,4)⊗space(Q,5)⊗space(Q,6)
    neutral(A)=to_device(TensorMap(reshape(Array(A),ntuple(_->2,6)),P,P;tol=1e-11))
    z=[embedded_operator(i,SZ) for i in 1:3]
    plus=[embedded_operator(i,SP) for i in 1:3];minus=adjoint.(plus)
    pair=[neutral(z[a]*z[b]+.5*(plus[a]*minus[b]+minus[a]*plus[b])) for (a,b) in ((1,2),(2,3),(3,1))]
    if sectortype(space(Q,1))==U1Irrep
        W=nothing
        for q in (2,-2)
            candidate=U1Space(q=>1)
            try
                t=TensorMap(reshape(plus[1],(2,2,2,2,2,2,1)),P,P⊗candidate';tol=1e-11)
                norm(convert(Array,t)-reshape(plus[1],size(convert(Array,t))))<1e-10 || continue
                W=candidate;break
            catch err
                err isa InterruptException && rethrow()
            end
        end
        isnothing(W) && error("charged operator is incompatible with physical arrows")
    else
        W=ComplexSpace(1)
    end
    charged(A,V)=to_device(TensorMap(reshape(Array(A),(2,2,2,2,2,2,1)),P,P⊗V';tol=1e-11))
    (;z=neutral.(z),plus=[charged(a,W) for a in plus],minus=[charged(a,W') for a in minus],
       pairs=pair,triple=neutral(z[1]*z[2]*z[3]))
end
function enlarged_corner(env,q,r,mode,x,y,g;charged=false)
    c(dx,dy,k)=CTM.get_Cset(env.Cset[mod1(x+dx,g.Lx)][mod1(y+dy,g.Ly)],k)
    t(dx,dy,k)=CTM.get_Tset(env.Tset[mod1(x+dx,g.Lx)][mod1(y+dy,g.Ly)],k)
    e=mode==:LU ? (c(0,0,1),t(1,0,1),t(0,1,4)) :
      mode==:RU ? (t(2,0,1),c(3,0,2),t(3,1,2)) :
      mode==:LD ? (t(0,2,4),c(0,3,4),t(1,3,3)) :
                  (t(3,2,2),t(2,3,3),c(3,3,3))
    CTM.factor_corner(q,r,e...,mode,1;charged)
end
function close_corners(corners,charged_positions=())
    a,b,c,d=corners
    if isempty(charged_positions)
        @tensor value[] := a[5,6,1,2]*b[1,2,7,8]*c[5,6,3,4]*d[3,4,7,8]
    elseif charged_positions==(1,2)
        @tensor value[] := a[5,6,1,2,9]*b[1,2,7,8,9]*c[5,6,3,4]*d[3,4,7,8]
    elseif charged_positions==(1,3)
        @tensor value[] := a[5,6,1,2,9]*b[1,2,7,8]*c[5,6,3,4,9]*d[3,4,7,8]
    elseif charged_positions==(1,4)
        @tensor value[] := a[5,6,1,2,9]*b[1,2,7,8]*c[5,6,3,4]*d[3,4,7,8,9]
    elseif charged_positions==(2,3)
        @tensor value[] := a[5,6,1,2]*b[1,2,7,8,9]*c[5,6,3,4,9]*d[3,4,7,8]
    elseif charged_positions==(2,4)
        @tensor value[] := a[5,6,1,2]*b[1,2,7,8,9]*c[5,6,3,4]*d[3,4,7,8,9]
    elseif charged_positions==(3,4)
        @tensor value[] := a[5,6,1,2]*b[1,2,7,8]*c[5,6,3,4,9]*d[3,4,7,8,9]
    else
        error("charged insertions must be a conjugate pair")
    end
    # Older TensorKit cannot convert a symmetry-carrying rank-0 map to Array.
    # Its unique vacuum block is a 1x1 matrix, also safe to copy from CUDA.
    sum(sum(Array(b)) for (_,b) in blocks(value);init=0.0+0.0im)
end
function native_observables(f::NativeFactors,env;h=.38)
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
        z0=[inserted(1,a) for a in op.z]
        mz=[expectation(Dict(1=>v)) for v in z0]
        up=[expectation(Dict(1=>inserted(1,a))) for a in op.pairs]
        cup=expectation(Dict(1=>inserted(1,op.triple)))
        # Td(R): B2(R), B3(R+ey), B1(R+ex), exactly the old Julia convention.
        specs=((1,2),(3,3),(2,1));localz=Any[];localp=Any[];localm=Any[]
        for (i,b) in specs
            op=ops[owners[i]]
            push!(localz,i==1 ? z0[b] : inserted(i,op.z[b]))
            push!(localp,inserted(i,op.plus[b];charged=true))
            push!(localm,inserted(i,op.minus[b];charged=true))
        end
        down=Float64[]
        for (a,b) in ((1,2),(2,3),(3,1))
            ca,cb=specs[a][1],specs[b][1];pair=Tuple(sort([ca,cb]))
            e=expectation(Dict(ca=>localz[a],cb=>localz[b]))
            e+=.5expectation(Dict(ca=>localp[a],cb=>localm[b]);charged=pair)
            e+=.5expectation(Dict(ca=>localm[a],cb=>localp[b]);charged=pair)
            push!(down,e)
        end
        cdown=expectation(Dict(specs[i][1]=>localz[i] for i in 1:3))
        push!(cells,Dict("x"=>x,"y"=>y,"representative"=>"$(owners[1]),1","Sz"=>mz,
            "E_up"=>sum(up),"E_down"=>sum(down),"bonds_up"=>up,"bonds_down"=>down,
            "C_up"=>cup,"C_down"=>cdown))
    end
    mz=sum(sum(c["Sz"]) for c in cells)/(3length(cells))
    exchange=sum(c["E_up"]+c["E_down"] for c in cells)/(3length(cells))
    Dict("energy_per_spin"=>exchange+h*mz,"exchange_energy_per_spin"=>exchange,
        "field_energy_per_spin"=>h*mz,"mean_Sz"=>mz,"m_normalized"=>-2mz,
        "distance_to_one_ninth"=>-2mz-1/9,"cells"=>cells,"max_observable_imaginary_part"=>max_imag)
end
