# Preserve complex <Sp Sm> until Hermitian exchange/vector-chirality assembly.
# Taking real(pm) and real(mp) separately would erase the entire DMI energy.
function dmi_observables(f,env;J=1.,Dz=0.,h=.38)
    g=f.state.geometry;cells=Dict[];maximag=0.
    ops=[N.physical_operators(Q) for Q in f.Q]
    dmops=Any[];xyops=Any[]
    for Q in f.Q
        P=space(Q,4)⊗space(Q,5)⊗space(Q,6)
        neutral(a)=N.to_device(TensorMap(reshape(a,ntuple(_->2,6)),P,P;tol=1e-11))
        p=[N.embedded_operator(i,N.SP) for i in 1:3];m=adjoint.(p)
        push!(dmops,[neutral(-.5im*(p[a]*m[b]-m[a]*p[b])) for (a,b) in CYCLE])
        if sectortype(space(Q,1))==U1Irrep
            push!(xyops,nothing) # exact one-point transverse moments vanish
        else
            push!(xyops,([neutral((p[i]+m[i])/2) for i in 1:3],
                         [neutral((p[i]-m[i])/(2im)) for i in 1:3]))
        end
    end
    real_checked(v)=begin
        maximag=max(maximag,abs(imag(v)))
        isfinite(v) || error("nonfinite Hermitian observable $v")
        real(v)
    end
    for x in 1:g.Lx,y in 1:g.Ly
        positions=((x,y),(x+1,y),(x,y+1),(x+1,y+1));names=(:LU,:RU,:LD,:RD)
        owners=[N.owner(g,p...) for p in positions]
        corners=[N.enlarged_corner(env,env.Q[mod1(p[1],g.Lx)][mod1(p[2],g.Ly)],
            env.R[mod1(p[1],g.Lx)][mod1(p[2],g.Ly)],names[i],x-1,y-1,g)
            for (i,p) in enumerate(positions)]
        normal=N.close_corners(corners)
        abs(normal)>1e-250 || error("invalid CTM norm")
        function inserted(i,op;charged=false)
            r=owners[i];uq,ur=N.factor_fusers(f,r)
            q=N.double_Q(f.Q[r],uq;operator=op,charged);p=positions[i]
            N.enlarged_corner(env,q,env.R[mod1(p[1],g.Lx)][mod1(p[2],g.Ly)],names[i],x-1,y-1,g;charged)
        end
        function expectation(replacements;charged=())
            cs=Any[corners...]
            for (i,v) in replacements;cs[i]=v;end
            N.close_corners(cs,charged)/normal
        end
        op=ops[owners[1]];z0=[inserted(1,a) for a in op.z]
        mz=[real_checked(expectation(Dict(1=>v))) for v in z0]
        xy=xyops[owners[1]]
        mx=isnothing(xy) ? zeros(3) : [real_checked(expectation(Dict(1=>inserted(1,a)))) for a in xy[1]]
        my=isnothing(xy) ? zeros(3) : [real_checked(expectation(Dict(1=>inserted(1,a)))) for a in xy[2]]
        up=[real_checked(expectation(Dict(1=>inserted(1,a)))) for a in op.pairs]
        kup=[real_checked(expectation(Dict(1=>inserted(1,a)))) for a in dmops[owners[1]]]
        # Down order B2(R),B3(R+ey),B1(R+ex) is a cyclic shift of legacy
        # simplex axes, hence still CLOCKWISE. No reversal is introduced here.
        specs=((1,2),(3,3),(2,1));localz=Any[];localp=Any[];localm=Any[]
        for (i,b) in specs
            op=ops[owners[i]]
            push!(localz,i==1 ? z0[b] : inserted(i,op.z[b]))
            push!(localp,inserted(i,op.plus[b];charged=true))
            push!(localm,inserted(i,op.minus[b];charged=true))
        end
        down=Float64[];kdown=Float64[]
        for (a,b) in CYCLE
            ca,cb=specs[a][1],specs[b][1];pair=Tuple(sort([ca,cb]))
            zz=expectation(Dict(ca=>localz[a],cb=>localz[b]))
            pm=expectation(Dict(ca=>localp[a],cb=>localm[b]);charged=pair)
            mp=expectation(Dict(ca=>localm[a],cb=>localp[b]);charged=pair)
            push!(down,real_checked(zz+.5*(pm+mp)))
            push!(kdown,real_checked(-.5im*(pm-mp)))
        end
        push!(cells,Dict("x"=>x,"y"=>y,"Sx"=>mx,"Sy"=>my,"Sz"=>mz,
            "bonds_up_SS"=>up,"bonds_down_SS"=>down,"kappa_up_CCW"=>kup,"kappa_down_CCW"=>kdown,
            "bonds_up_energy"=>J.*up+Dz.*kup,"bonds_down_energy"=>J.*down+Dz.*kdown))
    end
    n=3length(cells);mz=sum(sum(c["Sz"]) for c in cells)/n
    ex=J*sum(sum(c["bonds_up_SS"])+sum(c["bonds_down_SS"]) for c in cells)/n
    dm=Dz*sum(sum(c["kappa_up_CCW"])+sum(c["kappa_down_CCW"]) for c in cells)/n
    Dict("J"=>J,"Dz"=>Dz,"h"=>h,"energy_per_spin"=>ex+dm+h*mz,
        "exchange_energy_per_spin"=>ex,"dmi_energy_per_spin"=>dm,"field_energy_per_spin"=>h*mz,
        "mean_Sz"=>mz,"m_normalized"=>-2mz,"cells"=>cells,"max_hermitian_imaginary_part"=>maximag,
        "observables_hermiticity_ok"=>maximag<1e-8,
        "convention"=>"CCW every triangle; H=JSS+Dz*kappa+hSz (legacy Julia field sign)")
end

function measure(s;J=1.,Dz=0.,h=.38,chi=40,iterations=50,tolerance=1e-6,verbose=true)
    start=time();f=N.build_factors(s)
    env,info=N.native_ctmrg(f;chi,iterations,tolerance,verbose)
    result=dmi_observables(f,env;J,Dz,h)
    result["ctm"]=info;result["seconds"]=time()-start
    result["measurement_reliable"]=info["converged"] && result["observables_hermiticity_ok"]
    if !result["observables_hermiticity_ok"] && verbose
        @warn "CTM Hermiticity residual is large; these observables need larger chi/more CTM iterations" maximag=result["max_hermitian_imaginary_part"]
    end
    result
end
