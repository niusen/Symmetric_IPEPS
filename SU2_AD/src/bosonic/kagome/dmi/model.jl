function triangle_hamiltonian(;J=1.,Dz=0.,h=.38,representation=:u1)
    all(isfinite,(J,Dz,h)) || error("nonfinite couplings")
    representation in (:u1,:dense) || error("Dz breaks SU2: use :u1 or :dense")
    base=representation==:u1 ? SU.u1_heisenberg_triangle() : N.dense_tensor(BS.heisenberg_triangle())
    z=[N.embedded_operator(i,N.SZ) for i in 1:3]
    p=[N.embedded_operator(i,N.SP) for i in 1:3];m=adjoint.(p)
    # Both legacy triangle orders are clockwise, so NEGATE the cyclic kappa.
    # J and Dz are independent; J=0 must not turn the DMI interaction off.
    dm=sum(-.5im*(p[a]*m[b]-m[a]*p[b]) for (a,b) in CYCLE)
    tensor(a)=TensorMap(reshape(a,ntuple(_->2,6)),codomain(base),domain(base);tol=1e-11)
    # Each physical site belongs to two triangles: half field per triangle.
    J*base+Dz*tensor(dm)+(h/2)*tensor(sum(z))
end

function random_dense_state(D,seed;Lx=3,Ly=3,pattern=:sqrt3xsqrt3)
    D>0 || error("D must be positive")
    g=BS.KagomeSUCellGeometry(Lx,Ly;pattern);rng=MersenneTwister(seed)
    V=ComplexSpace(D);P=ComplexSpace(2);cells=BS.KagomeSUTensors[]
    for _ in g.representatives
        b=[permute(randn(rng,ComplexF64,V'⊗V',P'),((1,2,3),())) for _ in 1:3]
        t=[permute(randn(rng,ComplexF64,V,V'⊗V'),((1,2,3),())) for _ in 1:2]
        b=[x/norm(x) for x in b];t=[x/norm(x) for x in t]
        w=[unitary(space(x,2)',space(x,2)') for x in b]
        push!(cells,BS.KagomeSUTensors(t...,b...,w...,nothing,nothing,nothing))
    end
    N.DenseKagomeState(g,cells,Dict{Int,Int}())
end

"""D=1 q=0 planar product reference, not the exact quantum ground state."""
function product_120(;chirality=1,Lx=1,Ly=1,pattern=:rectangular)
    s=random_dense_state(1,1;Lx,Ly,pattern)
    for c in s.cells
        for (i,name) in enumerate(BS.B_FIELDS)
            t=getfield(c,name);theta=chirality*2pi*(i-1)/3
            a=reshape(ComplexF64[1,exp(im*theta)]/sqrt(2),(1,1,2))
            setfield!(c,name,TensorMap(a,codomain(t),domain(t)))
        end
        for name in (:T_u,:T_d)
            t=getfield(c,name);setfield!(c,name,TensorMap(ones(ComplexF64,1,1,1),codomain(t),domain(t)))
        end
    end
    s
end

"""Legacy SU gate sequence and truncation kernels, with a new Hamiltonian only.

No charge restriction for dense states. U1 states retain their existing charge
sector; enlarging D does not change the total auxiliary charge/magnetization.
"""
function evolve!(s;schedule,Dmax,J=1.,Dz=0.,h=.38,trun_tol=1e-8,callback=nothing)
    rep=s isa SU.U1KagomeState ? :u1 : :dense
    H=triangle_hamiltonian(;J,Dz,h,representation=rep)
    Dmax>0 && 0<=trun_tol<1 || error("invalid truncation")
    for (stage,cfg) in enumerate(schedule)
        dt=cfg.dt;tau=cfg.tau
        isfinite(dt) && dt>0 && isfinite(tau) && tau>=0 || error("invalid time stage")
        n=round(Int,tau/dt);gate,half=SU.Kernels.trotter_gate(H,dt)
        SU._sweep!(s,half,:dn;Dmax,trun_tol)
        for step in 1:n
            SU._sweep!(s,gate,:up;Dmax,trun_tol)
            SU._sweep!(s,gate,:dn;Dmax,trun_tol)
            isnothing(callback) || callback(s,stage,step,n)
        end
        SU._sweep!(s,half,:up;Dmax,trun_tol)
    end
    s isa SU.U1KagomeState && SU.check_u1(s;Dmax)
    s
end
