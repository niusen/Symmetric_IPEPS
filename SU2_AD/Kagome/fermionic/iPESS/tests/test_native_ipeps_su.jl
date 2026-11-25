using Test, TensorKit, LinearAlgebra
include("../../../../src/fermionic/kagome/ipeps/bootstrap.jl")
using .KagomeFermionicZ2, .KagomeGFPEPSValidation, .KagomeSpinOptimization, .KagomeIPEPSSU
const KF=KagomeFermionicZ2
const NSU=KagomeIPEPSSU
BLAS.set_num_threads(1)

# Reference in the notes' electron gauge, basis (source A,B,C), not b-hole.
function source_bloch(k,phi)
    M=zeros(ComplexF64,9,9)
    for x in 0:2, down in (false,true)
        q=down ? 1 : 0
        sites=((x,0,0),(x-q,0,1),(x,-q,2))
        for (e,(i,j)) in enumerate(((1,2),(2,3),(3,1)))
            a,y,s=sites[i]; b,z,t=sites[j]
            val=cis(phi/3+(0.,-2pi/3*x*q,2pi/3*x*q)[e]-k[1]*(fld(b,3)-fld(a,3))-k[2]*(z-y))
            u=3mod(a,3)+s+1; v=3mod(b,3)+t+1
            M[v,u]+=val;M[u,v]+=conj(val)
        end
    end
    M
end

@testset "three spin-resolved Bloch models, including cross-cell gauge" begin
    ops=KF._triangle_operators()
    for (kind,phase) in ((:orig,-pi/8),(:mild,-pi/16),(:zero,0.)), s in 1:2
        model=parton_model(kind;mu=(0.,0.));k=(.37,-1.12)
        M=zeros(ComplexF64,9,9)
        for x in 1:3,o in (:up,:dn)
            H=reshape(convert(Array,Hamiltonian_triangle(model;orientation=o,x)),64,64)
            # Positions mapped back to the notes: b=source A, a=B, c=C.
            u=mod(1-x,3); q=Int(o===:dn)
            pos=((u-q,0,1),(u,0,0),(u,-q,2))
            occ=[1+(s+1)*4^(j-1) for j in 1:3]
            for (i,j) in ((1,2),(2,3),(3,1))
                coeff=H[occ[i],occ[j]]
                z=s==1 ? coeff : -conj(coeff)
                a,y,b=pos[i];c,v,d=pos[j]
                m=3mod(a,3)+b+1;n=3mod(c,3)+d+1
                z*=cis(-k[1]*(fld(a,3)-fld(c,3))-k[2]*(y-v))
                M[m,n]+=z;M[n,m]+=conj(z)
            end
        end
        @test M ≈ source_bloch(k,s==1 ? phase : -pi/8)
    end
end
function initial_product(spin,occ;unitcell=nothing)
    P=spin ? projected_spin_space() : KF.physical_space()
    d=dim(P);U=unitary(fuse(P*P*P),P*P*P)
    v=zeros(ComplexF64,d,d,d);v[occ...]=1
    fv=reshape(convert(Array,U),d^3,d^3)*vec(v)
    V=Rep[ℤ₂](0=>1)
    a=TensorMap(reshape(fv,1,1,1,1,d^3),V*V*V'*V'*space(U,1),one(V))
    SquareState(((a,),(a,),(a,)),U;unitcell)
end

@testset "3x3 periodic cell and disjoint triangle gates" begin
    layers=NSU.down_layers(3,3)
    @test length.(layers)==[3,3,3]
    sites=Set(CartesianIndices((3,3)))
    @test Set(Iterators.flatten(layers))==sites
    for layer in layers
        blocks=[CartesianIndex(p) for site in layer for p in NSU.down_blocks(Tuple(site)...,3,3)]
        @test length(blocks)==9 && Set(blocks)==sites
    end
    sq=initial_product(false,(4,3,1);unitcell=(3,3))
    @test size(sq.A)==(3,3) && size(sq.weight)==(3,3,4)
    @test sq.A[1,1]==sq.A[1,3] && sq.A[1,1]!==sq.A[1,3]
    @test_throws ArgumentError NSU.tile_su_cell(sq,(3,2))
    s=initial_iPESS(3,1)
    tiled=NSU.tile_su_cell(s,(3,3))
    @test check_state(tiled)===tiled
    for field in (:T_u,:T_d,:B,:lambda_u,:lambda_d)
        original=getproperty(s,field); copies=getproperty(tiled,field)
        for i in CartesianIndices(copies)
            x,y,rest...=Tuple(i)
            @test copies[i]==original[x,1,rest...]
            @test copies[i]!==copies[x,mod1(y+1,3),rest...]
        end
    end
    for orientation in (:up,:dn)
        positions=[p for site in sites for p in triangle_sites(tiled,Tuple(site)...,orientation)]
        @test length(positions)==length(unique(positions))==27
    end
end

@testset "isolated down trimer: exact state versus native PEPS and independent CTM" begin
    # Last configuration leaves odd spectator modes in each of two blocks.
    for (spin,occ) in ((true,(1,2,1)),(false,(1,2,1)),(false,(4,3,1)))
        state=initial_product(spin,occ);x,y=2,1;P=domain(state.fusion)[1];d=dim(P)
        m=spin ? HeisenbergField(1.,.5) : parton_model(:mild;mu=(-.03,-.8))
        H=KF.triangle_at(m,:dn,x)
        g=first(trotter_gate(H,.19))
        v=zeros(ComplexF64,d^3);v[LinearIndices((d,d,d))[occ...]]=1
        v=reshape(convert(Array,g),d^3,d^3)*v;normalize!(v)
        loss=NSU.down!(state,x,y,NSU.reverse_gate(g,P),8,1e-13)
        @test loss < 1e-12
        A=ctm_cell(state)
        if spin
            r=measure_heisenberg(A,state.fusion;J=m.J,h=m.h,chi=12,maxiter=8,verbose=false)
            ops=blocked_spin_operators(state.fusion)
            for (e,(i,j)) in enumerate(((1,2),(2,3),(3,1)))
                O=sum(s[i]*s[j] for s in (ops.Sx,ops.Sy,ops.Sz))
                B=reshape(convert(Array,state.fusion'*O*state.fusion),8,8)
                @test r.down_bonds[x,y,e] ≈ dot(v,B*v) atol=2e-8
            end
        else
            r=measure_gaussian_cell(A,state.fusion,m.mu;model=m,chi=12,maxiter=8,verbose=false)
            ops=KF._triangle_operators()
            for (e,(i,j)) in enumerate(((1,2),(2,3),(3,1))), s in 1:2
                @test r.hoppings[x,e,s] ≈ dot(v,ops.c[i][s]'*ops.c[j][s]*v) atol=2e-8
            end
        end
    end
end

@testset "physical Hubbard parameters on the particle-hole MAT basis" begin
    U=1.3; mu=(.2,-.4); t=(.7,1.2)
    H=reshape(convert(Array,Hamiltonian_triangle(hole_hubbard(;U,mu,t);onsite_fraction=1.)),64,64)
    # b vacuum means one down electron. b-up alone means doubly occupied.
    @test H[1,1] ≈ -3mu[2]
    @test H[3,3] ≈ U-mu[1]-3mu[2]
    @test H[3,9] ≈ -t[1]
    @test H[4,13] ≈ t[2]
end
