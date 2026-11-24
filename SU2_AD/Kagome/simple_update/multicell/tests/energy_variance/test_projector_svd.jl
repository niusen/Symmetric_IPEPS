# Exact QR vs the original direct product, SMALL tensors only. No SU/server scan.
using Test,TensorKit,LinearAlgebra,Random,TOML
native=get(ENV,"KAGOME_VARIANCE_NATIVE_ROOT",normpath(joinpath(@__DIR__,"..","..","..","..","..",
    "src","bosonic","kagome","native_measurement")))
include(joinpath(native,"energy_variance","KagomeEnergyVariance.jl"))
const V=KagomeEnergyVariance;const N=V.N
BLAS.set_num_threads(2)
println("TensorKit ",pkgversion(TensorKit),"; testing exact TensorMap QR with no dense conversion");flush(stdout)

@testset "exact product SVD: dense and U1 spaces" begin
    rng=MersenneTwister(1729)
    z=U1Space(0=>1)
    spaces=((ComplexSpace(4)⊗ComplexSpace(3),ComplexSpace(2)⊗ComplexSpace(2),ComplexSpace(5)⊗ComplexSpace(3)),
        (U1Space(0=>6,1=>5,-1=>4)⊗z,U1Space(0=>2,1=>1,-1=>1)⊗z,U1Space(0=>7,1=>5,-1=>5)⊗z))
    for (out,middle,inp) in spaces
        a=randn(rng,ComplexF64,out,middle);b=randn(rng,ComplexF64,middle,inp)
        a0=copy(a);b0=copy(b)
        @test V.variance_reduces_product(a,b)
        qa,ra=V.variance_left_qr(a);qb,rb=V.variance_left_qr(b')
        @test norm(a-qa*ra)<1e-11
        @test norm(b' -qb*rb)<1e-11
        @test norm(a*b)≈norm(ra*rb') atol=1e-11 rtol=1e-12
        @test dim(codomain(ra))<dim(codomain(a))
        for chi in (2,3,4)
            ud,sd,vd=V.variance_product_svd(a,b,chi,1e-12;algorithm=:direct)
            uq,sq,vq=V.variance_product_svd(a,b,chi,1e-12)
            @test sectortype(uq)==sectortype(a)
            @test codomain(sd)==codomain(sq)
            @test norm(sd-sq)<1e-11
            @test norm(ud*sd*vd-uq*sq*vq)<1e-11
            function oblique(u,s,v)
                inv=complex(V.C.safe_inv_sqrt(s/norm(s),1e-12))
                return (b*v'*inv)*(inv*u'*a)
            end
            @test norm(oblique(ud,sd,vd)-oblique(uq,sq,vq))<1e-10
        end
        @test a==a0 && b==b0
    end
end

include("test_support.jl")
function dense_D2_fixture()
    g=N.BS.KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3)
    rng=MersenneTwister(1730);vv=ComplexSpace(2);vp=ComplexSpace(2);aux=ComplexSpace(1)
    qs=Any[];rs=Any[];fusers=Any[]
    for _ in 1:3
        p=vv⊗vv⊗vv⊗vp⊗vp⊗vp⊗aux;q=randn(rng,ComplexF64,p,one(p));push!(qs,q/norm(q))
        p=vv'⊗vv'⊗vv';r=randn(rng,ComplexF64,p,one(p));push!(rs,r/norm(r))
        push!(fusers,[unitary(fuse(vv'⊗vv),vv'⊗vv) for _ in 1:3])
    end
    N.NativeFactors((geometry=g,),qs,rs,fusers)
end

@testset "old/new SMALL-D variance and individual connected observations" begin
    u1,_=product_factors(quantum=true)
    for (label,f,anchors) in (("U1 D1",u1,nothing),
            ("dense D2",dense_D2_fixture(),[(representative=1,orientation=:down,edge=2)]))
        env,ctm=N.native_ctmrg(f;chi=2,iterations=2,tolerance=1e-8,verbose=false)
        prepared=(f=f,environment=env,ctm=ctm,chi=2,iterations=2,tolerance=1e-8)
        options=(L=9,window_sizes=[1],chi=2,ctm_iterations=2,ctm_tolerance=1e-8,
            sweeps=1,anchors=anchors,norm_environment=prepared,verbose=false)
        println("Comparing ",label," direct vs QR variance");flush(stdout)
        old=V.exchange_variance(f;options...,projector_svd=:direct)
        new=V.exchange_variance(f;options...,projector_svd=:auto)
        @test old["exchange_energy_per_spin"]≈new["exchange_energy_per_spin"] atol=1e-12
        @test length(old["connected_terms"])==length(new["connected_terms"])
        errors=Float64[]
        for (a,b) in zip(old["connected_terms"],new["connected_terms"])
            push!(errors,hypot(a["connected_real"]-b["connected_real"],a["connected_imag"]-b["connected_imag"]))
            @test last(errors)<2e-8
        end
        @test old["cumulative"][1]["variance_per_spin"]≈new["cumulative"][1]["variance_per_spin"] atol=2e-8
        println(label," max connected error=",maximum(errors),"; variance difference=",
            abs(old["cumulative"][1]["variance_per_spin"]-new["cumulative"][1]["variance_per_spin"]));flush(stdout)
    end
end
println("EXACT VARIANCE PRODUCT QR PASSED; no production state or large-D run")
