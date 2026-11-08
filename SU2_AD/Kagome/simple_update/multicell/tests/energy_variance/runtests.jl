using Test,TensorKit,LinearAlgebra,TOML
ENV["KAGOME_VARIANCE_NATIVE_ROOT"]=get(ENV,"KAGOME_VARIANCE_NATIVE_ROOT",
    normpath(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","native_measurement")))
if !isdefined(Main,:KagomeEnergyVariance)
    include(get(ENV,"KAGOME_VARIANCE_MODULE",
        joinpath(ENV["KAGOME_VARIANCE_NATIVE_ROOT"],"energy_variance","KagomeEnergyVariance.jl")))
end
const V=KagomeEnergyVariance
const N=V.N
BLAS.set_num_threads(1)
include("test_support.jl")
report=Dict{String,Any}("TensorKit"=>string(pkgversion(TensorKit)),"julia"=>string(VERSION),
    "scope"=>"D1 product/trimer exact tests and D2 central-bond smoke; no production state evolved")

@testset "LC reverse moves use the original coordinate convention" begin
    for d in 1:4,j in 2:8
        origin=V.move_origin(j,d,9)
        p=V.C.convert_cell_posit(origin,5,1,2,d,9,9)
        @test p[d in (1,3) ? 1 : 2]==j
    end
end

@testset "all source/target bonds: analytic D1 product state, radius and shift" begin
    f,psi=product_factors()
    @test length(V.bond_catalogue(f.state.geometry))==18
    result=V.exchange_variance(f;L=9,radii=[0,1],chi=2,ctm_iterations=3,sweeps=1,verbose=true)
    @test result["full_anchor_sum"]
    source_terms=V.bond_catalogue(f.state.geometry)
    expected=0.;worst=0.
    for row in result["connected_terms"]
        source=(representative=row["source_representative"],orientation=Symbol(row["source_orientation"]),edge=row["source_edge"])
        exact=exact_product_correlator(f,psi,source,row["dx"],row["dy"],Symbol(row["target_orientation"]),row["target_edge"])
        val=row["connected_real"]+im*row["connected_imag"]
        worst=max(worst,abs(val-exact));@test val≈exact atol=1e-10
        expected+=real(exact)/9
    end
    @test length(result["connected_terms"])==18*9*6
    @test result["cumulative"][1]["variance_per_spin"]≈expected atol=1e-10
    @test result["cumulative"][2]["variance_per_spin"]≈expected atol=1e-10
    selected=[(representative=1,orientation=:down,edge=1)]
    other=V.exchange_variance(f;L=9,radii=[0,1],chi=2,ctm_iterations=3,sweeps=2,source_shift=2.,anchors=selected,verbose=false)
    @test !other["full_anchor_sum"]
    for row in other["connected_terms"]
        ref=only(filter(r->all(r[k]==row[k] for k in ("source_representative","source_orientation","source_edge","dx","dy","target_orientation","target_edge")),result["connected_terms"]))
        @test row["connected_real"]≈ref["connected_real"] atol=1e-10
    end
    report["product_D1"]=Dict("variance"=>expected,"max_pair_error"=>worst,"connected_pairs"=>length(result["connected_terms"]))
end

@testset "quantum trimer-product: exact 2D connected energy correlations" begin
    f,psi=product_factors(quantum=true)
    result=V.exchange_variance(f;L=9,radii=[0,1],chi=3,ctm_iterations=3,sweeps=2,verbose=true)
    expected=0.;worst=0.
    for row in result["connected_terms"]
        source=(representative=row["source_representative"],orientation=Symbol(row["source_orientation"]),edge=row["source_edge"])
        exact=exact_product_correlator(f,psi,source,row["dx"],row["dy"],Symbol(row["target_orientation"]),row["target_edge"])
        val=row["connected_real"]+im*row["connected_imag"]
        worst=max(worst,abs(val-exact));@test val≈exact atol=1e-9
        expected+=real(exact)/9
    end
    @test result["cumulative"][2]["variance_per_spin"]≈expected atol=1e-9
    @test abs(result["cumulative"][2]["imaginary_per_spin"])<1e-10
    report["trimer_product_D1"]=Dict("variance"=>expected,"max_pair_error"=>worst,"connected_pairs"=>length(result["connected_terms"]))
end

@testset "small entangled dense D2: central-bond algebra, no SU" begin
    f,_=product_factors();g=f.state.geometry;rng=MersenneTwister(701)
    vv=ComplexSpace(2);vp=ComplexSpace(2);onev=ComplexSpace(1)
    qs=Any[];rs=Any[];fusers=Any[]
    for _ in 1:3
        P=vv⊗vv⊗vv⊗vp⊗vp⊗vp⊗onev
        q=randn(rng,ComplexF64,P,one(P));push!(qs,q/norm(q))
        P=vv'⊗vv'⊗vv';r=randn(rng,ComplexF64,P,one(P));push!(rs,r/norm(r))
        push!(fusers,[unitary(fuse(vv'⊗vv),vv'⊗vv) for _ in 1:3])
    end
    f=N.NativeFactors((geometry=g,),qs,rs,fusers)
    env,info=N.native_ctmrg(f;chi=4,iterations=20,tolerance=1e-8,verbose=false)
    base=V.base_context(f,env,9);errors=Float64[]
    for o in (:up,:down),edge in 1:3
        e=V.bond_expectation(base,4,4,o,edge)
        ctx,cq,cr=V.central_context(f,env,9,4,4,o,edge)
        mix=V.large_cell_sweeps(ctx,unique([cq;cr]);chi=4,sweeps=2)
        variance=(1+e)*(V.bond_expectation(mix,4,4,o,edge)-e)
        exact=3/16-0.5*e-e^2 # exact operator identity for spin-1/2 S.S
        push!(errors,abs(variance-exact))
        @test abs(variance-exact)<1e-4
    end
    report["dense_D2"]=Dict("central_bond_max_error"=>maximum(errors),"baseline_ctm"=>info)
end
open(io->TOML.print(io,report;sorted=true),get(ENV,"KAGOME_VARIANCE_CHECK_OUT",joinpath(@__DIR__,"checks.toml")),"w")
println("VARIANCE CHECKS PASSED")
