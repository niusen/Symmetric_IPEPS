using Test,TensorKit,LinearAlgebra
native=get(ENV,"KAGOME_VARIANCE_NATIVE_ROOT",normpath(joinpath(@__DIR__,"..","..","..","..","..",
    "src","bosonic","kagome","native_measurement")))
if !isdefined(Main,:KagomeEnergyVariance)
    include(joinpath(native,"energy_variance","KagomeEnergyVariance.jl"))
end
if !isdefined(Main,:V);const V=KagomeEnergyVariance;end
if !isdefined(Main,:N);const N=V.N;end
isdefined(Main,:product_factors) || include(joinpath(@__DIR__,"test_support.jl"))
BLAS.set_num_threads(10)
f,psi=product_factors(quantum=true)
env,ctm=N.native_ctmrg(f;chi=2,iterations=3,tolerance=1e-10,verbose=false)

@testset "horizontal example: default 55 moves, former schedule opt-in" begin
    ctx,cq,cr=V.central_context(f,env,12,6,6,:down,3)
    support=unique([cq;cr]);seen=Tuple{Int,Int}[]
    V.large_cell_sweeps(ctx,support;chi=2,sweeps=2,
        callback=i->push!(seen,(i.direction,i.absorbed_coordinate)))
    expected=[(3,7)]
    for _ in 1:2
        append!(expected,[(1,j) for j in 5:11])
        append!(expected,[(3,j) for j in 8:-1:2])
        append!(expected,[(2,j) for j in 5:11])
        append!(expected,[(4,j) for j in 7:-1:2])
    end
    @test seen==expected
    @test length(seen)==55
    @test count(==((4,6)),seen)==2 # normal bottom scans remain in both sweeps
    empty!(seen)
    V.large_cell_sweeps(ctx,support;chi=2,sweeps=2,extra_bottom_warmup=true,
        callback=i->push!(seen,(i.direction,i.absorbed_coordinate)))
    @test seen==[expected[1],(4,6),expected[2:end]...]
    @test length(seen)==56
end

@testset "tiny storage check: tuple forwards the warmup option" begin
    # Only initialization on a 3x3 grid: a kernel/storage test, not a physical
    # finite-window variance calculation. Avoid large tuple compilation costs.
    ctx,cq,cr=V.central_context(f,env,3,1,1,:down,3)
    support=unique([cq;cr])
    for extra in (false,true)
        logs=[Tuple{Int,Int}[],Tuple{Int,Int}[]]
        a=V.large_cell_sweeps(ctx,support;chi=2,sweeps=0,extra_bottom_warmup=extra,
            callback=i->push!(logs[1],(i.direction,i.absorbed_coordinate)))
        b=V.large_cell_sweeps(ctx,support;chi=2,sweeps=0,storage=:tuple,extra_bottom_warmup=extra,
            callback=i->push!(logs[2],(i.direction,i.absorbed_coordinate)))
        @test logs[1]==logs[2]
        @test length(logs[1])==(extra ? 2 : 1)
        for x in 1:3,y in 1:3,k in 1:4
            @test norm(V.C.get_Cset(a.Cset[x][y],k)-V.C.get_Cset(b.Cset[x][y],k))<1e-11
            @test norm(V.C.get_Tset(a.Tset[x][y],k)-V.C.get_Tset(b.Tset[x][y],k))<1e-11
        end
    end
end

@testset "high-level default and explicit old option: exact trimer-product correlations" begin
    source=(representative=1,orientation=:down,edge=3)
    prepared=(f=f,environment=env,ctm=ctm,chi=2,iterations=3,tolerance=1e-10)
    for extra in (false,true)
        options=extra ? (;extra_bottom_warmup=true) : (;)
        result=V.exchange_variance(f;L=9,window_sizes=[1],chi=2,ctm_iterations=3,
            ctm_tolerance=1e-10,sweeps=2,anchors=[source],norm_environment=prepared,verbose=false,options...)
        @test result["extra_bottom_warmup"]===extra
        for row in result["connected_terms"]
            exact=exact_product_correlator(f,psi,source,row["dx"],row["dy"],Symbol(row["target_orientation"]),row["target_edge"])
            @test abs(row["connected_real"]+im*row["connected_imag"]-exact)<1e-10
        end
    end
end
