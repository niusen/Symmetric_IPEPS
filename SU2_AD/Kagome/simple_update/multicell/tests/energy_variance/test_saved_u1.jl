# Extra integration check on the existing SMALL D4 portable seed. No SU.
using Test,TensorKit,LinearAlgebra,TOML,SHA
native=ENV["KAGOME_VARIANCE_NATIVE_ROOT"]
if !isdefined(Main,:KagomeEnergyVariance)
    include(get(ENV,"KAGOME_VARIANCE_MODULE",joinpath(native,"energy_variance","KagomeEnergyVariance.jl")))
end
const V=KagomeEnergyVariance
const N=V.N
# Re-read the independent workflow file for the integration/API test.
# This also lets a long small-D kernel test finish before testing new I/O code.
module_file=get(ENV,"KAGOME_VARIANCE_MODULE",joinpath(native,"energy_variance","KagomeEnergyVariance.jl"))
Base.include(V,joinpath(dirname(module_file),"variance_workflow.jl"))
const ServerU1=N.KagomeU1Cells
const ServerBase=N.BS
isdefined(Main,:load_u1_seed) || include(joinpath(native,"u1_seed_io.jl"))
root=normpath(joinpath(native,"..","..","..",".."))
file=joinpath(root,"Kagome","simple_update","multicell","data","u1_D4_optimized_seed.toml")
before=bytes2hex(sha256(read(file)));state=load_u1_seed(file)
BLAS.set_num_threads(1)
f=N.build_factors(state);env,info=N.native_ctmrg(f;chi=4,iterations=12,tolerance=1e-8,verbose=true)
baseline=V.base_context(f,env,9);errors=Float64[];energies=Float64[]
@testset "saved U1 D4: exact same-bond identity before LC truncation" begin
    for r in 1:3,o in (:up,:down),edge in 1:3
        x=r+3;y=4;e=V.bond_expectation(baseline,x,y,o,edge);push!(energies,real(e)/9)
        ctx,_,_=V.central_context(f,env,9,x,y,o,edge)
        value=(1+e)*(V.bond_expectation(ctx,x,y,o,edge)-e)
        error=abs(value-(3/16-0.5*e-e^2));push!(errors,error)
        @test error<1e-10
    end
    old=N.native_observables(f,env;h=.38)
    complete_mean=sum(real(V.bond_expectation(baseline,x,y,o,e)) for x in 1:3 for y in 1:3
        for o in (:up,:down) for e in 1:3)/27
    @test abs(complete_mean-old["exchange_energy_per_spin"])<1e-12
    @test bytes2hex(sha256(read(file)))==before
end
@testset "saved U1 D4: mixed enlarged charge spaces through LC sweep" begin
    ctx,cq,cr=V.central_context(f,env,9,4,4,:down,2)
    mixed=V.large_cell_sweeps(ctx,unique([cq;cr]);chi=4,sweeps=1)
    value=V.bond_expectation(mixed,4,4,:down,2)
    @test isfinite(real(value)) && isfinite(imag(value))
    println("Saved U1 D4 LC central bond = ",value)
end
@testset "saved U1 D4: checkpoint transport and all result tables, one anchor only" begin
    mktempdir() do tmp
        input=joinpath(tmp,"small_D4.jld2");output=joinpath(tmp,"measurement")
        N.KagomeU1Cells.save_u1_checkpoint(input,state,N.KagomeU1Cells.u1_heisenberg_triangle())
        input_hash=bytes2hex(sha256(read(input)))
        result=V.measure_variance_checkpoint(input,output;threads=1,L=9,radii=[0],chi=2,
            ctm_iterations=8,sweeps=1,anchors=[(representative=1,orientation=:down,edge=2)],verbose=false)
        @test !result["full_anchor_sum"]
        @test isfinite(result["cumulative"][1]["variance_per_spin"])
        @test result["cumulative"][1]["shell_absolute_sum_per_spin"]>=abs(result["cumulative"][1]["variance_per_spin"])
        @test length(result["connected_terms"])==6
        @test all(isfile(joinpath(output,name)) for name in
            ("variance.json","variance.toml","variance_by_radius.csv","connected_terms.csv"))
        @test bytes2hex(sha256(read(input)))==input_hash
        @test TOML.parsefile(joinpath(output,"variance.toml"))["source_sha256"]==input_hash
    end
end
println("SAVED U1 D4 PASSED; maximum exact insertion error=",maximum(errors))
