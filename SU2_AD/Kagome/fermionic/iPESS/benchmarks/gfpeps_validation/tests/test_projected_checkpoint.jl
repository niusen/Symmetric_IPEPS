include("../run.jl")
using Test
@testset "Blocked spin operator site order" begin
    S=projected_spin_space();U=unitary(fuse(S*S*S),S*S*S)
    ops=blocked_spin_operators(U);localops=projected_spin_operators()
    u=reshape(convert(Array,U),8,8);id=Matrix{ComplexF64}(I,2,2)
    for key in (:Sx,:Sy,:Sz),site in 1:3
        factors=[i==site ? convert(Array,getproperty(localops,key)) : id for i in 1:3]
        expected=u*kron(factors[3],factors[2],factors[1])*u'
        @test convert(Array,getproperty(ops,key)[site])≈expected
    end
end
@testset "Projected spin checkpoint format" begin
    raw=initial_iPESS(1,1,Rep[ℤ₂](0=>1,1=>1))
    before=deepcopy(raw.B)
    state=gutzwiller_project(raw)
    @test raw.B==before
    mktempdir() do dir
        path=joinpath(dir,"spin.jld2")
        save_checkpoint(path,state;parameters=Dict("J"=>1.0),metadata=Dict("projected"=>true))
        loaded,p,m=load_projected_checkpoint(path)
        @test loaded isa ProjectedSpinIPESS
        @test loaded.B==state.B && loaded.T_u==state.T_u && loaded.T_d==state.T_d
        @test loaded.lambda_u==state.lambda_u && loaded.lambda_d==state.lambda_d
        @test p["J"]==1 && m["projected"]
        @test_throws ArgumentError load_checkpoint(path)
    end
end
