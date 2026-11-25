include(joinpath(@__DIR__, "../../../../../../src/fermionic/kagome/gfpeps/diagnostics/spin_sector.jl"))
@testset "Gaussian charge audit independent small states" begin
    p=.3;u=sqrt(1-p);v=sqrt(p)
    w=ComplexF64[0 v; -v 0; u 0; 0 u]
    P=w*w';prob=occupation_probabilities(P);mom=number_moments(P)
    @test prob≈[1-p,0,0,p] atol=1e-14
    @test mom.mean≈2p
    @test mom.variance≈4p*(1-p)
    @test projected_sectors(prob).mz_over_saturation≈2p-1
    P=Matrix(Diagonal(ComplexF64[1,0,0,1]))
    @test occupation_probabilities(P)≈[0,1,0,0] atol=1e-14
    @test abs(number_moments(P).variance)<1e-14
    P=Matrix(Diagonal(ComplexF64[1,0,1,0,0,1,0,1]))
    sectors=projected_sectors(occupation_probabilities(P))
    @test sectors.weights≈[0,1,0] atol=1e-14
    @test sectors.mz_over_saturation==0
end
