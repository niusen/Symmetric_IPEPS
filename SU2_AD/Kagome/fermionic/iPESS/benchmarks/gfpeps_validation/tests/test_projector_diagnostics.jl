include("../run.jl")
using Test
function test_projector_diagnostics()
    C=KagomeFermionicCTM.Legacy
    previous=(C.chi,C.projector_trun_tol)
    V=Rep[ℤ₂](0=>3,1=>3)
    _,S,_=C.tsvd(randn(ComplexF64,V,V))
    @testset "Read-only projector instrumentation" begin
        try
            for chi in (2,4,6),tol in (1e-10,0.2)
                C.chi=chi;C.projector_trun_tol=tol
                baseline=invoke(C.sdiag_inv_sqrt,Tuple{DiagonalTensorMap},S)
                actual=C.sdiag_inv_sqrt(S)
                @test actual==baseline
                @test last(C.gfpeps_projector_ranks)<=chi
            end
        finally
            C.chi,C.projector_trun_tol=previous
            empty!(C.gfpeps_projector_ranks)
        end
    end
end
test_projector_diagnostics()
