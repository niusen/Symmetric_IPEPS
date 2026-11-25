include("tests/test_spin_sector.jl")
BLAS.set_num_threads(1)
if abspath(PROGRAM_FILE)==@__FILE__
    @testset "Frozen archives: explicit Fock probabilities versus Wick moments" begin
        for family in ("ipess","ipeps"),D in (4,8)
            audit_saved_sector(family,D)
        end
    end
end
