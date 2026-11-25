include("run.jl")

if abspath(PROGRAM_FILE)==@__FILE__
    include("tests/test_heisenberg.jl")
    for family in ("ipess","ipeps")
        run_spin_energy(family,4,16;tol=1e-7,maxiter=100)
        run_spin_energy(family,4,32;tol=1e-7,maxiter=100,warmstart="heisenberg_chi16_cut1.0e-10.jld2")
    end
end
