include("run_heisenberg.jl")

function projected_D8_suite()
    root=joinpath(KAGOME_WORKSPACE,"Kagome/fermionic/iPESS/results/gfpeps_validation")
    cases=(("ipeps",joinpath(root,"ipeps_D8")),
           ("ipess",joinpath(root,"ipess_D8")),
           ("ipess",joinpath(root,"ipess_D8_seed12103")))
    # Obtain the missing small-environment estimates first, then refine all three.
    for chi in (16,32)
        tag="heisenberg_chi$(chi)_cut1.0e-8_gauge8"
        for (family,directory) in cases
            file=joinpath(directory,tag*".toml")
            if isfile(file) && isfile(joinpath(directory,tag*".jld2"))
                println("Keeping completed run: ",file);flush(stdout)
                continue
            end
            initial=chi==16 ? nothing : "heisenberg_chi16_cut1.0e-8_gauge8.jld2"
            run_spin_energy(family,8,chi;cutoff=1e-8,maxiter=chi==16 ? 60 : 40,
                tol=1e-7,gauge_sweeps=8,warmstart=initial,directory_override=directory)
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    projected_D8_suite()
end
