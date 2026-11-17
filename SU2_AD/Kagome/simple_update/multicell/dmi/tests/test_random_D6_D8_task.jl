using Test, LinearAlgebra, TOML, SHA
include(joinpath(@__DIR__, "..", "server", "run_independent_random_D6_D8.jl"))
const R = KagomeDMIIndependentFields

@testset "Production plan only: no D=6/8 tensors" begin
    @test options.dims == [6, 8]
    @test options.trials == [1, 2, 3]
    @test extrema(options.Dz) == (0., .7)
    @test extrema(options.fields) == (.28, .64)
    cases = [(D,dz,h,t) for D in options.dims for dz in options.Dz for h in options.fields for t in options.trials]
    @test length(cases) == 792
    @test allunique(R.seed_for(options, c...) for c in cases)
    @test allunique(R.folder_for(options, c...) for c in cases)
    @test allunique(options.Dz) && allunique(options.fields)
    @test options.chi == 12 && options.ctm_max == 50 && options.ctm_tol == 1e-6
    @test options.threads == 10
    @test sum(round(Int, s.tau/s.dt) for s in options.fresh) == 9000
    @test isempty(options.warm)
    @test occursin("D6_D8_positive_Dz070", options.out)
end

println("Production plan checked; running TWO D=1 states with ONE short step each."); flush(stdout)
@testset "Small random initializations and point-local resume" begin
    mktempdir() do tmp
        cfg = merge(options, (dims=[1], Dz=[.02], fields=[.38], trials=[1,2],
            threads=2, chi=2, ctm_max=2, fresh=[(dt=.01,tau=.01)], out=tmp))
        R.run(cfg)
        @test isfile(joinpath(tmp, "COMPLETED.txt"))
        @test isfile(joinpath(tmp, "progress.csv"))
        initial_states = []
        hashes = String[]
        for trial in cfg.trials
            folder = R.folder_for(cfg, 1, .02, .38, trial)
            for file in ("initial.jld2", "stage.jld2", "final.jld2", "initialization.toml",
                         "initialization_audit.toml", "independent_result.toml")
                @test isfile(joinpath(folder,file))
            end
            audit = TOML.parsefile(joinpath(folder,"initialization_audit.toml"))
            @test audit["seed"] == R.seed_for(cfg,1,.02,.38,trial)
            @test audit["previous_field_source"] == "none" && audit["previous_D_source"] == "none"
            @test !audit["imposed_U1"] && !audit["imposed_C3"]
            @test audit["initial_checkpoint_sha256"] == bytes2hex(sha256(read(joinpath(folder,"initial.jld2"))))
            state,_ = R.P.K.load_checkpoint(joinpath(folder,"initial.jld2"))
            @test isempty(state.auxiliary_charges)
            push!(initial_states,state)
            push!(hashes,bytes2hex(sha256(read(joinpath(folder,"final.jld2")))))
        end
        @test norm(initial_states[1].cells[1].B_a-initial_states[2].cells[1].B_a)>1e-3
        R.run(cfg)
        for (i,trial) in enumerate(cfg.trials)
            folder = R.folder_for(cfg,1,.02,.38,trial)
            @test bytes2hex(sha256(read(joinpath(folder,"final.jld2")))) == hashes[i]
        end
    end
end
println("PASS: 792 unique planned points, independent small random kets and unchanged resume checkpoints.")
