# A separate diagnostic: do not mix these additional SU seeds into the
# matched-initialization SU/FU/AD table produced by run.jl.
include("run.jl")
parameters = Dict{String,Any}("t_up"=>1.0,"t_dn"=>0.2,"U"=>0.0,"μ"=>-0.6,"centered"=>true)
exact = exact_reference()
for seed in (1,42,889)
    directory = joinpath(OUTPUT,"initialization","seed$(seed)"); mkpath(directory)
    checkpoint = joinpath(directory,"su.jld2")
    if isfile(checkpoint)
        state = load_checkpoint(checkpoint)[1]
    else
        state = initial_iPESS(1,1,Rep[ℤ₂](0=>2,1=>2);rng=MersenneTwister(seed))
        for (stage,(tau,dt)) in enumerate(SU_SCHEDULE)
            elapsed = @elapsed itebd!(state,parameters,tau,dt,4;trun_tol=1e-12)
            note(directory,"SU seed=$seed stage=$stage seconds=$elapsed sectors=$(sectors(state))")
        end
        save_checkpoint(checkpoint,state;parameters,metadata=Dict("seed"=>seed,"sectors"=>sectors(state)))
    end
    measure(directory,"su",state,parameters,exact)
end
