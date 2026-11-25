include(joinpath(@__DIR__, "../../../src/fermionic/kagome/workflows/compute_observables.jl"))
main(args=ARGS)=compute_saved_observables(args)
if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
