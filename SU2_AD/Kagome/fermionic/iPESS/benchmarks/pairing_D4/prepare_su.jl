include("run.jl")
for delta in (isempty(ARGS) ? [1.0,0.5] : parse.(Float64,ARGS))
    run_pairing(delta;su_only=true)
end
