include("run.jl")
# A separate log permits independent FU and AD runs on the same fixed SU
# checkpoint without two processes appending to the same progress file.
function note(directory,message)
    line="$(Dates.now()) $message"
    println(line); flush(stdout)
    open(io->println(io,line),joinpath(directory,"fu_progress.log"),"a")
end
seed=isempty(ARGS) ? 999 : parse(Int,only(ARGS))
run_flux_fu(;directory=joinpath(FLUX_OUTPUT,"seed$seed"),refine_tau=.05)
