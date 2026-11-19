# Recontract only the lowest measured-energy seed at each D. No SU evolution.
KAGOME_SETTINGS_ONLY=true
include("run_mila_random_D234.jl")
include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","workflows","mila_random_dense.jl"))
using TOML, LinearAlgebra
const Search=KagomeMilaRandom
cfg=mila_random_settings
BLAS.set_num_threads(cfg.threads)
rechecked=Dict[]
for D in cfg.dims
    candidates=[TOML.parsefile(joinpath(cfg.out,"D$(D)_seed$(seed)","summary.toml")) for seed in cfg.seeds]
    best=argmin(r->r["energy"],candidates)
    folder=best["folder"]
    result=Search.Native.measure_checkpoint(joinpath(folder,"final.jld2"),joinpath(folder,"ctm_chi24");
        representation=:dense,h=cfg.h,chi=24,iterations=50,tolerance=1e-6,device="cpu",threads=cfg.threads,D)
    row=Search.summary_row(result,D,best["seed"],best["lambda_drift_per_step"],best["seconds_su"],joinpath(folder,"ctm_chi24"))
    row["energy_chi16"]=best["energy"];row["delta_energy_chi24_minus_chi16"]=row["energy"]-best["energy"]
    push!(rechecked,row)
    Search.Native.write_csv(joinpath(cfg.out,"best_chi24_summary.csv"),rechecked)
    Search.Native.write_json(joinpath(cfg.out,"best_chi24_summary.json"),rechecked)
end
