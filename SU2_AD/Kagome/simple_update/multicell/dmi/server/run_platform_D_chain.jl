# Task 1: each h has its own D=2 -> 10 lineage. All settings are HERE.
import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","plateau_continuation.jl"))
platform_seed=joinpath(@__DIR__,"..","..","results","DMI_independent_random3_D2_D4_20261114",
    "D2_Dz0.02_random1","h0.32","final.jld2")
options=(
    mode=:D_chain,dims=collect(2:10),Dz_values=[.02],
    fields=[.28,.32,.36,.38,.40,.44,.48],
    input=platform_seed,seed_observables=joinpath(dirname(platform_seed),"independent_result.toml"),
    seed_D=2,seed_Dz=.02,seed_h=.32,seed_magnetization=1/9,seed_m_tolerance=1e-5,
    threads=10,trun_tol=1e-8,chi=12,ctm_max=50,ctm_tol=1e-6,
    schedule=[(dt=.03,tau=40.),(dt=.01,tau=20.),(dt=.003,tau=5.),(dt=.001,tau=2.),(dt=.0005,tau=1.)],
    out=joinpath(@__DIR__,"..","..","results","DMI_platform_D2_D10_Dz002_sqrt3_chi12"),
)
KagomeDMIPlateauContinuation.run(options)
