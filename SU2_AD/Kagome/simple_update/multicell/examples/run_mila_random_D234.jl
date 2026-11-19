# All options live here. Independent dense random states; no fixed magnetization.
mila_random_settings=(
    dims=[2,3,4],
    seeds=[1101,1102,1103],
    h=0.38,
    chi=16,
    threads=10,
    ctm_iterations=50,
    ctm_tolerance=1e-6,
    truncation_tolerance=1e-8,
    schedule=[(dt=.03,tau=40.),(dt=.01,tau=20.),(dt=.003,tau=5.),
              (dt=.001,tau=2.),(dt=.0005,tau=1.)],
    progress_every=500,
    checkpoint_every=500,
    out=joinpath(@__DIR__,"..","results","mila_random_dense_D234_20261030"),
)
if !isdefined(@__MODULE__,:KAGOME_SETTINGS_ONLY)
    include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","workflows","mila_random_dense.jl"))
    KagomeMilaRandom.run_search(mila_random_settings)
end
