# Edit options HERE, then run this file; no command-line parameters.
# Activate the existing multicell TensorKit environment. CPU is the default.
import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","KagomeDMI.jl"))
options=(
    representation=:u1, # :dense with input="" gives unrestricted random tensors
    input=joinpath(@__DIR__,"..","..","data","u1_D4_optimized_seed.toml"),
    out=joinpath(@__DIR__,"..","..","results","dmi_u1_D4"),
    D=4,seed=91,Lx=3,Ly=3,pattern=:sqrt3xsqrt3, # or :rectangular
    J=1.,Dz=0.,h=.38, # independent couplings; pure DMI: J=0,Dz=1,h=0
    threads=10,trun_tol=1e-8,progress_every=100,
    schedule=[(dt=.03,tau=40.),(dt=.01,tau=20.),(dt=.003,tau=5.),
              (dt=.001,tau=2.),(dt=.0005,tau=1.)],
    chi=40,ctm_iterations=50,ctm_tolerance=1e-6,
)
KagomeDMI.run(options)
