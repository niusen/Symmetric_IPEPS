using LinearAlgebra
include("../../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
using .HoneycombSU4
BLAS.set_num_threads(1);
folder=isempty(ARGS) ? joinpath(@__DIR__,"../../logs/tests/D9_HOSVD_bond_audit") : abspath(ARGS[1]);
trace_honeycomb_star_hosvd(folder;sweeps=length(ARGS)>=2 ? parse(Int,ARGS[2]) : 3);
