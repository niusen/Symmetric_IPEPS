# Julia U1 conversion, simple update and measurement

Use run_kagome_settings.jl and edit the options in the file; no command-line
parameters are needed. task=:charged_trial / :simple_update retain the existing
SU2 and U1 simple-update kernels and initial-state conventions.

For task=:convert, set conversion_options.input to a SU2 or U1 JLD2 checkpoint.
representation=:u1 is the default; :dense creates dense TensorKit tensors and
ordinary Julia arrays. component=1 selects the auxiliary highest weight.
measure=true computes infinite-system CTMRG and all observables in Julia;
measure=false only converts/saves. CPU is the default; select device="cuda:1"
explicitly for GPU. Select chi and
CTMRG tolerance/cap in this same options block. No Python subprocess is used.

The fixed one-dimensional auxiliary charge is retained in U1 TensorMaps.
Restriction is exact: virtual multiplicities and arrows are not truncated.
The physical convention is H=Hex+h*sum(Sz), m=-2*mean(Sz), q=2Sz.
Magnetization1/9 is the selected auxiliary sector and does not establish a
finite plateau width. The field term is constant in the chosen sector and
is added once during measurement.

Core source functions are in src/bosonic/kagome/simple_update_u1_cells.jl
and src/bosonic/kagome/native_measurement/. The native CTMRG retains separate
Q/R factors, handles charged transverse exchange explicitly, and saves
energy, local magnetization, bond and triangle observables plus convergence.
It supports ordinary3x3 and tied sqrt3xsqrt3 cells through the existing geometry.
The earlier Python measurement bridge has been removed.

For the long D4..12 job, see SU2_D4_D12_SERVER.md and run_su2_D4_D12_server.jl.
CPU dependencies use Project.toml; GPU dependencies use gpu/Project.toml.
The server bootstrap selects and instantiates the appropriate Julia project.
