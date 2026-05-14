include("clean_vmc_peps.jl")
using .CleanVMCPEPS

psi = random_peps(4, 4, 2; seed=11)
E, g = energy_and_grad_exact(psi)
Em, gm = markov_energy_and_grad(psi; nsamples=50_000, therm=5_000, sweeps_between=3, seed=31)

println("Exact E = ", E)
println("Markov E = ", Em)
println("overlap = ", CleanVMCPEPS.grad_overlap(gm, g))
println("relnorm = ", CleanVMCPEPS.grad_norm(CleanVMCPEPS.sub_grads(gm, g)) / CleanVMCPEPS.grad_norm(g))
