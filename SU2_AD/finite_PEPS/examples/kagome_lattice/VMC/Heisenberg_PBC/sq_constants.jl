using Random
using LinearAlgebra
using Distributions

# Constants related to the lattice

const N = 2             # SU(N); number of spin flavors
# const L_N = L ÷ N
const J = 1.0           # Coupling constant


# Constants related to the Monte Carlo
# const twositemove = 0.3
# const threesitemove = 0.7  # Probability of three-site permutation
# const Nchunk = 30000       # Size of a "chunk" of simulation
# const thermsteps = 1000    # Thermalization steps
# const Npars = Nsteps/Nchunk
# const Nscra = Nchunk/4   # Recalculate W matrix frequency
# const bin = Nsteps/1     # Size of each bin
# const Ny = Nsteps/bin    # Number of bins
# const Nthrow = 2 * bin     # Equilibration time
# const Nopt = 100000        # Steps before parameter optimization
# const Threshold = 1e-6     # Minimum threshold for eigenvalues




# Random number generators
seed = rand(UInt32)
#seed = 100
rng = MersenneTwister(seed)  # Seed the generator
# distr = 1:L
# distr2 = 1:fnn
# dis = Uniform(0.0, 1.0)
# distr3 = 1:(L/2)

# Example usage of random distributions
#=
random_site = rand(rng, distr)
random_nn = rand(rng, distr2)
random_float = rand(rng, dis)
random_half_site = rand(rng, distr3)
random_fermion = rand(rng, distr4)

# Display random samples for verification
println("Random site: ", random_site)
println("Random nearest neighbor: ", random_nn)
println("Random float: ", random_float)
println("Random half site: ", random_half_site)
println("Random fermion: ", random_fermion)
=#