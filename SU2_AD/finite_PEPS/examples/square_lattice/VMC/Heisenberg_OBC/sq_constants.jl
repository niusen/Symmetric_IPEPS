using Random
using LinearAlgebra
using Distributions

# Constants related to the lattice
# const fnn = 4           # number of first nearest neighbors
# const snn = 4           # number of second nearest neighbors
const Ly = 6      # number of sites along y / number of rows in the lattice
const Lx = 6      # number of sites along x / number of columns in the lattice
const L = Lx * Ly # total number of lattice sites
const N = 2             # SU(N); number of spin flavors
const L_N = L ÷ N
const Ne = L            # Number of electrons on the lattice (for spin models this will always be equal to L)
const J = 1.0           # Coupling constant
const t1 = 1.00         # Nearest neighbor hopping
const DELTA = 0.0500000 # Constant related to changing paramters in stochastic reconfiguration. 
const hmag = 0.569049   # Staggered magnetic field in ansatz
const a = 1             # lattice constant, never used in simulations
const nsdp = 1          # number of slater determinant parameters we want to optimize for, in the stochastic reconfiguration.

# Constants related to the Monte Carlo
const twositemove = 0.3
const threesitemove = 0.7  # Probability of three-site permutation
const Nbra = L             # Inner loop size, to generate uncorrelated samples, usually must be of size O(L).
const Nsteps = 1000000       # Total Monte Carlo steps
const Nchunk = 30000       # Size of a "chunk" of simulation
const thermsteps = 1000    # Thermalization steps
const binn = 1000          # Bin size to store the data during the monte carlo run. 
const Npars = Nsteps/Nchunk
const Nscra = Nchunk/4   # Recalculate W matrix frequency
const bin = Nsteps/1     # Size of each bin
const Ny = Nsteps/bin    # Number of bins
const Nthrow = 2 * bin     # Equilibration time
const Nopt = 100000        # Steps before parameter optimization
const Threshold = 1e-6     # Minimum threshold for eigenvalues




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