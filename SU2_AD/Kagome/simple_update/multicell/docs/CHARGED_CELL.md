# Experimental polarized auxiliary spin on a sqrt3 kagome cell

The construction uses **SU(2)-invariant local tensors with one extra open
spin-j index per nine-physical-spin cell**. One up-simplex, selected by
`anchor=1,2,3`, has shape `(virtual-a,virtual-b,virtual-c,auxiliary)`. Every
other simplex and every B tensor retains its existing convention.

Before fixing the extra index the network is an invariant tensor in the
combined physical and auxiliary spaces. Fixing a highest-weight auxiliary
component produces a covariant, magnetized physical state. This distinction
is standard in the symmetric-tensor-network framework; see
[Singh and Vidal, PRB 86, 195114 (2012)](https://arxiv.org/abs/1208.3919)
and the discussion of non-singlet covariant PEPS in Sec. II A of
[Mambrini, Orus and Poilblanc, PRB 94, 205124 (2016)](https://arxiv.org/abs/1608.06003).
The present placement of a spin per kagome cell is our experimental ansatz.

## Magnetization convention

The Hamiltonian convention remains `H = H_exchange + h*sum(Sz)` and the
normalized magnetization is `m = -2*mean(Sz)`. The added index is nondual
`SU2Space(j=>1)`. TensorKit orders its dense basis as `m_q=j,j-1,...,-j`.
Consequently choosing `component=1` imposes

```text
physical Sz per nine-site cell = -j
normalized physical magnetization = 2j/9
j=1/2 -> m=1/9
j=3/2 -> m=1/3
```

On a finite periodic array this follows from total physical-plus-auxiliary
spin invariance. A nine-site torus with a single spin-j auxiliary leg is a
convenient exact check: after projection its physical total spin is j.
On an infinite lattice, repeating the polarized leg selects the corresponding
magnetization density, subject to the usual thermodynamic boundary limit.
This fixes the average density, not an isolated spin-j object on each cell:
physical cell magnetization can fluctuate through the virtual bonds. The
finite nine-site total-S check must not be interpreted as an infinite-state
local-cell S^2 constraint or a statement about correlation length.

The auxiliary index is a charge label, **not an added physical spin in the
Hamiltonian**. It has no Heisenberg bonds or independent Zeeman term. The
unprojected multiplet must not be traced as if it were a physical on-site
degree of freedom: tracing over it gives an unpolarized mixed state.

## Update

New implementation: `src/bosonic/kagome/simple_update_charged_cell.jl`.
All original source files remain untouched. Ordinary down triangles and the
two ordinary up triangles call the existing kernels. The charged up update
extends the original HOSVD to seven axes:

```text
A(v_a,p_a,v_b,p_b,v_c,p_c,q)
```

Each SVD keeps a full virtual SU(2) multiplet and includes q only in the
complement. The resulting simplex remains `(v_a,v_b,v_c,q)`; q is never
truncated, frozen inside the reduced blocks, or absorbed into a virtual bond.
The original lambda dressing/undressing, truncation helper and Trotter
boundary half-steps are retained. The preceding multicell implementation's
local compatibility adapters support old and new TensorKit APIs.

**Approximation to assess:** this HOSVD uses the norm of the full auxiliary
multiplet. It is not optimized with the environment of the subsequently
polarized physical state. Without truncation, exchange evolution commutes
with the final auxiliary projection. With truncation, the retained subspace
is an SU approximation; a lower polarized-state energy is not guaranteed.
The included tests explicitly check the untruncated physical gate action.

For a fixed projected charge, a uniform magnetic field is a constant energy
shift. The SU gates therefore contain exchange only. The field contribution
is added exactly once in measurement, `h*total_Sz/N`. Scanning h within this
fixed-charge family does not change its imposed magnetization.

## Try a small calculation

Use the environment in this directory or your existing compatible environment:

```sh
julia --project=. tests/test_charged_cell.jl
julia --project=. ../../../src/bosonic/kagome/workflows/run_charged_trial.jl results/charged_sqrt3_D3
```

The trial defaults to j=1/2, Dmax=3, seed=1234, h=0.38, and 10 BLAS threads.
It uses only six full SU steps across three short stages, plus the original
boundary half-steps. It is a workflow and sector check, not converged
production data. To change the dimension or seed:

```sh
KAGOME_CHARGED_DMAX=6 KAGOME_CHARGED_SEED=1234 julia --project=. ../../../src/bosonic/kagome/workflows/run_charged_trial.jl results/charged_sqrt3_D6
```

The trial saves raw SU(2) checkpoints (including auxiliary leg and lambdas),
the explicitly polarized dense tensors, and `report.toml`. The report gives
initial and final finite-torus energies, site magnetizations, total Sz and
S^2, and the exact 126-dimensional finite-sector ground-state energy as a
lower bound. Results are clearly labelled as finite nine-site data.

For a custom charge or schedule:

```julia
include("../../../src/bosonic/kagome/simple_update_charged_cell.jl")
using .KagomeChargedSU2
state = initialize_charged(;auxiliary_spin=1//2,anchor=1)
H = KagomeChargedSU2.BaseSU.heisenberg_triangle()
state = itebd_charged(state,H,0.02,0.01;Dmax=6)
obs = exact_torus_observables(state;component=1,h=0.38)
physical_cells = polarized_dense_cells(state;component=1)
```

`polarized_dense_cells` preserves the original three virtual axes of the
simplex and `(down,up,physical)` order of B. These physical arrays generally
retain only U(1), so the existing singlet SU(2) CTMRG cannot simply consume
them as invariant SU(2) TensorMaps. A polarized dense/U(1) CTMRG or an
environment with explicit auxiliary projectors is the next required step.
No such infinite-lattice measurement is claimed in this prototype.

## What the experiment can establish

The tests can establish that the extra-leg ansatz produces the desired
physical spin sector, survives SU, and exports the correct projected state.
Perfect `m=1/9` here is imposed by construction; it is **not** evidence for
a stable magnetization plateau or its width. Establishing a plateau requires
comparing optimized exchange energies in competing magnetization sectors
and taking their field-dependent lower envelope, with environment and D
convergence. In particular, an isolated nine-site torus cannot provide the
thermodynamic lower edge near zero magnetization.
