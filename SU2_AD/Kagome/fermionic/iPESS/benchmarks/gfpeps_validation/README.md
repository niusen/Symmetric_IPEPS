> 目录整理：功能函数已迁入 `src/fermionic/kagome/gfpeps/`。
> 本目录的 `run.jl` / `run_heisenberg.jl` 等是兼容入口；测试在 `tests/`，
> 历史报告在 `reports/`，Python 辅助入口在 `tools/`，运行日志在 `../../results/gfpeps_validation/logs/`。
> 服务器优先使用 [配置与启动说明](../../server/README.md)。
> Python 导出需先设置 `KAGOME_GAUSSIAN_SOURCE`，已导出的 MAT 测量不需要它。

# Saved spinful Gaussian states → Z2 tensors → CTMRG

This benchmark compares **two contractions of the same saved variational state**.
The reference is its exact Gaussian contraction integrated over the Brillouin
zone, not the exact ground-state energy of the hopping Hamiltonian. No Gaussian
state is reoptimized and no Gutzwiller projection is applied.

Final results and reproducible run choices: [中文验证报告](reports/VALIDATION_REPORT.md).
D=4 energy errors are below 1e-7 for both constructions; D=8 at chi=64 is below
1e-6 for both kinetic and grand energies. All runs, including failed corner
stopping criteria, remain in the generated comparison table.

## Optional spin projection (implemented in Julia)

**Spin-sector correction:** the selected direct iPEPS D=4 archive does not
represent a fixed 1/9 magnetization state. Both selected D=8 archives also mix
total-Sz sectors. Local Gutzwiller projection fixes single occupancy, not total
magnetization. Only the selected native D=4 iPESS passes the numerical sector
checks. See [the independent charge audit](reports/SPIN_SECTOR_AUDIT.md). This does not
change the unprojected same-state Gaussian/CTMRG comparisons above.

An additional native D=8 candidate, `orig_m3_s12103`, passes the numerical spin
sector audit and is kept separately as `ipess_D8_seed12103`. The serial Julia
driver `run_projected_D8.jl` measures both native D=8 candidates and the direct
D=8 iPEPS after projection, with chi=16/32 checks. It preserves completed runs.
Projected Heisenberg energies, magnetizations, and environment comparisons are
collected in [PROJECTED_HEISENBERG_D4_D8.md](reports/PROJECTED_HEISENBERG_D4_D8.md).
Run `summarize_projected_runs.jl` to generate a fresh report in `../../results/gfpeps_validation/reports/` from the saved TOML results.

The validation results above concern the unprojected states. Projection is off
by default and does not change any existing source archive or checkpoint.

```julia
include("Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/run.jl")
root = "Kagome/fermionic/iPESS/results/gfpeps_validation"

# Keep the native B/Tu/Td structure, physical B dimension 2, virtual D unchanged.
state, source_fields = import_gaussian_ipess(
    joinpath(root,"ipess_D4/tensors.mat"); gutzwiller=true)
A = convert_iPESS_to_iPEPS(state)
_, _, U = build_A_kagome(state,1,1) # blocked spin dimension 8
save_projected_checkpoint(joinpath(root,"ipess_D4/gutzwiller_spin.jld2"), state;
    metadata=Dict("source_fields"=>source_fields))
state2, parameters, metadata = load_projected_checkpoint(
    joinpath(root,"ipess_D4/gutzwiller_spin.jld2"))

# Or project the direct Gaussian iPEPS. D8 uses the same keyword/API.
A, U, source_fields = import_gaussian_ipeps(
    joinpath(root,"ipeps_D4/tensors.mat"); gutzwiller=true)

# Physical spin operators and an onsite CTMRG measurement, without assuming a model.
ops = blocked_spin_operators(U) # ops.Sx[i], Sy[i], Sz[i], Sp[i], Sm[i]; i=(a,b,c)
result = measure_projected_spins(A,U; chi=16,maxiter=100,tol=1e-8)
# result.magnetization[x,y,site,component], component=(x,y,z)

# Fixed-state Heisenberg expectation: H=J sum_<ij> S_i dot S_j, h=0.
heis = measure_heisenberg(A,U; J=1,chi=32,maxiter=100,tol=1e-7)
# heis.energy_per_site; heis.magnetization_per_site = [Mx,My,Mz]
# heis.mz_over_saturation = 2Mz; no -1/4 constant is included in H.
```

The input basis is `b=(up,down-hole)`: physical single occupancy corresponds to
the b vacuum and b pair, mapped to `(up,down)` by `[0 1 0 0; 1 0 0 0]`.
The spin physical legs are even in this grading; fermionic virtual swaps remain
essential. A projected state is generally non-Gaussian, so the original exact
Gaussian energy cannot be used as its reference. The Heisenberg evaluator sums
all three up- and three down-triangle bonds per primitive cell, using even-spin
insertions in the shared fermionic double layer. No odd hopping parity strings
or diagonal-hopping minus sign are applied to spin-spin operators. It measures
the fixed projected state; it does not reoptimize it for the spin Hamiltonian.

Run `tests/test_gutzwiller.jl` for projector/spin algebra, saved D=4 and D=8 imports,
independent dense basis selection, projection/blocking commutation, and a
known fully polarized product-state CTMRG check. The checkpoint round-trip
check is in `tests/test_projected_checkpoint.jl` (include after `run.jl`).

`run_heisenberg.jl` first tests every bond on a nonuniform 3x2 spin product state
and an entangled singlet, then measures both saved D=4 projected families at
chi=16 and 32. Results/checkpoints use separate `heisenberg_*` names. For further
convergence runs, include it and call `run_spin_energy(family,4,chi;...)`.

The completed D=4 measurements, convergence limitations and all nine local spin
vectors are documented in [HEISENBERG_D4.md](reports/HEISENBERG_D4.md). Native iPESS gives
E/site approximately -0.3600329 and Mz approximately 0.0555556. The direct iPEPS
estimate is approximately -0.3860 and 0.06246; its corner-spectrum criterion
has not converged, so it must not be reported with the same precision.

## Sources

Read-only source workspace:
`D:/My Documents/My Notes/discussion with AI/magnetic plateau/kagome`.
The `orig` states are selected by `parton_summary/data/es_manifest.json`:

| Family | Original virtual leg D | Archive |
|---|---:|---|
| Gaussian iPESS | 4 | `gaussian_ipess/orig_m2_s12102.npz` |
| Gaussian iPESS | 8 | `gaussian_ipess/orig_m3_s12003.npz` |
| Gaussian iPEPS | 4 | `gfpeps_finite_D_flux/orig_m2_r0.npz` |
| Gaussian iPEPS | 8 | `gfpeps_finite_D_flux/orig_m3_r0.npz` |

The preparation script verifies each SHA256 against the manifest. D=2^m is
the total Fock dimension of one original virtual leg, including both spin
channels. It is not a per-spin dimension and is not the dimension of a blocked
magnetic-cell leg. The older nine-site blocked files are not substituted.

## Conventions and independent checks

The original physical canonical modes are `b=(up,down-hole)`. We keep this basis
in the tensors. Each two-mode physical space is ordered
`(vac,up+hole,up,hole)` with even parity first. Full spin mixing and anomalous
correlations are retained. These checkpoints must be measured with the
transformed Hamiltonian, not passed to an ordinary electron Hubbard evaluator.
In particular `n_down=1-n_hole`, down-hole hopping is minus the conjugate of
down-electron hopping, and the chemical-potential term includes `-mu_down` per
physical site. Both kinetic energy and `H_kin-mu_up*N_up-mu_down*N_down` are saved.

`src/fermionic/kagome/gfpeps/python/gaussian_fock.py` converts a full pure Nambu projector to Pfaffian Fock
amplitudes using a nonsingular particle-hole vacuum chart. Explicit Fock-space
operators reconstruct the **entire** Nambu covariance as a local check. A
separate random small-state test fixes the virtual bond convention: Gaussian
`C_in=+swap` projects onto `(10+01)/sqrt(2)`.

One endpoint of each virtual mode is particle-hole transformed so this Slater
bond becomes a diagonal Bell bond; its bra phases are absorbed explicitly.
For odd m the endpoint choice preserves even parity of every site and simplex.
Native iPESS import reverses both spatial axes and maps current `(a,b,c)` to
source `(B,A,C)`. A four-tensor triangle is independently contracted by Gaussian
algebra and compared with the native TensorKit triangle. This tests all local
swaps and bond phases before using CTMRG.

For the primitive iPEPS, current `(L,D,R,U)` maps to source `(R,D,L,U)`.
This is derived from `primitive_fit.bonds` and the native iPESS connectivity;
the D/U names must not be treated as Cartesian +/-y labels. The first draft
mistakenly exchanged these legs. Its failed measurement is retained only as
`debug_wrong_y_legs.*`, not as a valid CTMRG result. The independent five-tensor
open-block check agrees with Gaussian contraction without any fitted signs.

The source magnetic cell contains nine physical sites (three kagome primitive
cells). The `orig` hopping model has triangle flux `-pi/8`, primitive-cell flux
`2pi/3`, and source chemical fields retained in `tensors.mat`. Energy is divided
by nine physical sites. Gaussian references use shifted 48², 96², and 192² grids;
`refine_reference.py` additionally checks 384² and 768² meshes using the frozen
source archives. All four energies change by only a few 1e-15 on refinement.

The existing kagome blocking and shared CTMRG environment updates and observable
contractions are reused. A generic tuple-cell rotation method in **the kagome
adapter** supports the 3×1 cell omitted by the shared enumerated method.
`tests/test_rotation.jl` compares it with all tested old cell shapes.

For D=8 and blocked physical dimension 64, the shared double-layer construction
forms an unnecessarily large physical intermediate (observed process peak near
20 GB). The optional `compact_double_layer.jl` first contracts physical legs and
then fuses the four bra/ket pairs directly. It preserves every bra swap, fusion
map and final parity crossing. Nineteen tests against the original routines,
including odd extra legs of dimensions 1 and 2, passed with maximum relative
tensor error 2.5e-16. Shared/triangular source files are untouched. The benchmark
driver enables this equivalent contraction order by default; set
`KAGOME_COMPACT_DOUBLE_LAYER=0` to use the original implementation. Result names
ending in `_compact` identify the equivalent low-memory implementation.

The optional `gauge_sweeps` argument balances virtual-bond Gramians using
parity-even invertible gauges. It inserts `G` and `inv(G)` on opposite endpoints,
without discarding any singular values or changing the infinite state. A small
Gramian regularizer chooses the gauge only. `tests/test_gauge.jl` checks every physical
amplitude of a closed three-site network, with ordinary and parity-twisted
boundaries; agreement is better than 7e-16. Results record gauge identity
residuals and condition numbers. Warm starts must use the same gauge choice.
The `update_method` argument can select the existing `continuous_update` or
`together_update` routine. Both are unchanged shared CTMRG implementations.
The projector diagnostic in `compact_double_layer.jl` delegates to the original inverse-square-root method
and records retained ranks without altering its values. Twelve comparisons
passed exactly. In both final D=8 chi=64 runs every projector retained rank 64.

## Reproduce from SU2_AD

```powershell
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/test_conversion.py
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/prepare.py --m 2
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/prepare.py --m 3
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/prepare.py --family ipeps --m 2
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/prepare.py --family ipeps --m 3
julia --startup-file=no --heap-size-hint=6000M Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/run.jl 4 16 1e-10 100
julia --startup-file=no --heap-size-hint=6000M Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/run.jl 8 16 1e-9 60
julia --startup-file=no --heap-size-hint=6000M Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/run.jl 4 16 1e-9 60 ipeps
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/check_patch.py 2
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/check_patch.py 3
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/check_block.py
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/check_block.py 3
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/check_archives.py
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/check_reference.py
python Kagome/fermionic/iPESS/benchmarks/gfpeps_validation/tools/summarize.py
```

Outputs are under `Kagome/fermionic/iPESS/results/gfpeps_validation/`.
The command-line runner uses four BLAS threads; set `KAGOME_BLAS_THREADS=1`
to reproduce the initial single-thread runs. A persistent Julia session can
include `session_runs.jl` and call `validate_saved`; its `warmstart` keyword
loads the saved environment from a previous result of the same state.
With `save_environment=true`, it also writes an environment checkpoint before
starting the more expensive observable measurements.
Every run measures the full observable set. `tests/test_measurement.jl` compares the
original and compact contraction orders in exactly the same saved CTM environment.
Each family/D directory contains source metadata, dense Fock tensors, a Julia
checkpoint, Gaussian references, and CTMRG comparison results. `converged`
refers specifically to the requested corner-spectrum stopping tolerance; it
must not be inferred from the energy error or vice versa.

Passing these tests validates the conversion, blocking, contraction and tested
observables for these states. It does not by itself validate SU/FU/AD updates,
their gradients, or convergence to the variational optimum.
