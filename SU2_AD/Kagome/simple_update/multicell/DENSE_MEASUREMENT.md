# Pure Julia kagome conversion and measurement

Use task=:convert in run_kagome_settings.jl. All choices are in the Julia
configuration block. The default output representation is :u1; choose :dense
for symmetry-free TensorKit ComplexSpace tensors and ordinary Julia arrays.
Both representations use the same native factorized CTMRG and observables.
No Python executable or external Python package is called.

The core implementation lives in src/bosonic/kagome/native_measurement/ and
reuses the original geometry, CTM helpers/settings and GPU storage helpers.
Output includes per-spin total/exchange/field energy, nine independent site Sz,
18 bonds, six triangle energies and raw <Sz Sz Sz> correlations, actual corner
dimensions, convergence flag/error/iterations and timing. These are saved in
JLD2/TOML/JSON and CSV. The onsite field is counted once.

The auxiliary highest-weight selection is explicit. The U1 representation
keeps a one-dimensional charged spectator; dense exports select the same
component. Ordinary arrays use the original B(down,up,physical), Tu and Td
axes without changing the state. No large-D finite-torus expansion is used.

The independent function export_and_verify in verify_charged_dense.jl is
retained for small finite nine-site checks (actualD<=6); its historical
validation records are in DENSE_VALIDATION.md. Its optional CTMRG entry now
uses the native Julia backend. For current server settings and tests see
SU2_D4_D12_SERVER.md and SU2_D4_D12_SERVER_VALIDATION.md.
