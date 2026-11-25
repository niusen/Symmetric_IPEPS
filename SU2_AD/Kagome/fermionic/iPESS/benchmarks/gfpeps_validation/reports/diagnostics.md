# Diagnostic runs and interpretation

The following exploratory runs were deliberately stopped and have no final
energy result. They must not be counted as converged validations.

* Native iPESS D=4, chi=48, cutoff=1e-12: stopped after 31 iterations because
  the corner error remained near 1e-6. The preceding chi=16/24/32 runs already
  established energy convergence; subsequent cutoff checks were more useful.
* Primitive iPEPS D=4, chi=32, first importer draft: stopped after identifying
  the incorrectly exchanged D/U virtual legs. The failed chi=16 result is
  retained as `ipeps_D4/debug_wrong_y_legs.*`. The source connectivity and an
  independent open-block Gaussian contraction determined the corrected map.
* Native iPESS D=8, chi=16, cutoff=1e-9: stopped after 15 iterations; the last
  corner error was 1.0178679044236187, with persistent oscillation. Full log:
  `ipess_D8_chi16.log`. Restarted at cutoff=1e-8, retaining environments in the
  new driver for controlled follow-up runs.

A completed primitive iPEPS D=4 chi=48/cutoff=1e-11 run also failed to converge
after 30 iterations (corner error about 1.1); its energy and correlation errors
are included in the full results table. Increasing chi while keeping poorly
conditioned projector modes is not an accuracy guarantee.

The first detailed primitive iPEPS D=8 measurement with the original double-layer
routine reached corner convergence at chi=16/cutoff=1e-8 (11 iterations, error
6.06e-8), but its observable stage was stopped before a final energy was saved.
Its large physical intermediate caused a process peak near 20 GB and prolonged
measurement time. The second native iPESS D=8 attempt was first suspended to
avoid concurrent memory pressure, then stopped before the CTM loop. Both were
replaced by the independently tested equivalent compact double-layer order.
Neither stopped run is counted as a completed energy comparison.

The generic kagome cell-rotation adapter passed seven comparisons: six
previously supported cell shapes exactly reproduced the shared enumerated
routine, and the new 3x1 shape matched the four per-tensor rotations. The
shared implementation is unchanged.

## D=4 conditioning and convergence checks

Primitive iPEPS at chi=48 and cutoff=1e-10 develops a repeating corner-spectrum
cycle with both shared update orders. This is retained as a failed stopping
criterion, even where the energy agrees well. An invertible virtual gauge
balance does not change the state and was independently checked on closed
ordinary/parity-twisted networks (amplitude residual below 7e-16).

With eight gauge sweeps and cutoff=3e-10, chi=48 converges in 10 iterations to
corner error 2.98e-8. Its kinetic and grand energy errors are respectively
-9.81e-9 and +6.56e-8. A chi=64 warm-start check changes those energies by only
8.70e-12 and 1.68e-11. That stricter run requested corner tolerance 1e-9 and
does not pass it: the corner difference reaches a numerical floor near 3e-8.
These distinct tolerance outcomes are recorded without relabeling runs.

The same cutoff without gauge balancing gives kinetic/grand errors +8.14e-8
and +7.11e-8. Thus the consistency check is not dependent on a single virtual
gauge or an accidental cancellation with the chemical-potential term.
