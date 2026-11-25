# Gaussian Fock import conventions

The physical modes used by the source archives are
`b_1=c_up`, `b_2=c_down†`. Both are canonical fermion annihilation operators in
the particle/hole description. Each local Fock basis is
`|0>, b_1† b_2†|0>, b_1†|0>, b_2†|0>`, represented by
`Rep[Z2](0=>2,1=>2)`. The source's full spin and anomalous blocks are retained.

For `z=(b,b†)`, the imported pure projector is
`P_ij=<z_j† z_i>=W W†`. After choosing a nonsingular vacuum chart, write
`W=(U;V)` and `Z=U V⁻¹`. `Z` is antisymmetric. The amplitude of an even occupied
subset S is proportional to `Pf(Z[S,S])`. Undoing the chart's particle/hole
changes requires Jordan-Wigner signs; applying creation/annihilation operators
to explicit bitstrings fixes those signs. The converter verifies the full
Nambu covariance reconstructed from the resulting Fock vector.

The source's `C_in=+swap` bond is the symmetric Slater projection
`(|10>+|01>)/sqrt(2)` for one mode, independently checked by direct Fock-space
projection. Transform exactly one endpoint per virtual mode to holes. The
resulting Bell bond has diagonal Fock coefficients when the two endpoint
occupation bases agree. Its bra coefficients are absorbed into the site
tensors before ordinary TensorKit index summation; dropping these phases is
incorrect for multi-mode legs.

For m=2 (D=4), all simplex virtual modes are particle/hole transformed. For
m=3 (D=8), the first mode of each A-site virtual leg is transformed on the site
instead; the corresponding A mode is omitted from the simplex transformation.
Thus each site and simplex undergoes an even number of transformations and
remains an even Z2 tensor. This uses D=2^m total virtual states, not D per spin.

Native import maps current `(a,b,c)` to source `(B,A,C)` and the three current
x positions to source x indices `(0,2,1)`. Current down simplices connect
`a(x+1,y),b(x,y),c(x,y+1)`. For primitive iPEPS import, the actual source
connectivity fixes current `(L,D,R,U)` to source `(R,D,L,U)`; virtual-leg names
are not interchangeable with Cartesian coordinate signs. After collecting
physical legs and applying fermionic permutations, use the same three legacy
crossings as `build_A_kagome`: `(L,D)`, `(L,R)`, `(D,R)`.

For the selected `orig` Hamiltonian, `theta=-pi/24` and `flux=2pi/3`.
Coefficients multiplying current `(a†b,b†c,c†a)` are `(exp(i theta),... )` on an
up triangle and
`(exp(i theta),exp(i(theta+flux*x)),exp(i(theta-flux*x)))` on a down triangle,
where x is the source cell index. These are **positive** hopping coefficients
from the source `bloch` function. The hole coefficients are their negative
complex conjugates. The local chemical term is
`-mu_up*n_up + mu_down*n_hole - mu_down`.

Measured physical quantities are recovered as

* `n_down = 1 - <b_2† b_2>`;
* physical spin flip `<c_down† c_up> = <b_2 b_1>`;
* physical pair creation `<c_up† c_down†> = <b_1† b_2>`;
* `Ekin = Egrand + mu_up*n_up + mu_down*n_down`, per physical site.

The saved result arrays named `pair` and `mix` refer to the **b-mode** pair
and mixing operators above. Initial exploratory TOML files used the names
`max_pair_error` and `max_spin_mix_error` for these b-mode quantities; newer
reports use explicit physical spin-flip and pair-creation names.

These checkpoints retain the documented particle/hole physical basis. They
should not be measured with the ordinary electron Hubbard operator constructor
without transforming that Hamiltonian. Import defaults to no Gutzwiller projection.

## Optional Gutzwiller projection (Julia)

`import_gaussian_ipess(path; gutzwiller=true)` and
`import_gaussian_ipeps(path; gutzwiller=true)` impose physical-electron single
occupancy on every original site. Since `n_down=1-n_hole`, this means retaining
the **even b-mode** states, not the odd b-mode states:

```
physical down = b vacuum
physical up   = b_up† b_hole† |b vacuum>
G = [0 1 0 0; 1 0 0 0]          # output order (up,down)
G†G = diag(1,1,0,0)
```

The relative phase is fixed by `S+ = b_up† b_hole†`, so `Sz=diag(1/2,-1/2)`
and `S+=[0 1;0 0]` in the output basis. The spin space is `Rep[Z2](0=>2)`
in this particle/hole grading. Virtual spaces and their fermionic crossings
must still be retained; projection does not turn the virtual network into an
ordinary bosonic contraction.

Native iPESS projection acts on the physical leg of each B and returns a
`ProjectedSpinIPESS`. B becomes `(D,D,2)`; Tu/Td and both lambda sets are copied
unchanged. `build_A_kagome` and `convert_iPESS_to_iPEPS` have the same geometry
and virtual crossing signs, with a blocked physical dimension of 8 instead of
64. The separate state type and checkpoint format keep it out of the original
four-state Hubbard state loader. For a blocked cell, projection is
`U_spin*(G⊗G⊗G)*U_original†`, so no assumed fused-basis ordering is used.

Projection is local and exact, without virtual-bond truncation. Input tensors
are not mutated. It generally produces a **non-Gaussian** state: the saved
Gaussian energy reference does not apply to the projected state. The Gaussian
energy evaluators reject the projected physical space. Returned source chemical
fields are metadata, not automatically fields in a spin Hamiltonian.
