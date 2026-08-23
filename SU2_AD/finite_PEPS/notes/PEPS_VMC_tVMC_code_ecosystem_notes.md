# PEPS + VMC / tVMC Code Ecosystem Notes

**Last checked:** 2026-08-23  
**Purpose:** working note for a Codex project on finite PEPS + variational Monte Carlo (VMC), stochastic reconfiguration (SR), time-dependent VMC (tVMC/TDVP), PEPS gauge removal, sequential/direct sampling, and fermionic PEPS.

---

## 0. Executive summary

The public code ecosystem currently splits into four layers:

1. **Closest public implementation of the Wu–Nys PEPS-tVMC ideas**
   - `fliingelephant/VMC`
   - JAX + NetKet interfaces
   - finite PEPS, sequential sampling, QGT/SR, gauge removal, TDVP, time-dependent examples
   - **not** the official production code of Yantao Wu / Jannes Nys
   - explicitly marked “under active development and not ready for use”

2. **More mature finite-PEPS VMC implementation**
   - `QuantumLiquids/PEPS`
   - C++20 + MPI + TensorToolkit
   - ground-state finite PEPS VMC, SR, fermions, measurements
   - no general real-time PEPS-tVMC driver advertised in the README
   - author Hao-Xin Wang; README acknowledges Wen-Yuan Liu and Zheng-Cheng Gu

3. **Paper-specific / pedagogical PEPS sampling implementation**
   - `tvieijra/DirectSamplingPEPS.jl`
   - Julia
   - official code cited directly by the 2021 PRB direct-sampling paper

4. **General VMC / tVMC infrastructure, not PEPS-specific**
   - `netket/netket`
   - `netket/netket_fidelity`
   - `markusschmitt/vmc_jax`
   - useful for QGT, SR, TDVP/tVMC, sampling, JAX design patterns, but they do not by themselves provide the PEPS contraction engine used in Wu–Nys.

A second important category is **official data repositories that are not production code**:
- `yantaow/open_data`
- `LiuWenyuan-Phy/2d-hubbard-data-fPEPS-PRL2025`

As of 2026-08-23, I have **not found a public repository for the actual production JAX PEPS-tVMC code used in Wu–Nys, PRX Quantum 7, 033035 (2026)**. The paper states that production calculations were implemented in JAX and parts were tested using TeNPy, but its public GitHub reference points to data/benchmark material rather than the full PEPS-tVMC implementation.

---

# 1. Core repository: fliingelephant/VMC

## Repository

- **Repo:** https://github.com/fliingelephant/VMC
- **Owner account:** `fliingelephant`
- **Public identity associated with this account:** Huanhai Zhou, HKUST (Guangzhou), based on public GitHub activity/issues.
- **Language / stack:** Python, JAX, NetKet interfaces
- **License:** MIT
- **Status:** README explicitly says:
  > “This repo is under active development and not ready for use.”

## Why this repo matters

This is currently the **closest public repository I found to the PEPS-tVMC method in Wu–Nys**.

The README describes:

- sequential sampling
- efficient energy and gradient evaluation with environment reuse
- gauge-invariant PEPS for finite Abelian lattice gauge theories \(Z_N\)
- sliced gradients for memory-efficient SR
- gauge removal for improved numerical stability

The repository explicitly cites the following methodological chain:

1. Wen-Yuan Liu, Yi-Zhen Huang, Shou-Shu Gong, Zheng-Cheng Gu,  
   **“Accurate Simulation for Finite Projected Entangled Pair States in Two Dimensions”**  
   Phys. Rev. B **103**, 235155 (2021)  
   DOI: https://doi.org/10.1103/PhysRevB.103.235155  
   arXiv: https://arxiv.org/abs/1908.09359

2. Yantao Wu, Wen-Yuan Liu,  
   **“Accurate Gauge-Invariant Tensor-Network Simulations for Abelian Lattice Gauge Theory in (2+1)D: Ground-State and Real-Time Dynamics”**  
   Phys. Rev. Lett. **135**, 130401 (2025)  
   DOI: https://doi.org/10.1103/3m3j-ds18  
   arXiv: https://arxiv.org/abs/2503.20566

3. Yantao Wu, Jannes Nys,  
   **“Real-Time Dynamics in Two Dimensions with Tensor Network States via Time-Dependent Variational Monte Carlo Method”**  
   PRX Quantum **7**, 033035 (2026)  
   DOI: https://doi.org/10.1103/tggc-8fjx  
   arXiv: https://arxiv.org/abs/2512.06768

## Repository structure worth reading first

Current top-level source structure:

```text
src/vmc/
    core/
    drivers/
    gauge/
    operators/
    peps/
    preconditioners/
    qgt/
    utils/
    config.py
    workflow.py
```

The most relevant directories for the Wu–Nys method are:

```text
src/vmc/drivers/
    integrators.py
    tdvp.py

src/vmc/qgt/
    jacobian.py
    netket_compat.py
    qgt.py
    solvers.py

src/vmc/gauge/
src/vmc/peps/
```

Examples currently include:

```text
examples/
    Schmitt_2022_TFIM2d/
    gauge_removal/
    ground_states/
    lgt/
    time_dependent/
```

### Time-dependent benchmark

`examples/time_dependent/README.md` describes a **3x3 exact benchmark**:

- exact Schrödinger evolution in the full \(2^9=512\) Hilbert space
- PEPS/VMC trajectory from `TDVPDriver`
- reports:
  - `mz_exact`
  - `mz_vmc`
  - `diff_mz`
  - state fidelity \(|\langle\psi_{\rm exact}|\psi_{\rm vmc}\rangle|^2\)

This is particularly useful because the Wu–Nys Chern-insulator benchmark mainly emphasizes local densities, whereas this small-system example explicitly checks global fidelity.

## Relation to Wu–Nys

**Important:** this is **not confirmed as the official Wu–Nys production code**.

The safe description is:

> an independent public JAX/NetKet PEPS-VMC/tVMC implementation that explicitly follows and cites the Liu–Wu–Nys methodological line.

For a Codex project, this should probably be the **first repo to inspect for TDVP/gauge-removal architecture**.

## Suggested first files for Codex

```text
README.md
examples/time_dependent/README.md
src/vmc/drivers/tdvp.py
src/vmc/drivers/integrators.py
src/vmc/qgt/jacobian.py
src/vmc/qgt/qgt.py
src/vmc/qgt/solvers.py
src/vmc/gauge/
src/vmc/peps/
```

---

# 2. Core repository: QuantumLiquids/PEPS

## Repository

- **Repo:** https://github.com/QuantumLiquids/PEPS
- **Author listed in README:** Hao-Xin Wang
- **Language:** C++20
- **Parallelization:** MPI
- **Tensor backend:** QuantumLiquids/TensorToolkit
- **License:** GPL-3.0
- **Status:** active development

## Advertised features

The README explicitly lists:

- finite-size PEPS
- Variational Monte Carlo optimizer
- stochastic reconfiguration (SR)
- first-order optimizers such as AdaGrad / Adam
- fermion support
- extensible model and Monte Carlo updater interfaces
- Monte Carlo measurement tools
- one- and two-point functions
- autocorrelation measurements
- common lattice models

This makes it the most obviously **mature finite-PEPS ground-state VMC codebase** among the public repositories surveyed here.

## Author / intellectual lineage

README author:

- **Hao-Xin Wang**

README acknowledgments:

- **Wen-Yuan Liu** — explicitly described as an expert in variational-Monte-Carlo PEPS
- **Zheng-Cheng Gu** — postdoc advisor

Therefore this code is closely related to the finite-PEPS VMC ecosystem around Liu/Gu, but the README does **not** claim to be the official code of one specific paper.

## Relevant background paper

The most important background paper to read alongside this repo is:

Wen-Yuan Liu, Yi-Zhen Huang, Shou-Shu Gong, Zheng-Cheng Gu,  
**“Accurate Simulation for Finite Projected Entangled Pair States in Two Dimensions”**  
Phys. Rev. B **103**, 235155 (2021)  
DOI: https://doi.org/10.1103/PhysRevB.103.235155  
arXiv: https://arxiv.org/abs/1908.09359

Main methodological relevance:

- finite PEPS + VMC
- large 2D lattices
- Monte Carlo evaluation instead of full double-layer expectation-value contraction
- basis for later sequential-sampling / PEPS-VMC developments

## Dependencies

### TensorToolkit

- Repo: https://github.com/QuantumLiquids/TensorToolkit
- Author listed in README: Hao-Xin Wang
- C++ tensor library
- MPI
- Abelian quantum numbers
- Grassmann tensor-network support
- CUDA single-card support
- contractions / decompositions / tensor operations

`QuantumLiquids/PEPS` explicitly depends on TensorToolkit.

## Why this repo is useful for a Codex project

Use it to understand a **production-style finite PEPS-VMC implementation**:

- PEPS data structures
- Monte Carlo updater architecture
- energy and observable estimators
- SR optimization
- fermionic support
- MPI parallelism

It is probably a better ground-state PEPS-VMC reference than `fliingelephant/VMC`, while the latter is more directly relevant to the new TDVP/gauge-removal dynamics.

---

# 3. Official paper code: DirectSamplingPEPS.jl

## Repository

- **Repo:** https://github.com/tvieijra/DirectSamplingPEPS.jl
- **Owner:** Tom Vieijra
- **Language:** Julia
- **Scale:** small / paper-specific codebase
- **Status:** old but useful as a readable methodological reference

Current repository structure is compact:

```text
Examples/
src/
Project.toml
Manifest.toml
README.md
```

## Associated paper

Tom Vieijra, Jutho Haegeman, Frank Verstraete, Laurens Vanderstraeten,  
**“Direct sampling of projected entangled-pair states”**  
Phys. Rev. B **104**, 235141 (2021)  
DOI: https://doi.org/10.1103/PhysRevB.104.235141  
arXiv: https://arxiv.org/abs/2109.07356

The published paper explicitly cites the GitHub repo as a reference, so this is an **official paper-code link**, not merely a related implementation.

## Main methodological content

The paper develops direct / independent sampling of PEPS using an auxiliary probability distribution and importance sampling, avoiding the long autocorrelation times of local Metropolis chains.

This code is useful for isolating the **sampling problem** from the much larger Wu–Nys tVMC stack.

## Best use

If the JAX code is too engineering-heavy, read this paper/repo pair first to understand:

- sequential/direct sampling logic
- boundary contraction during sampling
- how an approximate PEPS contraction becomes a sampling distribution
- importance reweighting

---

# 4. Independent VMC/TNS ecosystem: sjdu10/vmc_torch

## Repository

- **Repo:** https://github.com/sjdu10/vmc_torch
- **Primary author / software citation:** Si-Jing Du
- **Language:** Python / PyTorch
- **Parallelization:** MPI via `mpi4py`
- **Tensor-network dependencies:** `quimb`, `symmray`
- **License:** Apache-2.0

## Advertised ansätze

The README states support for:

- Neural Quantum States
- MPS
- PEPS
- tensor-network states with general geometries
- bosonic and fermionic TNS
- neuralized fermionic TNS
- tensor-network functions

Ground-state VMC includes automatic differentiation and SR.

At present the README is primarily oriented toward **ground-state VMC**, not the Wu–Nys real-time PEPS-tVMC algorithm.

## Associated publications listed by the repo

### Paper 1

Si-Jing Du, Ao Chen, Garnet Kin-Lic Chan,  
**“Neuralized fermionic tensor networks for quantum many-body systems”**  
Phys. Rev. B **113**, 085134 (2026)  
DOI: https://doi.org/10.1103/x8vl-qf14  
arXiv: https://arxiv.org/abs/2506.08329

### Paper 2

Wen-Yuan Liu, Si-Jing Du, Ruojing Peng, Johnnie Gray, Garnet Kin-Lic Chan,  
**“Tensor Network Computations That Capture Strict Variationality, Volume Law Behavior, and the Efficient Representation of Neural Network States”**  
Phys. Rev. Lett. **133**, 260404 (2024)  
arXiv: https://arxiv.org/abs/2405.03797

## Important dependencies

### quimb

- Repo: https://github.com/jcmgray/quimb
- Maintainer/author: Johnnie Gray and contributors
- general tensor-network framework
- supports arbitrary tensor-network geometry
- PEPS, MPS, MERA, circuits
- can use JAX / PyTorch backends
- fermionic and symmetric tensors can be handled through `symmray`

### symmray

- Repo: https://github.com/jcmgray/symmray
- Maintainer/author: Johnnie Gray
- block-sparse symmetric and fermionic tensor library

## Why this repo matters

This repo shows that PEPS/TNS + VMC is **not restricted to the Liu/Wu group**.

It provides a second, independently developed architecture:

```text
PyTorch
  + quimb
  + symmray
  + MPI
  + SR
  + fermionic TNS
```

This is valuable for checking whether a design idea in the Wu/Nys-style code is truly PEPS-specific or merely one implementation choice.

---

# 5. Yantao Wu official data repository: yantaow/open_data

## Repository

- **Repo:** https://github.com/yantaow/open_data
- **Owner:** Yantao Wu
- **Purpose:** open data / benchmark material for several papers
- **Important:** this is **not** the full production PEPS-VMC/tVMC source code.

Relevant subdirectories currently include:

```text
wu2025accurate/
wu2025algorithm/
wu2025real-time/
```

## 5.1 wu2025accurate

Direct link:

https://github.com/yantaow/open_data/tree/main/wu2025accurate

Associated paper:

Yantao Wu, Wen-Yuan Liu,  
**“Accurate Gauge-Invariant Tensor-Network Simulations for Abelian Lattice Gauge Theory in (2+1)D: Ground-State and Real-Time Dynamics”**  
Phys. Rev. Lett. **135**, 130401 (2025)  
DOI: https://doi.org/10.1103/3m3j-ds18  
arXiv: https://arxiv.org/abs/2503.20566

The public directory contains figure/data material such as:

```text
Fig2-4/
Fig5/
Fig6/
info
```

It should be treated as a **data repository**, not a reusable implementation of gauge-invariant PEPS-VMC.

## 5.2 wu2025algorithm

Direct link:

https://github.com/yantaow/open_data/tree/main/wu2025algorithm

Associated paper:

Yantao Wu, Zhehao Dai,  
**“Algorithms for variational Monte Carlo calculations of fermion projected entangled pair states in the swap gates formulation and the detailed balance of tensor network sequential sampling”**  
Chinese Physics B **35**, 020502 (2026)  
DOI: https://doi.org/10.1088/1674-1056/ae2673  
arXiv: https://arxiv.org/abs/2506.20106

The public directory currently contains:

```text
4x4_hubbard_fig/
info
```

Again, this is data / paper-support material, not a full fPEPS-VMC implementation.

### Why this paper is important

For a fermionic implementation it is one of the most directly relevant algorithm papers because it explains:

- fermionic PEPS VMC
- swap-gate formulation
- sequential tensor-network sampling
- detailed balance of the sampling method

This should be part of the mandatory reading set if the Codex project aims at fermionic PEPS.

## 5.3 wu2025real-time

Direct link:

https://github.com/yantaow/open_data/tree/main/wu2025real-time

Associated paper:

Yantao Wu, Jannes Nys,  
**“Real-Time Dynamics in Two Dimensions with Tensor Network States via Time-Dependent Variational Monte Carlo Method”**  
PRX Quantum **7**, 033035 (2026)  
DOI: https://doi.org/10.1103/tggc-8fjx  
arXiv: https://arxiv.org/abs/2512.06768

The public directory currently contains:

```text
cholesky_svd_compare/
info
```

The main paper states that the production code is implemented in **JAX**, with part of the algorithm tested using **TeNPy**, but the full production code is not exposed by this repository.

### Consequence for a Codex project

Do **not** assume that `yantaow/open_data` contains the implementation needed to reproduce Fig. 2 or the other large-scale PEPS-tVMC calculations.

Use it for:
- solver benchmark data
- paper cross-checks
- parameter information if supplied

Use `fliingelephant/VMC` as the closest public implementation reference currently found.

---

# 6. Official data-only repo: finite fPEPS Hubbard PRL

## Repository

- **Repo:** https://github.com/LiuWenyuan-Phy/2d-hubbard-data-fPEPS-PRL2025
- **Owner:** Wen-Yuan Liu
- **Type:** data repository, not production code

## Associated paper

Wen-Yuan Liu, Huanchen Zhai, Ruojing Peng, Zheng-Cheng Gu, Garnet Kin-Lic Chan,  
**“Accurate Simulation of the Hubbard Model with Finite Fermionic Projected Entangled Pair States”**  
arXiv: https://arxiv.org/abs/2502.13454

The repo README says it contains data supporting figures in the paper.

## Relevance

Useful as evidence that finite fPEPS calculations in this ecosystem are mature enough for Hubbard benchmarks, but it does not supply the underlying fPEPS-VMC engine.

---

# 7. General VMC framework: NetKet

## Repository

- **Repo:** https://github.com/netket/netket
- **Website:** https://www.netket.org
- **Language:** Python / JAX
- **License:** Apache-2.0

## Key relation to the Wu–Nys paper

Jannes Nys is one of the authors of the NetKet 3 paper.

NetKet provides mature infrastructure for:

- VMC
- samplers
- automatic differentiation
- QGT
- stochastic reconfiguration
- iterative / direct linear solvers
- quantum dynamics drivers
- JAX/GPU/TPU execution

It is **not a PEPS contraction engine by itself**.

## Associated code paper

Filippo Vicentini, Damian Hofmann, Attila Szabó, Dian Wu, Christopher Roth, Clemens Giuliani, Gabriel Pescia, Jannes Nys, Vladimir Vargas-Calderón, Nikita Astrakhantsev, Giuseppe Carleo,  
**“NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems”**  
SciPost Physics Codebases **7** (2022)  
DOI: https://doi.org/10.21468/SciPostPhysCodeb.7  
arXiv: https://arxiv.org/abs/2112.10526

## Why it matters for implementation

If building a new PEPS-tVMC code in JAX, one natural architecture is:

```text
custom PEPS amplitude/contraction/sampling layer
        +
NetKet-like VMC/QGT/SR/driver layer
```

This is essentially the architecture suggested by `fliingelephant/VMC`.

---

# 8. General projected/tVMC framework: netket_fidelity

## Repository

- **Repo:** https://github.com/netket/netket_fidelity
- **Authors / software citation:** Alessandro Sinibaldi, Filippo Vicentini
- **Framework:** NetKet / JAX
- **Method:** projected time-dependent VMC based on infidelity minimization

## Associated paper

Alessandro Sinibaldi, Clemens Giuliani, Giuseppe Carleo, Filippo Vicentini,  
**“Unbiasing time-dependent Variational Monte Carlo by projected quantum evolution”**  
Quantum **7**, 1131 (2023)  
DOI: https://doi.org/10.22331/q-2023-10-10-1131  
arXiv: https://arxiv.org/abs/2305.14294

## Why it is relevant

This is not PEPS-specific, but it is important for understanding an alternative to differential TDVP/tVMC.

The paper discusses:
- systematic bias / sample complexity issues in standard tVMC when the wavefunction has zeros or near-zeros
- projected evolution formulated as an optimization problem at each time step

For a fermionic PEPS project, this is conceptually important because zeros/nodes can be common.

---

# 9. General VMC/tVMC framework: jVMC

## Repository

- **Repo:** https://github.com/markusschmitt/vmc_jax
- **Package name:** jVMC
- **Authors:** Markus Schmitt, Moritz Reh
- **Language:** Python / JAX
- **License:** MIT

## Associated code paper

Markus Schmitt, Moritz Reh,  
**“jVMC: Versatile and performant variational Monte Carlo leveraging automated differentiation and GPU acceleration”**  
SciPost Physics Codebases **2** (2022)  
arXiv: https://arxiv.org/abs/2108.03409

## Relevance

Useful reference for:

- JAX VMC architecture
- automatic differentiation
- batching
- GPU acceleration
- distributed sampling
- quantum dynamics

Not PEPS-specific.

The `fliingelephant/VMC` repository also contains an example directory named:

```text
examples/Schmitt_2022_TFIM2d/
```

which makes jVMC-related algorithms particularly worth reading when tracing the broader tVMC lineage.

---

# 10. Row-update PEPS-VMC paper: code status

## Paper

Tao Chen, Jing Liu, Yantao Wu, Pan Zhang, Youjin Deng,  
**“Variational Monte Carlo with row-update projected entangled-pair states and its applications to quantum spin glasses”**  
accepted in Phys. Rev. B, 13 August 2026  
DOI: https://doi.org/10.1103/9z82-vlky  
arXiv: https://arxiv.org/abs/2601.20608

## Main method

- autoregressive row-wise sampling
- rejection-free row updates
- single-layer PEPS contractions
- reduced temporal correlations compared with local Metropolis updates
- hybrid row-update + local Metropolis strategy for rugged spin-glass landscapes

## Public code status

As of 2026-08-23, I did **not** find an explicit public GitHub repository linked to this paper.

Therefore record it as:

```text
paper available
public production repo not identified
```

This should be rechecked later.

---

# 11. Methodological paper map

The following reading order gives a relatively coherent historical path.

## A. finite PEPS + VMC foundation

W.-Y. Liu, Y.-Z. Huang, S.-S. Gong, Z.-C. Gu  
**Accurate simulation for finite projected entangled pair states in two dimensions**  
PRB 103, 235155 (2021)  
https://doi.org/10.1103/PhysRevB.103.235155  
https://arxiv.org/abs/1908.09359

Key ideas:
- finite PEPS
- VMC expectation values
- large lattices
- ground-state optimization

Related public code:
- no official production repo identified
- `QuantumLiquids/PEPS` is a strongly related later codebase

## B. direct sampling

T. Vieijra, J. Haegeman, F. Verstraete, L. Vanderstraeten  
**Direct sampling of projected entangled-pair states**  
PRB 104, 235141 (2021)  
https://doi.org/10.1103/PhysRevB.104.235141  
https://arxiv.org/abs/2109.07356

Official code:
- https://github.com/tvieijra/DirectSamplingPEPS.jl

## C. gauge-invariant PEPS + first PEPS-tVMC demonstration

Y. Wu, W.-Y. Liu  
**Accurate Gauge-Invariant Tensor-Network Simulations for Abelian Lattice Gauge Theory in (2+1)D: Ground-State and Real-Time Dynamics**  
PRL 135, 130401 (2025)  
https://doi.org/10.1103/3m3j-ds18  
https://arxiv.org/abs/2503.20566

Key ideas:
- gauge canonical form for gauge-invariant tensor networks
- VMC for 2+1D Abelian lattice gauge theory
- first PEPS + tVMC real-time demonstration

Official public material:
- https://github.com/yantaow/open_data/tree/main/wu2025accurate
- data/support material only

## D. fermionic PEPS-VMC implementation details

Y. Wu, Z. Dai  
**Algorithms for variational Monte Carlo calculations of fermion projected entangled pair states in the swap gates formulation and the detailed balance of tensor network sequential sampling**  
Chinese Physics B 35, 020502 (2026)  
https://doi.org/10.1088/1674-1056/ae2673  
https://arxiv.org/abs/2506.20106

Key ideas:
- swap-gate fPEPS VMC
- sequential sampling
- proof of detailed balance

Official public material:
- https://github.com/yantaow/open_data/tree/main/wu2025algorithm
- data/support material only

## E. generic stable PEPS-tVMC

Y. Wu, J. Nys  
**Real-Time Dynamics in Two Dimensions with Tensor Network States via Time-Dependent Variational Monte Carlo Method**  
PRX Quantum 7, 033035 (2026)  
https://doi.org/10.1103/tggc-8fjx  
https://arxiv.org/abs/2512.06768

Key ideas:
- analytical PEPS gauge-null-space treatment
- QR projection for explicit gauge removal
- minSR interpretation
- new minSR formulation
- small-\(o\) memory trick
- stable Cholesky solution
- generic boson / fermion / spin / gauge-theory applications

Official public material:
- https://github.com/yantaow/open_data/tree/main/wu2025real-time
- **not** full code

Closest public implementation found:
- https://github.com/fliingelephant/VMC

## F. row-wise sampling extension

T. Chen, J. Liu, Y. Wu, P. Zhang, Y. Deng  
**Variational Monte Carlo with row-update projected entangled-pair states and its applications to quantum spin glasses**  
PRB accepted (2026)  
https://doi.org/10.1103/9z82-vlky  
https://arxiv.org/abs/2601.20608

Public code:
- not identified as of 2026-08-23

---

# 12. Repository comparison table

| Repo | PEPS | fPEPS | VMC/SR | real-time | gauge removal | main language | official paper code? |
|---|---:|---:|---:|---:|---:|---|---|
| `fliingelephant/VMC` | yes | unclear / inspect | yes | **yes** | **yes** | Python/JAX | no; independent implementation |
| `QuantumLiquids/PEPS` | **yes** | **yes** | **yes** | not advertised | not Wu–Nys style | C++ | related ecosystem, not tied to one paper |
| `DirectSamplingPEPS.jl` | **yes** | no obvious focus | sampling/VMC | no | no | Julia | **yes** |
| `sjdu10/vmc_torch` | yes | **yes** | **yes** | not primary focus | no | Python/PyTorch | **yes for its listed research line** |
| `netket/netket` | ansatz-agnostic | ansatz-dependent | **yes** | **yes** | generic QGT tools | Python/JAX | **yes** |
| `netket/netket_fidelity` | ansatz-agnostic | ansatz-dependent | projected VMC | **yes** | different formulation | Python/JAX | **yes** |
| `markusschmitt/vmc_jax` | ansatz-agnostic | ansatz-dependent | **yes** | **yes** | generic | Python/JAX | **yes** |
| `yantaow/open_data` | data only | data only | no engine | no engine | no engine | misc | official **data** |
| `LiuWenyuan-Phy/2d-hubbard-data-fPEPS-PRL2025` | data only | data only | no engine | no | no | text/data | official **data** |

---

# 13. Recommended Codex project organization

A practical project layout:

```text
peps-vmc-study/
├── README.md
├── NOTES.md                     # this note
├── papers/
│   ├── Liu2021_PRB_finite_PEPS_VMC.pdf
│   ├── Vieijra2021_PRB_direct_sampling.pdf
│   ├── WuLiu2025_PRL_LGT.pdf
│   ├── WuDai2026_CPB_fPEPS_VMC.pdf
│   ├── WuNys2026_PRXQ_tVMC.pdf
│   └── ChenEtAl2026_row_update.pdf
├── upstream/
│   ├── fliingelephant_VMC/
│   ├── QuantumLiquids_PEPS/
│   ├── DirectSamplingPEPS.jl/
│   ├── vmc_torch/
│   ├── netket/
│   ├── netket_fidelity/
│   └── vmc_jax/
├── code_notes/
│   ├── fliingelephant_architecture.md
│   ├── quantumliquids_architecture.md
│   ├── sampling_comparison.md
│   ├── qgt_sr_tdvp.md
│   └── fermionic_peps.md
└── experiments/
    ├── exact_3x3_tdvp/
    ├── peps_sampling/
    └── future_fermion_test/
```

Suggested clone commands:

```bash
mkdir -p upstream
cd upstream

git clone https://github.com/fliingelephant/VMC.git fliingelephant_VMC
git clone https://github.com/QuantumLiquids/PEPS.git QuantumLiquids_PEPS
git clone https://github.com/tvieijra/DirectSamplingPEPS.jl.git
git clone https://github.com/sjdu10/vmc_torch.git
git clone https://github.com/netket/netket.git
git clone https://github.com/netket/netket_fidelity.git
git clone https://github.com/markusschmitt/vmc_jax.git
```

---

# 14. Suggested Codex investigation tasks

These are good first prompts/tasks for Codex after cloning the repositories.

## Task 1 — map the Wu–Nys implementation

Ask Codex:

```text
Trace the complete call graph for one time step in
fliingelephant/VMC, starting from TDVPDriver and ending at the
updated PEPS parameters. Identify:
1. where samples are generated,
2. where log derivatives/Jacobians are computed,
3. where the QGT or minSR matrix is built,
4. where PEPS gauge directions are removed,
5. which linear solver is used,
6. which integrator advances the state.
Give file names, classes/functions, and data shapes.
```

## Task 2 — isolate gauge-removal implementation

```text
Read src/vmc/gauge and the TDVP/QGT code.
Match every gauge-removal operation to the equations in
Wu & Nys, PRX Quantum 7, 033035 (2026).
Distinguish:
- PEPS virtual gauge redundancy,
- normalization/projective zero mode,
- minSR sample-space reduction.
State which parts are exact algebra and which parts depend on sampling.
```

## Task 3 — compare PEPS engines

```text
Compare fliingelephant/VMC and QuantumLiquids/PEPS at the level of:
- PEPS tensor storage
- boundary-MPS contraction
- sequential/local sampling
- environment reuse
- energy/local-energy evaluation
- gradient evaluation
- SR
- fermionic signs/symmetry
- parallelization
Identify which codebase is easier to extend to fermionic PEPS-tVMC.
```

## Task 4 — understand the sampling lineage

```text
Compare:
1. Liu et al. PRB 103, 235155 (2021),
2. Vieijra et al. PRB 104, 235141 (2021),
3. Wu & Dai CPB 35, 020502 (2026),
4. Chen et al. PRB accepted (2026),
in terms of the actual probability distribution sampled,
proposal/update rule, boundary contraction, autocorrelation,
detailed balance, and computational scaling.
Map each method to the public repos where possible.
```

## Task 5 — fermionic feasibility audit

```text
Determine whether fliingelephant/VMC currently supports a genuine
fermionic PEPS with swap gates or Grassmann/parity-aware tensors.
Search the entire codebase, tests, examples, dependencies, and issues.
Do not infer fermion support merely from the cited papers.
If absent, list the minimum components required to port the
Wu–Dai fermionic PEPS-VMC algorithm.
```

This last task is important because the `fliingelephant/VMC` README does not by itself establish that the full fermionic PEPS stack needed for the Chern-insulator example is implemented.

---

# 15. What is public and what is still missing

## Public enough to study now

- finite PEPS ground-state VMC:
  - `QuantumLiquids/PEPS`
- direct PEPS sampling:
  - `DirectSamplingPEPS.jl`
- JAX PEPS-tVMC/gauge-removal prototype:
  - `fliingelephant/VMC`
- general QGT/SR/tVMC machinery:
  - `NetKet`
  - `jVMC`
  - `netket_fidelity`
- fermionic/TNS VMC alternative ecosystem:
  - `vmc_torch`
  - `quimb`
  - `symmray`

## Not publicly identified

- the actual Wu–Nys production JAX code used for the large \(10\times10\)–\(13\times13\) real-time simulations
- a full official Wu–Dai fermionic PEPS-VMC production code
- code for the 2026 row-update PEPS-VMC paper

## Main technical uncertainty for a new project

The biggest unanswered question is not whether PEPS-tVMC mathematics can be implemented—the public JAX prototype shows that it can—but whether there is already a **complete, robust fermionic PEPS + sequential sampling + TDVP/gauge-removal stack** that can reproduce the Wu–Nys Chern-insulator calculation without substantial new engineering.

That is the first thing Codex should audit.

---

# 16. Priority ranking for this project

### Priority A — clone and read immediately

1. https://github.com/fliingelephant/VMC
2. https://github.com/QuantumLiquids/PEPS
3. https://github.com/tvieijra/DirectSamplingPEPS.jl
4. https://github.com/sjdu10/vmc_torch

### Priority B — infrastructure/reference

5. https://github.com/netket/netket
6. https://github.com/netket/netket_fidelity
7. https://github.com/markusschmitt/vmc_jax
8. https://github.com/QuantumLiquids/TensorToolkit
9. https://github.com/jcmgray/quimb
10. https://github.com/jcmgray/symmray

### Priority C — official data / paper cross-checks

11. https://github.com/yantaow/open_data
12. https://github.com/LiuWenyuan-Phy/2d-hubbard-data-fPEPS-PRL2025

---

# 17. Compact bibliography

1. W.-Y. Liu, Y.-Z. Huang, S.-S. Gong, Z.-C. Gu,  
   *Accurate simulation for finite projected entangled pair states in two dimensions*,  
   Phys. Rev. B 103, 235155 (2021).  
   https://doi.org/10.1103/PhysRevB.103.235155

2. T. Vieijra, J. Haegeman, F. Verstraete, L. Vanderstraeten,  
   *Direct sampling of projected entangled-pair states*,  
   Phys. Rev. B 104, 235141 (2021).  
   https://doi.org/10.1103/PhysRevB.104.235141

3. F. Vicentini et al.,  
   *NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems*,  
   SciPost Physics Codebases 7 (2022).  
   https://doi.org/10.21468/SciPostPhysCodeb.7

4. M. Schmitt, M. Reh,  
   *jVMC: Versatile and performant variational Monte Carlo leveraging automated differentiation and GPU acceleration*,  
   SciPost Physics Codebases 2 (2022).  
   https://arxiv.org/abs/2108.03409

5. A. Sinibaldi, C. Giuliani, G. Carleo, F. Vicentini,  
   *Unbiasing time-dependent Variational Monte Carlo by projected quantum evolution*,  
   Quantum 7, 1131 (2023).  
   https://doi.org/10.22331/q-2023-10-10-1131

6. W.-Y. Liu, S.-J. Du, R. Peng, J. Gray, G. K.-L. Chan,  
   *Tensor Network Computations That Capture Strict Variationality, Volume Law Behavior, and the Efficient Representation of Neural Network States*,  
   Phys. Rev. Lett. 133, 260404 (2024).  
   https://arxiv.org/abs/2405.03797

7. Y. Wu, W.-Y. Liu,  
   *Accurate Gauge-Invariant Tensor-Network Simulations for Abelian Lattice Gauge Theory in (2+1)D: Ground-State and Real-Time Dynamics*,  
   Phys. Rev. Lett. 135, 130401 (2025).  
   https://doi.org/10.1103/3m3j-ds18

8. Y. Wu, Z. Dai,  
   *Algorithms for variational Monte Carlo calculations of fermion projected entangled pair states in the swap gates formulation and the detailed balance of tensor network sequential sampling*,  
   Chinese Physics B 35, 020502 (2026).  
   https://doi.org/10.1088/1674-1056/ae2673

9. S.-J. Du, A. Chen, G. K.-L. Chan,  
   *Neuralized fermionic tensor networks for quantum many-body systems*,  
   Phys. Rev. B 113, 085134 (2026).  
   https://doi.org/10.1103/x8vl-qf14

10. Y. Wu, J. Nys,  
    *Real-Time Dynamics in Two Dimensions with Tensor Network States via Time-Dependent Variational Monte Carlo Method*,  
    PRX Quantum 7, 033035 (2026).  
    https://doi.org/10.1103/tggc-8fjx

11. T. Chen, J. Liu, Y. Wu, P. Zhang, Y. Deng,  
    *Variational Monte Carlo with row-update projected entangled-pair states and its applications to quantum spin glasses*,  
    Phys. Rev. B, accepted 13 Aug 2026.  
    https://doi.org/10.1103/9z82-vlky

---

# 18. Bottom line

For a Codex project aimed specifically at understanding or reproducing PEPS-tVMC, the most useful combination is:

```text
fliingelephant/VMC
    -> TDVP, QGT, gauge removal, JAX architecture

QuantumLiquids/PEPS
    -> mature finite PEPS-VMC and fermionic ground-state machinery

DirectSamplingPEPS.jl
    -> clean sampling reference

Wu–Dai CPB paper
    -> fermionic PEPS + swap-gate + sequential-sampling details

NetKet / jVMC
    -> general tVMC/SR numerical infrastructure
```

The key caution is:

> Do not treat any currently public repository as automatically equivalent to the unpublished production code behind Wu–Nys Fig. 2. The closest public implementation is `fliingelephant/VMC`, but it is an independent, actively developing codebase and should be validated module by module.
