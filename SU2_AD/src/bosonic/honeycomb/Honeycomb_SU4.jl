module HoneycombSU4

# Settings.jl already provides the repository's TensorKit compatibility layer.
using TensorKit, TensorKitSectors, SUNRepresentations
using LinearAlgebra, Random, Zygote, KrylovKit, ChainRulesCore
using Zygote: @ignore_derivatives

include("../Settings.jl")
include("../Settings_cell.jl")
include("../AD_lib.jl")
include("../SUN_spin_op.jl")
include("../CTMRG.jl")
include("../CTMRG_unitcell.jl")
include("../triangle/triangle_spin_model_cell.jl")
include("../triangle/simple_update/triangle_SimpleUpdate_iPEPS.jl")

# CTMRG's legacy dictionary conversion is unavailable in new TensorKit.
# Specialize on real singular values, without changing the shared old code.
function sdiag_inv_sqrt(S::DiagonalTensorMap{<:Real})
    values=sort([v for (c,b) in blocks(S) for v in diag(b) for _=1:Int(dim(c))];rev=true);
    boundary=length(values)>chi ? values[chi+1] : 0.0;
    cutoff=projector_trun_tol*maximum(values);
    R=copy(S);
    for (c,b) in blocks(R)
        for k=1:size(b,1)
            v=b[k,k];
            keep=v>cutoff && (boundary==0 || (v>boundary && abs(v-boundary)>multiplet_tol*boundary));
            b[k,k]=keep ? inv(sqrt(v)) : 0.0;
        end
    end
    return R
end

include("multiplet_truncation.jl")
include("honeycomb_SimpleUpdate.jl")
include("star/honeycomb_star_state.jl")
include("honeycomb_spin_model.jl")

export initial_honeycomb_iPEPS, initial_honeycomb_star_iPEPS, honeycomb_virtual_space, honeycomb_bonds,
       Hamiltonian_honeycomb_SU4, itebd_honeycomb, honeycomb_energy_simple,
       honeycomb_energy_CTM, convert_honeycomb_to_iPEPS,
       LS_CTMRG_settings, Algrithm_CTMRG_settings, WholeMultipletBudget

end
