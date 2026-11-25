# Optional adapter. Include AFTER KagomeFermionicZ2.jl, in the same parent module.
module KagomeFermionicCTM
using TensorKit, LinearAlgebra, Zygote, ChainRulesCore, KrylovKit, Random
using ..KagomeFermionicZ2
import ..KagomeFermionicZ2: evaluate_ob_cell
const KF = KagomeFermionicZ2

module Legacy
using TensorKit, LinearAlgebra, Zygote, ChainRulesCore, KrylovKit, Random
const SRC = normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(SRC,"bosonic","Settings.jl"))
include(joinpath(SRC,"bosonic","Settings_cell.jl"))
include(joinpath(SRC,"bosonic","CTMRG.jl"))
include(joinpath(SRC,"fermionic","Fermionic_CTMRG.jl"))
include(joinpath(SRC,"fermionic","Fermionic_CTMRG_unitcell.jl"))
include(joinpath(SRC,"fermionic","square_Hubbard_model_cell.jl"))
include(joinpath(SRC,"fermionic","swap_funs.jl"))
include(joinpath(SRC,"fermionic","double_layer_funs.jl"))
Lx = 1
Ly = 1
chi = 20
# Z2 has only one-dimensional irreps. Ordinary rank truncation preserves the
# requested symmetry and does not require the private truncmultiplet fork.
multiplet_tol = 0.0
projector_trun_tol = 1e-8
algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"

# The shared implementation enumerates selected cell sizes. Keep arbitrary
# magnetic cells local to this adapter, with the same per-site rotations.
function rotate_AA_cell(cell::Tuple,construct_double_layer::Bool)
    rotated=map(column->map(a->rotate_AA(a,construct_double_layer),column),cell)
    # Keep field names static: Zygote's pullback for getfield(a, d::Int)
    # fails for the legacy Tset_struc. These projections are exactly the
    # original four rotations, with no change to the contraction or signs.
    return (T1=map(column->map(a->a.T1,column),rotated),
            T2=map(column->map(a->a.T2,column),rotated),
            T3=map(column->map(a->a.T3,column),rotated),
            T4=map(column->map(a->a.T4,column),rotated))
end

# Local replacement for the legacy inverse square root: exact product states
# have zero Schmidt values, for which the old 0 * Inf expression gives NaN.
# Z2 sectors have unit dimension, so keep the largest chi scalar values.
function sdiag_inv_sqrt(S::DiagonalTensorMap)
    TensorKit.sectortype(S) == Z2Irrep || throw(ArgumentError("Z2 CTMRG only"))
    entries = [(real(b[i,i]),q,i) for (q,b) in blocks(S) for i in axes(b,1)]
    sort!(entries;by=first,rev=true)
    isempty(entries) && throw(ArgumentError("empty CTMRG spectrum"))
    cutoff = projector_trun_tol*entries[1][1]
    R = deepcopy(S)
    for (_,b) in blocks(R)
        fill!(b,0)
    end
    for (value,q,i) in Iterators.take(entries,chi)
        value > cutoff && value > 0 && (block(R,q)[i,i] = inv(sqrt(value)))
    end
    return R
end
end

export measure_ctmrg, evaluate_ob_cell

function _blocked_operators(U_phy,parameters)
    P = KF.physical_space()
    F = space(U_phy,1)
    fusion = reshape(convert(Array,U_phy),64,64)
    ops = KF._triangle_operators()
    even(M) = U_phy*KF._physical_map(M)*U_phy'
    # Same dummy-spin convention as Hamiltonians_spinful_Z2. Internal JW
    # strings are already in ops.c, before the three sites are fused.
    dummy = Rep[ℤ₂](1=>2)
    Cs,Cdags = TensorMap[],TensorMap[]
    for site in 1:3
        c = zeros(ComplexF64,2,64,64)
        cd = similar(c)
        for spin in 1:2
            c[spin,:,:] = fusion*ops.c[site][spin]*fusion'
            cd[spin,:,:] = fusion*ops.c[site][spin]'*fusion'
        end
        push!(Cs,TensorMap(c,dummy*F,F))
        cdmap = TensorMap(permutedims(cd,(2,3,1)),F,F*dummy)
        push!(Cdags,TensorKit.permute(cdmap,((3,1),(2,))))
    end
    H = Hamiltonian_triangle(parameters;orientation=:up,onsite_fraction=1.0)
    return (;H=U_phy*H*U_phy',n=map(even,ops.n),doublon=map(even,ops.doublon),pair=map(even,ops.pair),Cs,Cdags)
end

"""Full lattice energy per PHYSICAL SITE (three sites per kagome cell)."""
function evaluate_ob_cell(parameters,A_cell,AA_cell,CTM,ctm_setting,U_phy)
    Lx,Ly = length(A_cell),length(A_cell[1])
    Legacy.Lx = Lx
    Legacy.Ly = Ly
    op = _blocked_operators(U_phy,parameters)
    up = zeros(ComplexF64,Lx,Ly)
    down_hop = zeros(ComplexF64,Lx,Ly,3)
    n = zeros(ComplexF64,Lx,Ly,3)
    doublon = similar(n)
    pairing = similar(n)
    for y in 1:Ly, x in 1:Lx
        up[x,y] = Legacy.ob_onsite(CTM,op.H,A_cell,AA_cell,x-1,y-1,ctm_setting)
        # dn(x,y) = [a(x+1,y), b(x,y), c(x,y+1)].
        # hopping_x returns <b†(x,y) a(x+1,y)>; conjugate for the a->b bond.
        down_hop[x,y,1] = conj(Legacy.hopping_x(CTM,op.Cdags[2],op.Cs[1],
                                               A_cell,AA_cell,x-1,y-1,ctm_setting))
        # hopping_y acts on the RIGHT column of its 2x2 patch.
        down_hop[x,y,2] = Legacy.hopping_y(CTM,op.Cdags[2],op.Cs[3],
                                          A_cell,AA_cell,x-2,y-1,ctm_setting)
        # The legacy LD->RU odd string is opposite to our c†_c c_a bond
        # after exchanging the down simplex's D/R legs during blocking.
        down_hop[x,y,3] = -Legacy.hopping_diagonala(CTM,op.Cdags[3],op.Cs[1],
                                                  A_cell,AA_cell,x-1,y-1,ctm_setting)
        for site in 1:3
            n[x,y,site] = Legacy.ob_onsite(CTM,op.n[site],A_cell,AA_cell,x-1,y-1,ctm_setting)
            doublon[x,y,site] = Legacy.ob_onsite(CTM,op.doublon[site],A_cell,AA_cell,x-1,y-1,ctm_setting)
            pairing[x,y,site] = Legacy.ob_onsite(CTM,op.pair[site],A_cell,AA_cell,x-1,y-1,ctm_setting)
        end
    end
    ts = KF._hoppings(parameters,:dn)
    ed = [-2real(sum(ts[k]*down_hop[x,y,k] for k in 1:3)) for x in 1:Lx,y in 1:Ly]
    return (;energy_per_site=(real(sum(up))+sum(ed))/(3Lx*Ly),up_with_onsite=up,
             down_kinetic=ed,down_hopping=down_hop,n,double_occupancy_set=doublon,
             density=real(sum(n))/(3Lx*Ly),double_occupancy=real(sum(doublon))/(3Lx*Ly),
             pair_amplitude=sum(pairing)/(3Lx*Ly),pair_amplitude_set=pairing,
             method=:CTMRG)
end

"""
Reuse Fermionic_CTMRG_cell and the existing sign-correct hopping contractions.
Returns convergence diagnostics with the observables. The legacy adapter has
module-local globals, so simultaneous threaded calls are not supported.
"""
function measure_ctmrg(state,parameters;chi=20,maxiter=100,tol=1e-6,
                       trun_tol=1e-8,verbose=true,CTM0=nothing)
    chi > 0 && maxiter > 0 && tol > 0 || throw(ArgumentError("invalid CTMRG settings"))
    check_state(state)
    Legacy.Lx,Legacy.Ly = size(state.T_u)
    Legacy.chi = chi
    Legacy.projector_trun_tol = trun_tol
    settings = Legacy.LS_CTMRG_settings()
    settings.CTM_conv_tol = tol
    settings.CTM_ite_nums = maxiter
    settings.CTM_trun_tol = trun_tol
    settings.CTM_ite_info = verbose
    settings.CTM_conv_info = true
    settings.grad_checkpoint = false
    init = Legacy.initial_condition(init_type="PBC",reconstruct_CTM=isnothing(CTM0),reconstruct_AA=true)
    A_cell = convert_iPESS_to_iPEPS(state)
    CTM,AA,_,_,_,_,iterations,error = Legacy.Fermionic_CTMRG_cell(A_cell,chi,init,CTM0,settings)
    _,_,U_phy = build_A_kagome(state,1,1)
    observables = evaluate_ob_cell(parameters,A_cell,AA,CTM,settings,U_phy)
    return (;observables,CTM,iterations,error,converged=isfinite(error) && error<tol,chi)
end
end
