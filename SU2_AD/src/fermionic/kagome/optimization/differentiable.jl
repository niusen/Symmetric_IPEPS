# The explicit Z2 Fermi permutation is a fixed real unitary map. Its adjoint
# is the inverse permutation, including the same odd/odd crossing signs.
# Rules apply ONLY to the new kagome functions, not to legacy TensorKit calls.
function _restore_partition(A, template)
    n = KF._rank(template)
    k = length(codomain(template))
    return TensorKit.permute(A,(Tuple(1:k),Tuple(k+1:n)))
end

function ChainRulesCore.rrule(::typeof(KF.fpermute),A,order::Tuple)
    result = KF.fpermute(A,order)
    inverse = Tuple(invperm(collect(order)))
    function pullback(delta)
        d = unthunk(delta)
        d isa AbstractZero && return (NoTangent(),ZeroTangent(),NoTangent())
        return NoTangent(),_restore_partition(KF.fpermute(d,inverse),A),NoTangent()
    end
    return result,pullback
end

function ChainRulesCore.rrule(::typeof(KF._crossing),A,i,j)
    result = KF._crossing(A,i,j)
    function pullback(delta)
        d = unthunk(delta)
        d isa AbstractZero && return (NoTangent(),ZeroTangent(),NoTangent(),NoTangent())
        return NoTangent(),_restore_partition(KF._crossing(d,i,j),A),NoTangent(),NoTangent()
    end
    return result,pullback
end

# Work directly with independent symmetry-block entries: no forbidden entries
# are introduced by the optimizer, and a Z2 sector is never resized during AD.
_coefficients(A) = copy(A.data)
_from_coefficients(v,template) = TensorMap(v,space(template))

function ChainRulesCore.rrule(::typeof(_from_coefficients),v,template)
    result = _from_coefficients(v,template)
    function pullback(delta)
        d = unthunk(delta)
        return NoTangent(),d isa AbstractZero ? zero(v) : copy(d.data),NoTangent()
    end
    return result,pullback
end

function _fusion(P=KF.physical_space())
    return unitary(fuse(P*P*P),P*P*P)
end

"""Linear embedding of an unfactorized UP triangle, used for AD and FU only."""
function _block_triangle(W,Td,U_phy)
    W = KF.fpermute(W,(1,2,4,5,6,3))
    T = KF.fpermute(Td,(2,1,3))
    M = TensorKit.permute(W,((1,2,3,4,5),(6,))) *
        TensorKit.permute(T,((1,),(2,3)))
    A = KF.fpermute(M,(1,7,6,4,2,3,5))
    A = KF._crossing(KF._crossing(KF._crossing(A,1,2),1,3),2,3)
    @tensor Af[-1,-2,-3,-4,-5] := A[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3]
    return KF._allout(Af)
end

function _cell_tensor(Tu,Td,Bs,U_phy)
    oriented = map(B -> KF._oriented_bond(B,:up),Bs)
    W = KF._simplex_wavefunction(Tu,oriented)
    A = _block_triangle(W,Td,U_phy)
    return A/norm(A)
end

function _settings(;iterations,tol=0.0,verbose=false)
    settings = CTM.LS_CTMRG_settings()
    settings.CTM_ite_nums = iterations
    settings.CTM_conv_tol = tol
    settings.CTM_ite_info = verbose
    settings.CTM_conv_info = true
    settings.grad_checkpoint = false
    return settings
end

function _configure(Lx,Ly,chi;trun_tol=1e-10)
    CTM.Lx,CTM.Ly,CTM.chi = Lx,Ly,chi
    CTM.projector_trun_tol = trun_tol
    return nothing
end

function _environment(A,chi,settings;seed=nothing)
    init = CTM.initial_condition(init_type="PBC",reconstruct_CTM=isnothing(seed),reconstruct_AA=true)
    result = CTM.Fermionic_CTMRG_cell(A,chi,init,seed,settings)
    return (;CTM=result[1],AA=result[2],iterations=result[7],error=result[8])
end

# A separate loop is needed for AD: the legacy driver also has a stagnation
# break after 30 sweeps, even when its convergence tolerance is set to zero.
# Reuse its initialization/projectors/updates, but execute EXACTLY N sweeps.
function _fixed_environment(A,chi,settings)
    layers = map(column -> map(a -> CTM.build_double_layer_swap(a',a),column),A)
    extract(k) = map(column -> map(layer -> layer[k],column),layers)
    AA = extract(1)
    initial = CTM.init_CTM_cell(chi,A,AA,extract(2),extract(3),extract(4),extract(5),"PBC",false)
    rotated = CTM.rotate_AA_cell(AA,true)
    C,T = initial.Cset,initial.Tset
    for _ in 1:settings.CTM_ite_nums
        for direction in (3,4,1,2)
            C,T = CTM.CTM_ite_cell_continuous_update(C,T,CTM.get_Tset(rotated,direction),
                      chi,direction,settings.CTM_trun_tol,false,settings.projector_strategy,
                      false,settings.svd_lanczos_tol,true)
        end
    end
    return (;CTM=(Cset=C,Tset=T),AA,iterations=settings.CTM_ite_nums)
end

function _energy(A,AA,env,settings,operators,ts)
    Lx,Ly = length(A),length(A[1])
    energies = map(CartesianIndices((Lx,Ly))) do index
        x,y = Tuple(index)
        eu = CTM.ob_onsite(env,operators.H,A,AA,x-1,y-1,settings)
        hx = conj(CTM.hopping_x(env,operators.Cdags[2],operators.Cs[1],A,AA,x-1,y-1,settings))
        hy = CTM.hopping_y(env,operators.Cdags[2],operators.Cs[3],A,AA,x-2,y-1,settings)
        hd = -CTM.hopping_diagonala(env,operators.Cdags[3],operators.Cs[1],A,AA,x-1,y-1,settings)
        real(eu)-2real(ts[1]*hx+ts[2]*hy+ts[3]*hd)
    end
    return sum(energies)/(3Lx*Ly)
end
