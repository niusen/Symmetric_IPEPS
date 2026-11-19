# Canonical local form: B_i is the even map (external_i,physical_i) <- internal_i.
# The tensor product of these even maps has the usual matrix tensor product.
# Fermi swaps are needed to get to/from this form and to collect physical legs.
function _oriented_bond(B,orientation)
    order = orientation == :up ? (1,3,2) : (2,3,1)
    return TensorKit.permute(fpermute(B,order), ((1,2),(3,)))
end

function _simplex_wavefunction(T,Bs)
    # Sequential contractions avoid forming a rank-9 Kronecker tensor.
    @tensor A[-1,-2,-3,-4] := Bs[1][-1,-2,1]*T[1,-3,-4]
    @tensor A2[-1,-2,-3,-4,-5] := A[-1,-2,1,-5]*Bs[2][-3,-4,1]
    @tensor W[-1,-2,-3,-4,-5,-6] := A2[-1,-2,-3,-4,1]*Bs[3][-5,-6,1]
    return _allout(W)
end

function triangle_wavefunction(state,x,y,orientation; weighted=true)
    sites = triangle_sites(state,x,y,orientation)
    external = orientation == :up ? state.lambda_d : state.lambda_u
    leg = orientation == :up ? 1 : 2
    Bs = map(sites) do site
        B = state.B[site...]
        weighted && (B = _leg_map(B,external[site...],leg))
        _oriented_bond(B,orientation)
    end
    Lx,Ly = size(state.T_u)
    T = (orientation == :up ? state.T_u : state.T_d)[mod1(x,Lx),mod1(y,Ly)]
    return _simplex_wavefunction(T,Bs)
end

function _apply_gate(W,gate)
    Wp = fpermute(W,(1,3,5,2,4,6)) # external a,b,c; physical a,b,c
    @tensor Wg[-1,-2,-3,-4,-5,-6] := Wp[-1,-2,-3,1,2,3]*gate[-4,-5,-6,1,2,3]
    return fpermute(Wg,(1,4,2,5,3,6))
end

function _hosvd(W,Dmax,trun_tol)
    W = _normalized(W)
    Us, Ss = TensorMap[], TensorMap[]
    for order in ((1,2,3,4,5,6),(3,4,1,2,5,6),(5,6,1,2,3,4))
        M = TensorKit.permute(fpermute(W,order), ((1,2),(3,4,5,6)))
        U,S,_ = _svd(M,Dmax,trun_tol)
        push!(Us,U)
        push!(Ss,TensorMap(S))
    end
    # Project each even isometry in the SAME canonical pair order used in
    # _simplex_wavefunction; no unrecorded exchange of physical legs occurs.
    Ua,Ub,Uc = Us
    @tensor core[-1,-2,-3] := conj(Ua[1,2,-1])*conj(Ub[3,4,-2])*conj(Uc[5,6,-3])*W[1,2,3,4,5,6]
    reconstructed = _simplex_wavefunction(core,Us)
    discarded = norm(W-reconstructed)^2
    return _normalized(core),Us,Ss,discarded
end

"""One triangle gate and parity-preserving HOSVD, including exterior lambdas."""
function triangle_update!(state,x,y,orientation,gate; Dmax::Integer,trun_tol=1e-10)
    Dmax > 0 || throw(ArgumentError("Dmax must be positive"))
    0 <= trun_tol < 1 || throw(ArgumentError("trun_tol must lie in [0,1)"))
    sites = triangle_sites(state,x,y,orientation)
    W = triangle_wavefunction(state,x,y,orientation)
    core,Us,Ss,discarded = _hosvd(_apply_gate(W,gate),Dmax,trun_tol)
    external = orientation == :up ? state.lambda_d : state.lambda_u
    current = orientation == :up ? state.lambda_u : state.lambda_d
    external_leg = orientation == :up ? 1 : 2
    inverse_order = orientation == :up ? (1,3,2) : (3,1,2)
    for (k,site) in enumerate(sites)
        B = fpermute(Us[k],inverse_order)
        state.B[site...] = _normalized(_leg_map(B,_pinv_weight(external[site...]),external_leg))
        # S acts on the simplex space; its transpose acts on the dual B leg.
        current[site...] = _normalized(TensorKit.permute(Ss[k],((2,),(1,))))
    end
    Lx,Ly = size(state.T_u)
    (orientation == :up ? state.T_u : state.T_d)[mod1(x,Lx),mod1(y,Ly)] = core
    return discarded
end

function sweep!(state,gate,orientation; Dmax,trun_tol=1e-10)
    Lx,Ly = size(state.T_u)
    errors = zeros(Lx,Ly)
    for y in 1:Ly, x in 1:Lx
        errors[x,y] = triangle_update!(state,x,y,orientation,gate;Dmax,trun_tol)
    end
    return errors
end

"""
Second-order evolution: (dn dt/2, up dt, dn dt/2) per step, total time tau.
Returns one diagnostics record per completed step. No energy-based stopping
rule is imposed, since simple-update local energies are only estimates.
"""
function itebd!(state,parameters,tau,dt,Dmax; trun_tol=1e-10,callback=nothing)
    isfinite(tau) && tau >= 0 || throw(ArgumentError("invalid tau"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("invalid dt"))
    Dmax > 0 && 0 <= trun_tol < 1 || throw(ArgumentError("invalid truncation settings"))
    steps = round(Int,tau/dt)
    isapprox(steps*dt,tau;atol=1e-12,rtol=1e-10) ||
        throw(ArgumentError("tau must be an integer multiple of dt"))
    check_state(state)
    gu,_ = trotter_gate(Hamiltonian_triangle(parameters;orientation=:up),dt)
    _,gd_half = trotter_gate(Hamiltonian_triangle(parameters;orientation=:dn),dt)
    history = NamedTuple[]
    for step in 1:steps
        e1 = sweep!(state,gd_half,:dn;Dmax,trun_tol)
        e2 = sweep!(state,gu,:up;Dmax,trun_tol)
        e3 = sweep!(state,gd_half,:dn;Dmax,trun_tol)
        check_state(state;Dmax)
        record = (;step,tau=step*dt,discarded_weight=max(maximum(e1),maximum(e2),maximum(e3)))
        push!(history,record)
        isnothing(callback) || callback(state,record)
    end
    return history
end
