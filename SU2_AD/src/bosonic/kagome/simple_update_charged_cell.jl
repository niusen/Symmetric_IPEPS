"""Experimental SU(2)-covariant kagome PESS with one auxiliary spin per sqrt3 cell."""
module KagomeChargedSU2

using TensorKit, LinearAlgebra, Random, JLD2
include(joinpath(@__DIR__,"simple_update_cells.jl"))
using .KagomeSU2Cells
const BaseSU = KagomeSU2Cells
const Kernel = BaseSU.Legacy

export ChargedKagomeState, initialize_charged, check_charged, itebd_charged,
       polarized_dense_cells, closed_sqrt3_wavefunction, exact_torus_observables,
       save_charged_checkpoint, load_charged_checkpoint

"""
All local tensors remain SU(2) invariant BEFORE the extra index is polarized.
Only T_u(anchor) has a fourth, open spin-j index. Projecting it onto m_q=+j
selects physical S^z=-j per nine-spin cell (H=H_exchange+h*sum(Sz)).
Tracing this index instead would give zero magnetization.
"""
struct ChargedKagomeState
    raw::BaseSU.KagomeSUCell
    anchor::Int
    auxiliary_spin::Rational{Int}
end

function initialize_charged(;Vv=SU2Space(0=>1,1/2=>1), auxiliary_spin=1//2,
                            anchor::Integer=1, rng=Random.default_rng(), complex=true)
    j = rationalize(Int,auxiliary_spin)
    j > 0 && denominator(2j) == 1 || throw(ArgumentError("auxiliary spin must be positive integer/half-integer"))
    1 <= anchor <= 3 || throw(ArgumentError("anchor must be 1, 2 or 3"))
    geometry = BaseSU.KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3)
    raw = BaseSU.initialize_cells(geometry,Vv;rng,complex)
    Q = SU2Space(j=>1)
    scalar_type = complex ? ComplexF64 : Float64
    # Original Tu has virtual axes (up-a,up-b,up-c). The sole change is
    # appending an auxiliary spin as axis 4, with the SAME virtual duals.
    core = randn(rng,scalar_type,Vv,Vv'*Vv'*Q')
    core = permute(core,((1,2,3,4),()))
    norm(core) > 0 || throw(ArgumentError("initial virtual space cannot carry this auxiliary spin"))
    raw.cells[anchor].T_u = core/norm(core)
    return check_charged(ChargedKagomeState(raw,Int(anchor),j))
end

function check_charged(state::ChargedKagomeState;Dmax=nothing)
    BaseSU.check_state(state.raw;Dmax)
    g = state.raw.geometry
    g.pattern == :sqrt3xsqrt3 && length(state.raw.cells) == 3 ||
        throw(ArgumentError("this prototype requires exactly one sqrt3xsqrt3 cell"))
    expected = BaseSU.KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3)
    g.site_map == expected.site_map && g.representatives == expected.representatives ||
        throw(ArgumentError("sqrt3 representative mapping has been modified"))
    for (r,c) in enumerate(state.raw.cells)
        numind(c.T_u) == (r == state.anchor ? 4 : 3) ||
            throw(ArgumentError("only the selected Tu may have an auxiliary leg"))
        numind(c.T_d) == 3 || throw(ArgumentError("Td must retain three virtual legs"))
    end
    space(state.raw.cells[state.anchor].T_u,4) == SU2Space(state.auxiliary_spin=>1) ||
        throw(SpaceMismatch("wrong auxiliary SU(2) representation"))
    return state
end

function _charged_up(Tu,Ba,Bb,Bc,wa,wb,wc,gate,trun_tol,Dmax)
    # Same lambda dressing and physical gate order as original Tri_T_up.
    @tensor a[:] := Ba[1,-2,-3]*wa[1,-1]
    @tensor b[:] := Bb[1,-2,-3]*wb[1,-1]
    @tensor c[:] := Bc[1,-2,-3]*wc[1,-1]
    @tensor A[:] := Tu[1,2,3,-7]*a[-1,1,-2]*b[-3,2,-4]*c[-5,3,-6]
    @tensor A[:] := gate[-2,-4,-6,1,2,3]*A[-1,1,-3,2,-5,3,-7]
    A = A/norm(A)
    # Q is a SPECTATOR in each complementary space, never truncated/fused
    # into a virtual bond. Each retained bond remains a full SU(2) multiplet.
    Ua,la,Va = Kernel.tsvd(A,(1,2),(3,4,5,6,7);trunc=Kernel.truncdim(Dmax))
    Ua,la,Va = Kernel.Truncations(Ua,la,Va,Dmax,trun_tol)
    Ub,lb,Vb = Kernel.tsvd(A,(3,4),(1,2,5,6,7);trunc=Kernel.truncdim(Dmax))
    Ub,lb,Vb = Kernel.Truncations(Ub,lb,Vb,Dmax,trun_tol)
    Uc,lc,Vc = Kernel.tsvd(A,(5,6),(1,2,3,4,7);trunc=Kernel.truncdim(Dmax))
    Uc,lc,Vc = Kernel.Truncations(Uc,lc,Vc,Dmax,trun_tol)
    @tensor core[:] := A[1,2,3,4,5,6,-4]*Ua'[-1,1,2]*Ub'[-2,3,4]*Uc'[-3,5,6]
    Ba = permute(Ua,((1,3,2),()))
    Bb = permute(Ub,((1,3,2),()))
    Bc = permute(Uc,((1,3,2),()))
    ia,ib,ic = pinv(wa),pinv(wb),pinv(wc)
    @tensor Ba[:] := Ba[1,-2,-3]*ia[1,-1]
    @tensor Bb[:] := Bb[1,-2,-3]*ib[1,-1]
    @tensor Bc[:] := Bc[1,-2,-3]*ic[1,-1]
    return Ba,Bb,Bc,la/norm(la),lb/norm(lb),lc/norm(lc),core/norm(core)
end

function _up_sweep!(state,gate;Dmax,trun_tol)
    for (r,c) in enumerate(state.raw.cells)
        fn = r == state.anchor ? _charged_up : Kernel.Tri_T_up
        result = r == state.anchor ?
            fn(c.T_u,c.B_a,c.B_b,c.B_c,c.lambda_d_a,c.lambda_d_b,c.lambda_d_c,
               gate,trun_tol,Dmax) :
            fn(c.T_u,c.B_a,c.B_b,c.B_c,c.lambda_d_a,c.lambda_d_b,c.lambda_d_c,
               gate,trun_tol,Dmax,false)
        c.B_a,c.B_b,c.B_c,c.lambda_u_a,c.lambda_u_b,c.lambda_u_c,c.T_u = result
    end
    return state
end

"""
Evolve with SU(2)-scalar exchange gates and the original boundary half-steps.
The auxiliary leg is retained in the charged up-triangle HOSVD. All other
updates use the existing kernels. The HOSVD norm sums the auxiliary multiplet;
it is NOT an environment optimized for the polarized projected state. This
is an exploratory SU approximation, not a variational optimization thereof.

A uniform h*sum(Sz) is constant AFTER projection into this fixed-charge
sector; add its contribution when measuring energies, not as a non-SU(2) gate.
"""
function itebd_charged(state::ChargedKagomeState,H,tau::Real,dt::Real;
                       Dmax::Integer,trun_tol::Real=1e-6,callback=nothing)
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be positive and finite"))
    isfinite(tau) && tau >= 0 || throw(ArgumentError("tau must be nonnegative and finite"))
    Dmax > 0 && isfinite(trun_tol) && 0 <= trun_tol < 1 || throw(ArgumentError("invalid truncation settings"))
    check_charged(state)
    result = deepcopy(state)
    gate,half = Kernel.trotter_gate(H,dt)
    BaseSU.sweep!(result.raw,half,:dn;Dmax,trun_tol)
    steps = round(Int,tau/dt)
    for step in 1:steps
        _up_sweep!(result,gate;Dmax,trun_tol)
        BaseSU.sweep!(result.raw,gate,:dn;Dmax,trun_tol)
        isnothing(callback) || callback(result,step,steps)
    end
    _up_sweep!(result,half;Dmax,trun_tol)
    return check_charged(result;Dmax)
end

"""
Dense physical PESS after fixing ONE component of the extra index.
TensorKit's nondual spin-j dense basis is m=j,j-1,...,-j: component=1
is the highest-weight ancilla and selects negative physical magnetization.
These projected tensors are U(1)-covariant, not SU(2)-invariant TensorMaps.
The original B and simplex virtual-axis convention is preserved exactly.
"""
function polarized_dense_cells(state::ChargedKagomeState;component::Integer=1)
    1 <= component <= 2state.auxiliary_spin+1 || throw(BoundsError())
    return [begin
        core = convert(Array,c.T_u)
        Tu = r == state.anchor ? copy(selectdim(core,4,component)) : core
        (B1=convert(Array,c.B_a),B2=convert(Array,c.B_b),B3=convert(Array,c.B_c),
         Tup=Tu,Tdn=convert(Array,c.T_d))
    end for (r,c) in enumerate(state.raw.cells)]
end

"""
Exact nine-physical-spin periodic torus with ONE auxiliary leg left open.
This is a small correctness benchmark, NOT infinite-lattice CTMRG. It is
restricted to actual D<=6 to avoid accidental large local contractions.
"""
function closed_sqrt3_wavefunction(state::ChargedKagomeState)
    check_charged(state)
    maximum(max(b.down_dimension,b.up_dimension) for b in BaseSU.bond_report(state.raw)) <= 6 ||
        throw(ArgumentError("exact torus helper is restricted to actual D<=6"))
    order = [mod1(state.anchor+k,3) for k in 0:2]
    c1,c2,c3 = state.raw.cells[order]
    a1,b1,d1,t1 = c1.B_a,c1.B_b,c1.B_c,c1.T_u
    a2,b2,d2,t2 = c2.B_a,c2.B_b,c2.B_c,c2.T_u
    a3,b3,d3,t3 = c3.B_a,c3.B_b,c3.B_c,c3.T_u
    @tensor f1[:] := t1[1,2,3,-7]*a1[-1,1,-4]*b1[-2,2,-5]*d1[-3,3,-6]
    @tensor f2[:] := t2[1,2,3]*a2[-1,1,-4]*b2[-2,2,-5]*d2[-3,3,-6]
    @tensor f3[:] := t3[1,2,3]*a3[-1,1,-4]*b3[-2,2,-5]*d3[-3,3,-6]
    td1,td2,td3 = c1.T_d,c2.T_d,c3.T_d
    @tensoropt psi[:] := f1[1,2,3,-1,-2,-3,-10]*f2[4,5,6,-4,-5,-6]*
        f3[7,8,9,-7,-8,-9]*td1[4,2,9]*td2[7,5,3]*td3[1,8,6]
    return convert(Array,psi)
end

function _torus_bonds()
    triangles = [(3r+1,3r+2,3r+3) for r in 0:2]
    append!(triangles,[(3mod(r+1,3)+1,3r+2,3mod(r-1,3)+3) for r in 0:2])
    return [(t[i],t[mod1(i+1,3)]) for t in triangles for i in 1:3]
end

"""Finite nine-site observables with H=J1*sum(S.S)+h*sum(Sz)."""
function exact_torus_observables(state::ChargedKagomeState;component::Integer=1,h::Real=0,J1::Real=1)
    tensor = closed_sqrt3_wavefunction(state)
    1 <= component <= size(tensor,10) || throw(BoundsError())
    psi = vec(copy(selectdim(tensor,10,component)))
    squared_norm = real(dot(psi,psi))
    squared_norm > 1e-28 || throw(ArgumentError("zero/near-zero projected finite-torus state"))
    psi /= sqrt(squared_norm)
    sites = zeros(9)
    second_moment = 0.0
    Hpsi = zeros(eltype(psi),512)
    plus,minus = similar(Hpsi),similar(Hpsi)
    fill!(plus,0);fill!(minus,0)
    bonds = _torus_bonds()
    for index in 0:511
        m = 4.5-count_ones(index)
        second_moment += abs2(psi[index+1])*m^2
        for site in 1:9
            bit = (index >> (site-1)) & 1
            sites[site] += abs2(psi[index+1])*(.5-bit)
            target = xor(index,1 << (site-1))+1
            (bit == 1 ? plus : minus)[target] += psi[index+1]
        end
        for (i,j) in bonds
            bi,bj = (index >> (i-1)) & 1,(index >> (j-1)) & 1
            Hpsi[index+1] += J1*(.5-bi)*(.5-bj)*psi[index+1]
            if bi != bj
                target = xor(index,(1 << (i-1)) | (1 << (j-1)))+1
                Hpsi[target] += .5J1*psi[index+1]
            end
        end
    end
    total_Sz = sum(sites)
    spin_squared = second_moment + .5*(real(dot(plus,plus))+real(dot(minus,minus)))
    exchange = real(dot(psi,Hpsi))
    return (projected_norm=squared_norm,total_Sz=total_Sz,
            variance_Sz=max(0.,second_moment-total_Sz^2),spin_squared=spin_squared,
            magnetization_normalized=-2total_Sz/9,site_Sz=sites,
            exchange_energy_per_site=exchange/9,
            field_energy_per_site=(exchange+h*total_Sz)/9,
            auxiliary_m=Float64(state.auxiliary_spin+1-component),
            torus_spins=9,measurement="exact finite periodic torus; not CTMRG")
end

function save_charged_checkpoint(path,state::ChargedKagomeState,H;metadata=Dict{String,Any}())
    check_charged(state)
    target = abspath(path);mkpath(dirname(target))
    temporary = tempname(dirname(target))
    try
        jldsave(temporary;format="kagome_su2_charged_sqrt3_v1",state,H,metadata)
        mv(temporary,target;force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    return target
end

function load_charged_checkpoint(path)
    data = JLD2.load(path)
    data["format"] == "kagome_su2_charged_sqrt3_v1" || throw(ArgumentError("unknown checkpoint format"))
    return check_charged(data["state"]),data
end

end # module
