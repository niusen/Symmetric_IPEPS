"""
SU(2) kagome Simple Update with rectangular or sqrt(3) x sqrt(3) periodicity.

Only geometry/storage are new: the local updates, HOSVD, multiplet truncation,
and Trotter sequence are included from the existing implementation unchanged.
"""
module KagomeSU2Cells

using TensorKit, LinearAlgebra, Random, JLD2

export KagomeSUCellGeometry, KagomeSUTensors, KagomeSUCell,
       representative, triangle_sites, initialize_cells, check_state,
       sweep!, itebd_cells, heisenberg_triangle, to_ipess_cells,
       save_checkpoint, load_checkpoint, bond_report, su_energy_estimates

# Isolate legacy names (including its other `itebd` methods) from the caller.
module Legacy
include(joinpath(@__DIR__, "simple_update_compat.jl"))
include(joinpath(@__DIR__, "..", "..", "fermionic", "simple_update",
                 "fermi_triangle_SimpleUpdate.jl"))
include(joinpath(@__DIR__, "funs_1up_1down.jl"))
include(joinpath(@__DIR__, "..", "kagome_model.jl"))
end

"""
Periodic quotient of the original Julia kagome coordinates (x right, y down).
`rectangular` has Lx*Ly independent PESS; `sqrt3xsqrt3` uses a 3x3 display
grid with exactly THREE independent PESS, color = 1 + mod(x-y,3).
The latter is invariant under translations (1,1) and (-1,2).
"""
struct KagomeSUCellGeometry
    Lx::Int
    Ly::Int
    pattern::Symbol
    site_map::Matrix{Int}
    representatives::Vector{Tuple{Int,Int}}
end

function KagomeSUCellGeometry(Lx::Integer=1, Ly::Integer=1;
                             pattern::Symbol=:rectangular)
    Lx > 0 && Ly > 0 || throw(ArgumentError("Lx and Ly must be positive"))
    pattern in (:rectangular, :sqrt3xsqrt3) ||
        throw(ArgumentError("pattern must be :rectangular or :sqrt3xsqrt3"))
    if pattern == :sqrt3xsqrt3
        (Lx, Ly) == (3, 3) || throw(ArgumentError("sqrt3xsqrt3 uses a 3x3 display grid"))
        mapping = [1 + mod(x-y, 3) for x in 1:3, y in 1:3]
        reps = [(1,1), (2,1), (3,1)]
    else
        mapping = reshape(collect(1:Lx*Ly), Lx, Ly)
        reps = [(x,y) for y in 1:Ly for x in 1:Lx]
    end
    geometry = KagomeSUCellGeometry(Int(Lx), Int(Ly), pattern, mapping, reps)
    # Each up/down layer must partition all physical spins of the quotient.
    expected = Set((r,b) for r in eachindex(reps) for b in 1:3)
    for orientation in (:up, :dn)
        sites = [site for r in eachindex(reps)
                 for site in triangle_sites(geometry, r, orientation)]
        length(sites) == length(expected) && Set(sites) == expected ||
            throw(ArgumentError("periodic triangle layer does not partition the sites"))
    end
    return geometry
end

representative(g::KagomeSUCellGeometry, x::Integer, y::Integer) =
    g.site_map[mod1(x,g.Lx), mod1(y,g.Ly)]

"""
Return (cell,bond) for a triangle, in the ORIGINAL simplex virtual-axis order.
Up: [B1(x,y), B2(x,y), B3(x,y)].
Down: [B1(x+1,y), B2(x,y), B3(x,y+1)].
This is exactly build_A_kagome / build_special_PEPS; it is not a relabeling.
"""
function triangle_sites(g::KagomeSUCellGeometry, r::Integer, orientation::Symbol)
    x,y = g.representatives[r]
    orientation == :up && return ((r,1), (r,2), (r,3))
    orientation == :dn && return ((representative(g,x+1,y),1), (r,2),
                                 (representative(g,x,y+1),3))
    throw(ArgumentError("orientation must be :up or :dn"))
end

# A bond's two lambdas live at the SAME coordinate as its physical B tensor.
# They may change SU(2) content/total dimension independently under truncation.
# `Any` permits those TensorMap type/rank changes without padding or projection.
mutable struct KagomeSUTensors
    T_u::Any
    T_d::Any
    B_a::Any
    B_b::Any
    B_c::Any
    lambda_u_a::Any
    lambda_u_b::Any
    lambda_u_c::Any
    lambda_d_a::Any
    lambda_d_b::Any
    lambda_d_c::Any
end

struct KagomeSUCell
    geometry::KagomeSUCellGeometry
    cells::Vector{KagomeSUTensors}
end

const B_FIELDS = (:B_a, :B_b, :B_c)
const U_FIELDS = (:lambda_u_a, :lambda_u_b, :lambda_u_c)
const D_FIELDS = (:lambda_d_a, :lambda_d_b, :lambda_d_c)
const CHECKPOINT_FORMAT = "kagome_su2_cells_v1"

function _random_cell(Vv, rng; complex=false)
    Vp = SU2Space(1/2=>1)
    scalar_type = complex ? ComplexF64 : Float64
    random_tensor(out,incoming) = randn(rng,scalar_type,out,incoming)
    # Same spaces/duals and leg permutations as initial_SU2_state in
    # kagome_AD_SU2.jl. B axes are always (down,up,physical).
    bonds = [permute(random_tensor(Vv'*Vv',Vp'), ((1,2,3), ())) for _ in 1:3]
    triangles = [permute(random_tensor(Vv,Vv'*Vv'), ((1,2,3), ())) for _ in 1:2]
    for tensor in vcat(bonds, triangles)
        iszero(norm(tensor)) && throw(ArgumentError(
            "chosen virtual SU(2) space admits a zero B or simplex tensor"))
    end
    bonds = [tensor/norm(tensor) for tensor in bonds]
    triangles = [tensor/norm(tensor) for tensor in triangles]
    weights = [unitary(space(b,2)',space(b,2)') for b in bonds]
    return KagomeSUTensors(triangles...,bonds...,weights...,nothing,nothing,nothing)
end

"""
Initialize each independent representative separately, or repeat `seed_cell`.
No spatial/rotation symmetry, averaging, or perturbation is imposed implicitly.
Supply a seeded RNG for reproducibility. `seed_cell` retains its raw lambdas.
"""
function initialize_cells(g::KagomeSUCellGeometry, Vv=SU2Space(0=>1,1/2=>1);
                          rng=Random.default_rng(), complex=false, seed_cell=nothing)
    cells = [isnothing(seed_cell) ? _random_cell(Vv,rng; complex) : deepcopy(seed_cell)
             for _ in g.representatives]
    state = KagomeSUCell(g,cells)
    check_state(state)
    return state
end

function check_state(state::KagomeSUCell; Dmax=nothing)
    g = state.geometry
    length(state.cells) == length(g.representatives) ||
        throw(DimensionMismatch("one raw tensor set is required per representative"))
    for (r,c) in enumerate(state.cells)
        for name in (:T_u,:T_d,B_FIELDS...)
            tensor = getfield(c,name)
            isfinite(norm(tensor)) && norm(tensor) > 0 ||
                throw(ArgumentError("nonfinite/zero tensor $name in cell $r"))
            TensorKit.sectortype(space(tensor,1)) == SU2Irrep ||
                throw(ArgumentError("this module requires SU(2) TensorMaps"))
        end
        for orientation in (:up,:dn)
            triangle = orientation == :up ? c.T_u : c.T_d
            leg = orientation == :up ? 2 : 1
            for (axis,(owner,b)) in enumerate(triangle_sites(g,r,orientation))
                bond = getfield(state.cells[owner],B_FIELDS[b])
                space(triangle,axis) == space(bond,leg)' ||
                    throw(SpaceMismatch("simplex/B mismatch at $orientation triangle $r axis $axis"))
                isnothing(Dmax) || dim(space(triangle,axis)) <= Dmax ||
                    throw(DimensionMismatch("actual virtual dimension exceeds Dmax"))
            end
        end
        for b in 1:3, (fields,leg) in ((U_FIELDS,2),(D_FIELDS,1))
            weight = getfield(c,fields[b])
            isnothing(weight) && continue # down lambdas initialize on first dn sweep
            isfinite(norm(weight)) && norm(weight) > 0 ||
                throw(ArgumentError("nonfinite/zero bond weight"))
            V = space(getfield(c,B_FIELDS[b]),leg)'
            space(weight,1) == V && space(weight,2) == V' ||
                throw(SpaceMismatch("lambda/B mismatch in cell $r"))
        end
    end
    return state
end

"""Update EACH inequivalent triangle once using the unmodified legacy kernel."""
function sweep!(state::KagomeSUCell, gate, orientation::Symbol;
                Dmax::Integer, trun_tol::Real=1e-6)
    orientation in (:up,:dn) || throw(ArgumentError("orientation must be :up or :dn"))
    opposite = orientation == :dn ? U_FIELDS : D_FIELDS
    current = orientation == :dn ? D_FIELDS : U_FIELDS
    kernel = orientation == :dn ? Legacy.Tri_T_dn : Legacy.Tri_T_up
    simplex_field = orientation == :dn ? :T_d : :T_u
    for r in eachindex(state.cells)
        sites = triangle_sites(state.geometry,r,orientation)
        bonds = [getfield(state.cells[owner],B_FIELDS[b]) for (owner,b) in sites]
        weights = [getfield(state.cells[owner],opposite[b]) for (owner,b) in sites]
        any(isnothing,weights) && throw(ArgumentError(
            "down lambdas are uninitialized; perform the initial dn layer first"))
        # Rotation-symmetric HOSVD would identify inequivalent bonds and is
        # intentionally not used for either cell. General HOSVD is unchanged.
        result = kernel(getfield(state.cells[r],simplex_field),bonds...,weights...,
                        gate,trun_tol,Dmax,false)
        for (i,(owner,b)) in enumerate(sites)
            setfield!(state.cells[owner],B_FIELDS[b],result[i])
            setfield!(state.cells[owner],current[b],result[3+i])
        end
        setfield!(state.cells[r],simplex_field,result[7])
    end
    return state
end

"""
    itebd_cells(state, H, tau, dt; Dmax, trun_tol=1e-6, H_down=H, callback=nothing)

Return a new state; preserve the original sequence exactly:
dn/half, (up,dn)*round(tau/dt), up/half. Each layer has 3 triangle updates
for sqrt3xsqrt3, 9 for 3x3, never 9 independent updates of a tied sqrt3 cell.
H/H_down physical order must match triangle_sites; scalar Heisenberg exchange
is invariant under this ordering. A callback receives (state,step,total_steps)
after full up+dn pairs; stages do not stop early or call CTMRG.
"""
function itebd_cells(state::KagomeSUCell, H, tau::Real, dt::Real;
                     Dmax::Integer, trun_tol::Real=1e-6, H_down=H, callback=nothing)
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    isfinite(tau) && tau >= 0 || throw(ArgumentError("tau must be finite and nonnegative"))
    Dmax >= 1 || throw(ArgumentError("Dmax must be positive"))
    isfinite(trun_tol) && 0 <= trun_tol < 1 || throw(ArgumentError("invalid trun_tol"))
    check_state(state)
    result = deepcopy(state)
    gate,half = Legacy.trotter_gate(H,dt)
    gate_dn,half_dn = H_down === H ? (gate,half) : Legacy.trotter_gate(H_down,dt)
    steps = Int(round(tau/dt))
    sweep!(result,half_dn,:dn;Dmax,trun_tol)
    for step in 1:steps
        sweep!(result,gate,:up;Dmax,trun_tol)
        sweep!(result,gate_dn,:dn;Dmax,trun_tol)
        isnothing(callback) || callback(result,step,steps)
    end
    sweep!(result,half,:up;Dmax,trun_tol)
    check_state(result;Dmax)
    return result
end

"""Spin-1/2 J1 Heisenberg triangle in exactly the legacy gate convention (h=0)."""
function heisenberg_triangle(;J1::Real=1.0)
    Vp = SU2Space(1/2=>1)
    U = unitary(fuse(Vp' ⊗ Vp' ⊗ Vp'),Vp' ⊗ Vp' ⊗ Vp')
    fused = first(Legacy.Hamiltonians(U,J1,0,0,0,0))
    @tensor H[:] := U[2,-1,-2,-3]*fused[1,2]*U'[-4,-5,-6,1]
    return permute(H,((1,2,3),(4,5,6)))
end

"""
Expand to an Lx x Ly matrix for the existing kagome cell CTMRG/AD interface.
Pass its constructor, e.g. to_ipess_cells(state,Kagome_iPESS). No lambda is
multiplied into the physical tensors: this follows the old SU export exactly.
Different grid entries are deep copies; sqrt3 parameter tying is carried by
`geometry.site_map`, and is NOT automatically enforced by a later optimizer.
"""
function to_ipess_cells(state::KagomeSUCell, constructor)
    g = state.geometry
    return [begin
        c = deepcopy(state.cells[representative(g,x,y)])
        constructor(c.B_a,c.B_b,c.B_c,c.T_u,c.T_d)
    end for x in 1:g.Lx, y in 1:g.Ly]
end

function bond_report(state::KagomeSUCell)
    return [(cell=r,bond=b,
             down_dimension=dim(space(getfield(c,B_FIELDS[b]),1)),
             up_dimension=dim(space(getfield(c,B_FIELDS[b]),2)),
             down_space=string(space(getfield(c,B_FIELDS[b]),1)),
             up_space=string(space(getfield(c,B_FIELDS[b]),2)))
            for (r,c) in enumerate(state.cells) for b in 1:3]
end

"""Local SU-environment energy estimates, NOT a CTMRG/variational energy."""
function su_energy_estimates(state::KagomeSUCell,H;H_down=H)
    values = Dict(:up=>Float64[], :dn=>Float64[])
    for orientation in (:up,:dn), r in eachindex(state.cells)
        c = state.cells[r]
        sites = triangle_sites(state.geometry,r,orientation)
        fields = orientation == :up ? D_FIELDS : U_FIELDS
        bonds = [getfield(state.cells[owner],B_FIELDS[b]) for (owner,b) in sites]
        weights = [getfield(state.cells[owner],fields[b]) for (owner,b) in sites]
        any(isnothing,weights) && throw(ArgumentError("run the initial dn layer first"))
        a,b,d = bonds
        wa,wb,wd = weights
        if orientation == :up
            @tensor a[:] := a[1,-2,-3]*wa[1,-1]
            @tensor b[:] := b[1,-2,-3]*wb[1,-1]
            @tensor d[:] := d[1,-2,-3]*wd[1,-1]
            @tensor A[:] := c.T_u[1,2,3]*a[-1,1,-2]*b[-3,2,-4]*d[-5,3,-6]
            op = H
        else
            @tensor a[:] := a[-1,1,-3]*wa[1,-2]
            @tensor b[:] := b[-1,1,-3]*wb[1,-2]
            @tensor d[:] := d[-1,1,-3]*wd[1,-2]
            @tensor A[:] := c.T_d[1,2,3]*a[1,-1,-2]*b[2,-3,-4]*d[3,-5,-6]
            op = H_down
        end
        @tensor acted[:] := op[-2,-4,-6,1,2,3]*A[-1,1,-3,2,-5,3]
        push!(values[orientation],real(dot(A,acted)/dot(A,A)))
    end
    return (up=values[:up],down=values[:dn],
            per_site=(sum(values[:up])+sum(values[:dn]))/(3length(state.cells)))
end

function save_checkpoint(path::AbstractString,state::KagomeSUCell,H;
                         H_down=H,metadata=Dict{String,Any}())
    check_state(state)
    target = abspath(path)
    mkpath(dirname(target))
    temporary = tempname(dirname(target))
    try
        jldsave(temporary;format=CHECKPOINT_FORMAT,state,H,H_down,metadata)
        mv(temporary,target;force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    return target
end

function load_checkpoint(path::AbstractString)
    data = JLD2.load(path)
    data["format"] == CHECKPOINT_FORMAT || throw(ArgumentError("unknown checkpoint format"))
    state = data["state"]
    # Reconstruct geometry so a corrupt saved map cannot silently alter neighbors.
    g = state.geometry
    expected = KagomeSUCellGeometry(g.Lx,g.Ly;pattern=g.pattern)
    g.site_map == expected.site_map && g.representatives == expected.representatives ||
        throw(ArgumentError("checkpoint geometry disagrees with its pattern"))
    check_state(state)
    return state,data
end

end # module
