# Same physical basis as Hamiltonians_spinful_Z2: |0>, |up dn>, |up>, |dn>.
physical_space() = Rep[ℤ₂](0=>2, 1=>2)

function _local_operators()
    c_up = zeros(ComplexF64,4,4)
    c_dn = zeros(ComplexF64,4,4)
    c_up[1,3] = c_up[4,2] = 1
    c_dn[1,4] = 1
    c_dn[3,2] = -1
    n_up, n_dn = c_up'*c_up, c_dn'*c_dn
    id = Matrix{ComplexF64}(I,4,4)
    return (; id, parity=diagm(ComplexF64[1,1,-1,-1]), c=(c_up,c_dn),
             pair=c_dn*c_up,
             n=n_up+n_dn, doublon=n_up*n_dn, interaction=(n_up-id/2)*(n_dn-id/2))
end

# Julia tensor axis 1 is fastest. Reverse the kron factors to preserve the
# physical axis order (a,b,c), with Jordan-Wigner strings on earlier sites.
function _embed(O, site; odd=false)
    ops = _local_operators()
    factors = [j == site ? O : (odd && j < site ? ops.parity : ops.id) for j in 1:3]
    return kron(factors[3], factors[2], factors[1])
end

function _hoppings(parameters, orientation)
    orientation in (:up,:dn) || throw(ArgumentError("orientation must be :up or :dn"))
    t = get(parameters, orientation == :up ? "t_up" : "t_dn", get(parameters,"t",1.0))
    ts = t isa Number ? (t,t,t) : Tuple(t)
    length(ts) == 3 || throw(ArgumentError("hoppings must be scalar or (t_ab,t_bc,t_ca)"))
    all(isfinite,ts) || throw(ArgumentError("nonfinite hopping"))
    return ts
end

function _triangle_operators()
    ops = _local_operators()
    cs = [[_embed(ops.c[spin],site; odd=true) for spin in 1:2] for site in 1:3]
    hops = [sum(cs[i][s]'*cs[j][s] for s in 1:2) for (i,j) in ((1,2),(2,3),(3,1))]
    return (; c=cs, hops, n=[_embed(ops.n,i) for i in 1:3],
             pair=[_embed(ops.pair,i) for i in 1:3],
             doublon=[_embed(ops.doublon,i) for i in 1:3],
             interaction=[_embed(ops.interaction,i) for i in 1:3])
end

function _physical_map(M)
    P = physical_space()
    return TensorMap(reshape(M,4,4,4,4,4,4), P*P*P, P*P*P)
end

"""
Triangle Hubbard Hamiltonian, ordered (a,b,c). Directed hoppings are
`-t_ab c_a†c_b -t_bc c_b†c_c -t_ca c_c†c_a + h.c.` (sum over spin).
Each kagome site belongs to two triangles: each gate gets HALF the onsite
term `U(n_up-1/2)(n_dn-1/2) - μ n + Δ c_up† c_dn† + conj(Δ) c_dn c_up`.
`parameters["pairing"]` is Δ (default zero); it may be complex.
Set centered=false for U n_up n_dn.
"""
function Hamiltonian_triangle(parameters; orientation=:up, onsite_fraction=0.5)
    U, μ = get(parameters,"U",0.0), get(parameters,"μ",0.0)
    Δ = get(parameters,"pairing",0.0)
    Δ isa Number && isfinite(Δ) || throw(ArgumentError("pairing must be a finite scalar"))
    isreal(U) && isreal(μ) && isfinite(U) && isfinite(μ) ||
        throw(ArgumentError("U and μ must be finite real numbers"))
    ops = _triangle_operators()
    ts = _hoppings(parameters,orientation)
    H = zeros(ComplexF64,64,64)
    for k in 1:3
        H -= ts[k]*ops.hops[k] + conj(ts[k])*ops.hops[k]'
        interaction = get(parameters,"centered",true) ? ops.interaction[k] : ops.doublon[k]
        H += onsite_fraction*(U*interaction-μ*ops.n[k])
        iszero(Δ) || (H += onsite_fraction*(Δ*ops.pair[k]'+conj(Δ)*ops.pair[k]))
    end
    return _physical_map(H)
end

function trotter_gate(H, dt)
    isfinite(dt) && dt >= 0 || throw(ArgumentError("dt must be finite and nonnegative"))
    dense = convert(Array,H)
    M = reshape(dense,dim(codomain(H)),dim(domain(H)))
    isapprox(M,M'; atol=1e-12) || throw(ArgumentError("Hamiltonian must be Hermitian"))
    e = eigen(Hermitian(M))
    # An overall scalar is discarded in normalized imaginary-time evolution.
    function gate(step)
        dense_gate = e.vectors*Diagonal(exp.(-step*(e.values .- minimum(e.values))))*e.vectors'
        TensorMap(reshape(dense_gate,size(dense)),codomain(H),domain(H))
    end
    return gate(dt), gate(dt/2)
end
