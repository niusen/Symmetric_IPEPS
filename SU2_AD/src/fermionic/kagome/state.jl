"""
Periodic kagome state. Each (x,y) owns Tup, Tdn and B_a/B_b/B_c.
B axes: (down,up,physical); simplex axes: (a,b,c), all outgoing.
The lambdas belong to the physical B tensors, not to simplex coordinates.
"""
struct KagomeFermionicState
    T_u::Matrix{TensorMap}
    T_d::Matrix{TensorMap}
    B::Array{TensorMap,3}
    lambda_u::Array{TensorMap,3}
    lambda_d::Array{TensorMap,3}
end

function triangle_sites(state, x, y, orientation)
    Lx,Ly = size(state.T_u)
    x,y = mod1(x,Lx),mod1(y,Ly)
    orientation == :up && return ((x,y,1),(x,y,2),(x,y,3))
    orientation == :dn && return ((mod1(x+1,Lx),y,1),(x,y,2),(x,mod1(y+1,Ly),3))
    throw(ArgumentError("orientation must be :up or :dn"))
end

function initial_iPESS(Lx::Integer=1,Ly::Integer=1,V=Rep[ℤ₂](0=>1,1=>1);
                       rng=Random.default_rng())
    Lx > 0 && Ly > 0 || throw(ArgumentError("positive cell sizes required"))
    TensorKit.sectortype(V) == Z2Irrep || throw(ArgumentError("only Z2 is supported"))
    P = physical_space()
    T_u,T_d = (Matrix{TensorMap}(undef,Lx,Ly) for _ in 1:2)
    B,lambda_u,lambda_d = (Array{TensorMap}(undef,Lx,Ly,3) for _ in 1:3)
    for x in 1:Lx, y in 1:Ly
        T_u[x,y] = _normalized(_allout(randn(rng,ComplexF64,V,V'*V')))
        T_d[x,y] = _normalized(_allout(randn(rng,ComplexF64,V,V'*V')))
        for b in 1:3
            B[x,y,b] = _normalized(_allout(randn(rng,ComplexF64,V'*V',P')))
            lambda_u[x,y,b] = unitary(V',V')
            lambda_d[x,y,b] = unitary(V',V')
        end
    end
    return check_state(KagomeFermionicState(T_u,T_d,B,lambda_u,lambda_d))
end

function check_state(state::KagomeFermionicState; Dmax=nothing)
    Lx,Ly = size(state.T_u)
    size(state.T_d) == (Lx,Ly) || throw(DimensionMismatch("simplex cell sizes"))
    for A in (state.B,state.lambda_u,state.lambda_d)
        size(A) == (Lx,Ly,3) || throw(DimensionMismatch("site array size"))
    end
    Lx > 0 && Ly > 0 || throw(ArgumentError("empty cell"))
    for x in 1:Lx, y in 1:Ly
        for orientation in (:up,:dn)
            T = (orientation == :up ? state.T_u : state.T_d)[x,y]
            _rank(T) == 3 || throw(DimensionMismatch("simplex rank must be 3"))
            isfinite(norm(T)) && norm(T)>0 || throw(ArgumentError("invalid simplex"))
            leg = orientation == :up ? 2 : 1
            weights = orientation == :up ? state.lambda_u : state.lambda_d
            for (axis,site) in enumerate(triangle_sites(state,x,y,orientation))
                B = state.B[site...]
                _rank(B) == 3 || throw(DimensionMismatch("B rank must be 3"))
                space(B,3) == physical_space() || throw(SpaceMismatch("physical Z2 space"))
                isfinite(norm(B)) && norm(B)>0 || throw(ArgumentError("invalid B"))
                V = space(B,leg)
                TensorKit.sectortype(V) == Z2Irrep || throw(ArgumentError("only Z2 is supported"))
                space(T,axis) == V' || throw(SpaceMismatch("simplex/B mismatch at $site"))
                isnothing(Dmax) || dim(V)<=Dmax || throw(DimensionMismatch("Dmax exceeded"))
                λ = weights[site...]
                space(λ,1) == V && space(λ,2) == V' || throw(SpaceMismatch("lambda/B mismatch"))
                M = convert(Array,λ)
                isfinite(norm(M)) && norm(M)>0 &&
                    isapprox(M,Diagonal(diag(M));atol=1e-12) &&
                    all(z -> isreal(z) && real(z)>=0,diag(M)) ||
                    throw(ArgumentError("weights must be nonnegative diagonal matrices"))
            end
        end
    end
    return state
end

function save_checkpoint(path,state; parameters=Dict(), metadata=Dict())
    check_state(state)
    mkpath(dirname(abspath(path)))
    # Save arrays under ordinary JLD2 keys so they can also be inspected without
    # deserializing the state struct. All six lambda sets are retained.
    jldsave(path; format="kagome_fermionic_z2_v1", T_u=state.T_u,T_d=state.T_d,
            B=state.B,lambda_u=state.lambda_u,lambda_d=state.lambda_d,parameters,metadata)
    return path
end

function load_checkpoint(path)
    data = load(path)
    get(data,"format",nothing) == "kagome_fermionic_z2_v1" ||
        throw(ArgumentError("not a kagome fermionic Z2 checkpoint"))
    state = KagomeFermionicState(data["T_u"],data["T_d"],data["B"],
                                 data["lambda_u"],data["lambda_d"])
    return check_state(state), data["parameters"], data["metadata"]
end
