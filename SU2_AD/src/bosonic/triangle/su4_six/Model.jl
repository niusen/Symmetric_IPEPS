# SciPost Phys. 8, 076 (2020), Eqs. (2)-(5),(11)-(12):
# h_ij/J = sum_ab (c†_a c_b - δ_ab/2)_i (c†_b c_a - δ_ab/2)_j
#        = swap_6 - 6 P_singlet.  Eigenvalues: -5 (1), -1 (15), +1 (20').
const OCCUPATIONS = ((1,2), (1,3), (1,4), (2,3), (2,4), (3,4))
physical_su4_space() = Rep[SU{4}]((1,1,0,0)=>1)
default_virtual_space() = Rep[SU{4}]((0,0,0,0)=>1, (1,1,0,0)=>1)

function occupation_operators()
    masks = [sum(1 << (a-1) for a in pair) for pair in OCCUPATIONS]
    E = [zeros(ComplexF64,6,6) for a=1:4,b=1:4]
    for a=1:4,b=1:4,k=1:6
        n=masks[k]; bitb=1 << (b-1); bita=1 << (a-1)
        iszero(n & bitb) && continue
        signb=(-1)^count_ones(n & (bitb-1)); m=n ⊻ bitb
        !iszero(m & bita) && continue
        signa=(-1)^count_ones(m & (bita-1)); m |= bita
        E[a,b][findfirst(==(m),masks),k]=signa*signb
    end
    C = [E[a,b] - (a==b ? Matrix{ComplexF64}(I,6,6)/2 : zeros(6,6)) for a=1:4,b=1:4]
    H=zeros(ComplexF64,36,36)
    for a=1:4,b=1:4
        H += kron(C[b,a], C[a,b]) # first physical index varies fastest
    end
    swap=zeros(ComplexF64,36,36)
    for a=1:6,b=1:6
        swap[a+6*(b-1),b+6*(a-1)]=1
    end
    singlet=(swap-H)/6
    return (; H, swap, singlet, E, C)
end

# Some TensorKit 0.17 development checkouts define the expression helper after
# their generated repartition method. Julia 1.11 then reports a world-age error.
# Refresh only that method, with its original body, only if this exact bug occurs.
function ensure_repartition_compatibility!()
    V=physical_su4_space()
    try
        permute(id(V*V),(1,),(2,3,4,))
    catch err
        if err isa MethodError && isdefined(TensorKit,:_repartition_body) &&
            err.f === getfield(TensorKit,:_repartition_body)
            @warn "Refreshing TensorKit repartition generator after its helper definition (Julia world-age compatibility)"
            @eval TensorKit @generated function repartition(src::Union{FusionTreePair,FusionTreeBlock}, ::Val{N}) where {N}
                return _repartition_body(numout(src)-N)
            end
        else
            rethrow()
        end
    end
    return nothing
end

"""Columns map TensorKit's six-dimensional irrep basis to |12>,...,|34>."""
function occupation_rotation()
    V4=Rep[SU{4}]((1,0,0,0)=>1); V6=physical_su4_space()
    embedding=TensorMap{ComplexF64}(undef,V4*V4,V6)
    for (_,b) in blocks(embedding)
        b .= 1
    end
    gt=reshape(convert(Array,embedding),16,6)
    wedge=zeros(ComplexF64,16,6)
    for (k,(a,b)) in enumerate(OCCUPATIONS)
        wedge[a+4*(b-1),k]=1/sqrt(2)
        wedge[b+4*(a-1),k]=-1/sqrt(2)
    end
    R=wedge'*gt
    isapprox(R'*R,Matrix{ComplexF64}(I,6,6);atol=1e-12) || error("Nonunitary six-irrep basis map")
    return R
end

"""Both backends use h/J=swap-6P1, with no additive energy shift.
Dense physical basis: occupations. SU(4) backend: TensorKit irrep (1,1,0,0).
C/E are occupation-basis matrices in both backends; rotation maps irrep-basis
vectors to occupations (use rotation' * C[a,b] * rotation for an irrep matrix).
"""
function six_model(; backend=:dense, J=1.0)
    backend in (:dense,:su4) || throw(ArgumentError("backend must be :dense or :su4"))
    isfinite(J) || throw(ArgumentError("J must be finite"))
    local_ops=occupation_operators()
    if backend==:dense
        V=ComplexSpace(6)
        h=TensorMap(reshape(local_ops.H,6,6,6,6),V*V,V*V)
        swap=TensorMap(reshape(local_ops.swap,6,6,6,6),V*V,V*V)
        R=Matrix{ComplexF64}(I,6,6)
    else
        ensure_repartition_compatibility!()
        V=physical_su4_space()
        h=1.0*id(V*V); swap=1.0*id(V*V)
        for (sector,b) in blocks(h)
            d=dim(sector)
            d in (1,15,20) || error("Unexpected 6 x 6 fusion channel: $sector")
            b .= d==1 ? -5.0 : d==15 ? -1.0 : 1.0
            block(swap,sector) .= d==15 ? -1.0 : 1.0
        end
        R=occupation_rotation()
        U=kron(R,R)
        isapprox(U*reshape(convert(Array,h),36,36)*U',local_ops.H;atol=1e-11) ||
            error("SU(4) and occupation-basis Hamiltonians disagree")
    end
    p1=(swap-h)/6
    return (;backend,J=Float64(J),V,h,swap,singlet=p1,
        C=local_ops.C,E=local_ops.E,rotation=R,
        basis=backend==:dense ? :occupation : :gelfand_tsetlin)
end

function setup_cell(nx,ny)
    (nx,ny) in ((2,2),(4,4)) || throw(ArgumentError("original CTMRG supports 2x2/4x4 here"))
    global Lx=nx
    global Ly=ny
    return nothing
end

function initial_state(;backend=:dense,D=backend==:su4 ? 7 : 4,cell=(2,2),seed=1234,virtual_space=nothing)
    setup_cell(cell...)
    model=six_model(;backend)
    V=backend==:dense ? ComplexSpace(D) : isnothing(virtual_space) ? default_virtual_space() : virtual_space
    backend==:dense || sectortype(V)==SU4Irrep || throw(ArgumentError("virtual_space must carry SU(4)"))
    dim(V)<=D || throw(ArgumentError("initial virtual space exceeds D budget; the default SU(4) space needs D>=7"))
    Random.seed!(seed)
    B,T,l1,l2,l3=initial_iPESS(cell...,V,model.V)
    all(t->isfinite(norm(t)) && norm(t)>0,(B...,T...)) ||
        error("Virtual sectors cannot support nonzero simplex/site invariant tensors")
    return (;B,T,lambda1=l1,lambda2=l2,lambda3=l3)
end

"""Exact product of horizontal SU(4) singlet dimers; useful for normalization tests."""
function initial_dimer_state(;backend=:su4,cell=(2,2))
    setup_cell(cell...)
    sm=six_model(backend=:su4)
    one=Rep[SU{4}]((0,0,0,0)=>1); six=sm.V
    B=Matrix{Any}(undef,cell...); T=similar(B)
    for x=1:Lx,y=1:Ly
        middle=iseven(x) ? six : one
        right=isodd(x) ? six : one
        B[x,y]=TensorMap{ComplexF64}(undef,middle*one,middle')
        fill!(B[x,y],1)
        B[x,y] /= norm(B[x,y])
        t=TensorMap{ComplexF64}(undef,middle'*six,right*one)
        fill!(t,1)
        T[x,y]=permute(t,(1,),(2,3,4,))/norm(t)
    end
    if backend==:dense
        function as_dense(t;rotate=false)
            nout=length(codomain(t)); n=Rank(t)
            unfolded=permute(t,Tuple(1:n),())
            data=convert(Array,unfolded)
            if rotate
                @tensor rotated[-1,-2,-3,-4] := data[-1,1,-3,-4]*sm.rotation[-2,1]
                data=rotated
            end
            spaces=map(1:n) do k
                v=space(unfolded,k); w=ComplexSpace(dim(v))
                isdual(v) ? w' : w
            end
            dense=TensorMap(data,reduce(*,spaces),Base.one(first(spaces)))
            return permute(dense,Tuple(1:nout),Tuple(nout+1:n))
        end
        for x=1:Lx,y=1:Ly
            B[x,y]=as_dense(B[x,y])
            T[x,y]=as_dense(T[x,y];rotate=true)
        end
    elseif backend!=:su4
        throw(ArgumentError("unknown backend"))
    end
    # Match lambda1/2/3 to the three simplex legs (D1,R2,M3).
    diagonal(V)=DiagonalTensorMap(ones(V isa ComplexSpace ? dim(V) : sum(V.dims.values)),V)
    return (;B,T,lambda1=map(b->diagonal(space(b,2)'),B),
        lambda2=map(b->diagonal(space(b,1)'),B),lambda3=map(b->diagonal(space(b,3)'),B))
end
