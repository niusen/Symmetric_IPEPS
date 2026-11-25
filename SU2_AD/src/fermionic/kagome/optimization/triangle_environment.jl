"""
Exchange up/down triangles by reversing both Bravais axes and exchanging the
two B virtual legs WITH fermionic signs. This lets both orientations use the
same three-site hole in a square CTM patch. Applying twice is the identity.
"""
function reverse_triangles(state)
    Lx,Ly = size(state.T_u)
    Tu,Td = (Matrix{TensorMap}(undef,Lx,Ly) for _ in 1:2)
    B,lu,ld = (Array{TensorMap}(undef,Lx,Ly,3) for _ in 1:3)
    for y in 1:Ly,x in 1:Lx
        ox,oy = mod1(2-x,Lx),mod1(2-y,Ly)
        Tu[x,y],Td[x,y] = deepcopy(state.T_d[ox,oy]),deepcopy(state.T_u[ox,oy])
        for (b,site) in enumerate(triangle_sites(state,ox,oy,:dn))
            B[x,y,b] = KF.fpermute(state.B[site...],(2,1,3))
            lu[x,y,b],ld[x,y,b] = deepcopy(state.lambda_d[site...]),deepcopy(state.lambda_u[site...])
        end
    end
    return check_state(typeof(state)(Tu,Td,B,lu,ld))
end

_group(W) = KF.fpermute(W,(1,3,5,2,4,6))
_ungroup(W) = KF.fpermute(W,(1,4,2,5,3,6))
_weighted(root,W) = root*TensorKit.permute(_group(W),((1,2,3),(4,5,6)))

function _positive_root(E;rtol=1e-12)
    positive,root = zero(E),zero(E)
    negative,total = 0.0,0.0
    for (q,b) in blocks(E)
        ev = eigen(Hermitian((b+b')/2))
        negative += sum(abs,min.(ev.values,0))
        total += sum(abs,ev.values)
        cutoff = rtol*maximum(abs,ev.values;init=0.0)
        vals = [v>cutoff ? v : 0.0 for v in ev.values]
        block(positive,q) .= ev.vectors*Diagonal(vals)*ev.vectors'
        block(root,q) .= ev.vectors*Diagonal(sqrt.(vals))*ev.vectors'
    end
    norm(root)>0 || error("CTMRG triangle metric has no positive support")
    return positive,root,negative/max(total,eps())
end

"""
Build the full CTM metric on the THREE external simplex-cluster legs.
No product-lambda approximation is used. Matrix columns are obtained by
reverse contractions through the one-hole CTM overlap, batching independent
columns into distinct physical basis states of matching parity,
then checked against independent complete mixed-overlap contractions.
The physical identity is kept implicit; the metric has size D_external^3,
not 64*D_external^3. Hermitian/positive projection follows triangle FU.
"""
function triangle_environment(state,x,y;chi=20,ctm_iterations=50,ctm_tol=1e-7,
                               metric_tol=1e-8,psd_rtol=1e-12,projector_tol=1e-10)
    check_state(state)
    Lx,Ly = size(state.T_u)
    1<=x<=Lx && 1<=y<=Ly || throw(BoundsError(state.T_u,(x,y)))
    chi>0 && ctm_iterations>0 || throw(ArgumentError("positive CTMRG settings required"))
    0<projector_tol<1 || throw(ArgumentError("projector_tol must lie in (0,1)"))
    _configure(Lx,Ly,chi;trun_tol=projector_tol)
    settings = _settings(iterations=ctm_iterations,tol=ctm_tol)
    A = convert_iPESS_to_iPEPS(state)
    env = _environment(A,chi,settings)
    W0 = triangle_wavefunction(state,x,y,:up;weighted=false)
    U = _fusion(space(state.B[x,y,1],3))
    Td = state.T_d[x,y]
    function raw(w1,w2)
        a1,a2 = _block_triangle(w1,Td,U),_block_triangle(w2,Td,U)
        aa = CTM.build_double_layer_swap(a1',a2)[1]
        return CTM.ob_2x2(env.CTM,aa,env.AA[mod1(x+1,Lx)][y],
                           env.AA[x][mod1(y+1,Ly)],env.AA[mod1(x+1,Lx)][mod1(y+1,Ly)],x-1,y-1)
    end
    scale = raw(W0,W0)
    isfinite(scale) && abs(scale)>0 || error("zero or nonfinite CTM overlap")
    overlap(w1,w2) = raw(w1,w2)/scale
    grouped = _group(W0)
    V = ntuple(i -> space(grouped,i),3)
    dims = dim.(V)
    E = zeros(ComplexF64,prod(dims),prod(dims))
    even_dims = [dim(v,Z2Irrep(0)) for v in V]
    indices = collect(CartesianIndices(dims))[:]
    external_groups = [findall(index -> isodd(sum(index[k]>even_dims[k] for k in 1:3))==odd,indices) for odd in (false,true)]
    physical_spaces = ntuple(k -> space(grouped,k+3),3)
    physical_dims = dim.(physical_spaces)
    physical_even = map(v -> dim(v,Z2Irrep(0)),physical_spaces)
    physical_groups = [[Tuple(i) for i in CartesianIndices(physical_dims)
                        if isodd(sum(i[k]>physical_even[k] for k in 1:3))==odd] for odd in (false,true)]
    # An invariant W has no amplitudes in external sectors absent from the physical legs.
    supported = filter(q -> !isempty(physical_groups[q]),1:2)
    probes = maximum(cld(length(external_groups[q]),length(physical_groups[q])) for q in supported)
    for batch in 1:probes
        data = zeros(ComplexF64,dims...,physical_dims...)
        assignments = Tuple{Int,NTuple{3,Int}}[]
        for q in supported
            capacity = length(physical_groups[q])
            for slot in 1:capacity
                position = (batch-1)*capacity+slot
                position>length(external_groups[q]) && break
                column = external_groups[q][position]
                physical = physical_groups[q][slot]
                data[Tuple(indices[column])...,physical...] = 1
                push!(assignments,(column,physical))
            end
        end
        probe = _ungroup(TensorMap(data,codomain(grouped),domain(grouped)))
        response = Zygote.gradient(w -> real(overlap(w,probe)),zero(W0))[1]
        dense = convert(Array,_group(response))
        for (column,physical) in assignments
            E[:,column] = vec(dense[:,:,:,physical...])
        end
    end
    Vext = V[1]*V[2]*V[3]
    metric = TensorMap(reshape(E,dims...,dims...),Vext,Vext)
    # Verify that all physical states/parities share this external-leg metric.
    rng = MersenneTwister(741)
    extraction_error = 0.0
    for _ in 1:2
        wa = randn(rng,ComplexF64,codomain(W0),domain(W0))
        wb = randn(rng,ComplexF64,codomain(W0),domain(W0))
        ma = TensorKit.permute(_group(wa),((1,2,3),(4,5,6)))
        mb = TensorKit.permute(_group(wb),((1,2,3),(4,5,6)))
        exact = overlap(wa,wb)
        estimated = dot(ma,metric*mb)
        error = abs(exact-estimated)/max(sqrt(abs(overlap(wa,wa)*overlap(wb,wb))),eps())
        extraction_error = max(extraction_error,error)
    end
    extraction_error <= metric_tol || error("triangle CTM metric extraction failed: $extraction_error")
    hermitian_error = norm(metric-metric')/max(norm(metric),eps())
    positive,root,negative_weight = _positive_root(metric;rtol=psd_rtol)
    return (;metric=positive,root,raw_metric=metric,overlap,W0,
             diagnostics=(;iterations=env.iterations,ctm_error=env.error,
                           ctm_converged=env.error<ctm_tol,metric_probes=probes,extraction_error,hermitian_error,negative_weight))
end
