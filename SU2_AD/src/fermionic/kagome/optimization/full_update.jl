function _triangle(factors)
    return KF._simplex_wavefunction(factors[1],factors[2:4])
end

function _fit_loss(factors,target,root)
    return norm(_weighted(root,_triangle(factors)-target))^2/
           max(norm(_weighted(root,target))^2,eps())
end

function _fidelity(factors,target,root)
    a,b = _weighted(root,_triangle(factors)),_weighted(root,target)
    return abs(dot(a,b))/max(norm(a)*norm(b),eps())
end

# One local factor appears LINEARLY in the wavefunction. Form its weighted
# design matrix in symmetry-block coordinates and solve by a regularized SVD.
# This is the four-factor ALS of Full_Update_lib.jl, retaining the simplex.
function _als_factor(factors,target,root,index;solve_rtol=1e-10,ridge=1e-12)
    template = factors[index]
    n = length(template.data)
    rhs = _coefficients(_weighted(root,target))
    design = Matrix{ComplexF64}(undef,length(rhs),n)
    for k in 1:n
        basis = zeros(ComplexF64,n)
        basis[k] = 1
        candidate = copy(factors)
        candidate[index] = _from_coefficients(basis,template)
        design[:,k] = _coefficients(_weighted(root,_triangle(candidate)))
    end
    sv = svd(design;full=false)
    maximum(sv.S;init=0.0)>0 || return factors
    cutoff = solve_rtol*maximum(sv.S)
    reg = ridge*maximum(sv.S)^2
    inverse = [s>cutoff ? s/(s*s+reg) : 0.0 for s in sv.S]
    coefficients = sv.V*(inverse.*(sv.U'*rhs))
    candidate = copy(factors)
    candidate[index] = _from_coefficients(coefficients,template)
    # Null spaces and finite regularization must not silently worsen the fit.
    return _fit_loss(candidate,target,root)<=_fit_loss(factors,target,root)+1e-13 ? candidate : factors
end

function _gauges(metric)
    dims = ntuple(i -> dim(space(metric,i)),3)
    dense = reshape(convert(Array,metric),dims...,dims...)
    return ntuple(3) do leg
        rho = zeros(ComplexF64,dims[leg],dims[leg])
        for a in CartesianIndices(dims),b in CartesianIndices(dims)
            all(a[k]==b[k] for k in 1:3 if k!=leg) || continue
            rho[a[leg],b[leg]] += dense[Tuple(a)...,Tuple(b)...]
        end
        V = space(metric,leg)
        R = TensorMap(rho,V,V)
        _,g,_ = _positive_root(R)
        g = g/norm(g)
        inverse = pinv(g;rtol=1e-12)
        (;g,inverse)
    end
end

function _initial_factors(target,Dmax,trun_tol;gauges=nothing)
    W = target
    if !isnothing(gauges)
        for k in 1:3
            W = KF._leg_map(W,gauges[k].g,2k-1)
        end
    end
    core,Us,_,_ = KF._hosvd(W,Dmax,trun_tol)
    if !isnothing(gauges)
        Us = [TensorKit.permute(KF._leg_map(Us[k],gauges[k].inverse,1),((1,2),(3,))) for k in 1:3]
    end
    return TensorMap[core,Us...]
end

"""
CTM-weighted fit of a gated triangle by T and B_a/B_b/B_c. Optimize all four
rank-3 factors; never optimize a free square iPEPS and refactor it afterwards.
Both direct-HOSVD and environment-gauged seeds are tried, as in triangle FU.
"""
function fit_triangle(target,environment;Dmax,n_sweep=10,trun_tol=1e-10,
                       solve_rtol=1e-10,ridge=1e-12,fit_tol=1e-9,use_gauge=true)
    Dmax>0 && n_sweep>=0 && ridge>=0 || throw(ArgumentError("invalid fit settings"))
    target = target/norm(target)
    gauges = use_gauge ? _gauges(environment.metric) : nothing
    seeds = use_gauge ? (nothing,gauges) : (nothing,)
    results = map(seeds) do gauge
        factors = _initial_factors(target,Dmax,trun_tol;gauges=gauge)
        # Optimize the overall scalar before comparing seed residuals. HOSVD
        # normalizes its core; whitening changes that scalar substantially.
        a,b = _weighted(environment.root,_triangle(factors)),_weighted(environment.root,target)
        factors[1] *= dot(a,b)/dot(a,a)
        losses = [_fit_loss(factors,target,environment.root)]
        for _ in 1:n_sweep
            # Three physical tensors, then the simplex, matching triangle FU.
            for index in (2,3,4,1)
                factors = _als_factor(factors,target,environment.root,index;solve_rtol,ridge)
            end
            # Balance scalar gauges WITHOUT changing the represented triangle.
            for index in 2:4
                scale = norm(factors[index])
                factors[index] /= scale
                factors[1] *= scale
            end
            loss = _fit_loss(factors,target,environment.root)
            push!(losses,loss)
            abs(losses[end-1]-loss)<=fit_tol*max(losses[1],1e-14) && break
        end
        (;factors,losses,fidelity=_fidelity(factors,target,environment.root),
          seed=isnothing(gauge) ? :direct : :environment_gauge)
    end
    best = results[argmax([r.fidelity for r in results])]
    return (;best...,candidates=map(r -> (;seed=r.seed,initial_loss=r.losses[1],final_loss=r.losses[end],fidelity=r.fidelity),results))
end

function _write_up!(state,x,y,factors)
    state.T_u[x,y] = KF._normalized(factors[1])
    for b in 1:3
        state.B[x,y,b] = KF._normalized(KF.fpermute(factors[b+1],(1,3,2)))
        V = space(state.B[x,y,b],2)
        state.lambda_u[x,y,b] = unitary(V,V)
    end
    return state
end

function _replace_state!(destination,source)
    for field in fieldnames(typeof(destination))
        getfield(destination,field) .= getfield(source,field)
    end
    return destination
end

function full_update_triangle!(state,x,y,orientation,gate;Dmax,chi=20,
                                ctm_iterations=50,ctm_tol=1e-7,projector_tol=1e-8,kwargs...)
    orientation in (:up,:dn) || throw(ArgumentError("orientation must be :up or :dn"))
    work = orientation==:up ? state : reverse_triangles(state)
    Lx,Ly = size(work.T_u)
    wx,wy = orientation==:up ? (x,y) : (mod1(2-x,Lx),mod1(2-y,Ly))
    environment = triangle_environment(work,wx,wy;chi,ctm_iterations,ctm_tol,projector_tol)
    target = KF._apply_gate(environment.W0,gate)
    fit = fit_triangle(target,environment;Dmax,kwargs...)
    _write_up!(work,wx,wy,fit.factors)
    orientation==:dn && _replace_state!(state,reverse_triangles(work))
    check_state(state)
    return (;fit,environment=environment.diagnostics)
end

"""Second-order simplex Full Update, rebuilding the CTM metric after each triangle."""
function full_update!(state,parameters,tau,dt,Dmax;chi=20,ctm_iterations=50,
                       ctm_tol=1e-7,callback=nothing,kwargs...)
    isfinite(tau) && tau>=0 && isfinite(dt) && dt>0 || throw(ArgumentError("invalid imaginary time"))
    steps = round(Int,tau/dt)
    isapprox(steps*dt,tau;atol=1e-12,rtol=1e-10) || throw(ArgumentError("tau must be a multiple of dt"))
    gu,_ = trotter_gate(Hamiltonian_triangle(parameters;orientation=:up),dt)
    _,gd = trotter_gate(Hamiltonian_triangle(parameters;orientation=:dn),dt)
    Lx,Ly = size(state.T_u)
    history = NamedTuple[]
    for step in 1:steps
        for (orientation,gate) in ((:dn,gd),(:up,gu),(:dn,gd))
            for y in 1:Ly,x in 1:Lx
                result = full_update_triangle!(state,x,y,orientation,gate;Dmax,chi,ctm_iterations,ctm_tol,kwargs...)
                record = (;step,tau=step*dt,x,y,orientation,fidelity=result.fit.fidelity,
                           loss=result.fit.losses[end],seed=result.fit.seed,environment=result.environment)
                push!(history,record)
                isnothing(callback) || callback(state,record)
            end
        end
    end
    return history
end
