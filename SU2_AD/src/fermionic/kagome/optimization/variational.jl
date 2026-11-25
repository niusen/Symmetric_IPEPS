struct TensorLayout
    templates::Vector{TensorMap}
    ranges::Vector{UnitRange{Int}}
    Lx::Int
    Ly::Int
end

function pack_state(state)
    check_state(state)
    Lx,Ly = size(state.T_u)
    tensors = TensorMap[]
    for y in 1:Ly,x in 1:Lx
        append!(tensors,(state.T_u[x,y],state.T_d[x,y],state.B[x,y,1],state.B[x,y,2],state.B[x,y,3]))
    end
    sizes = length.(_coefficients.(tensors))
    offsets = cumsum(vcat(0,sizes))
    ranges = [offsets[i]+1:offsets[i+1] for i in eachindex(tensors)]
    return vcat(_coefficients.(tensors)...),TensorLayout(deepcopy(tensors),ranges,Lx,Ly)
end

function _tensors(v,layout)
    return map(eachindex(layout.templates)) do i
        _from_coefficients(v[layout.ranges[i]],layout.templates[i])
    end
end

function unpack_state(v,layout,state_type=KagomeFermionicState)
    tensors = _tensors(v,layout)
    Lx,Ly = layout.Lx,layout.Ly
    Tu,Td = (Matrix{TensorMap}(undef,Lx,Ly) for _ in 1:2)
    B,lu,ld = (Array{TensorMap}(undef,Lx,Ly,3) for _ in 1:3)
    for y in 1:Ly,x in 1:Lx
        k = 5*(x-1+Lx*(y-1))
        Tu[x,y],Td[x,y] = KF._normalized(tensors[k+1]),KF._normalized(tensors[k+2])
        for b in 1:3
            B[x,y,b] = KF._normalized(tensors[k+2+b])
            lu[x,y,b] = unitary(space(B[x,y,b],2),space(B[x,y,b],2))
            ld[x,y,b] = unitary(space(B[x,y,b],1),space(B[x,y,b],1))
        end
    end
    return check_state(state_type(Tu,Td,B,lu,ld))
end

abstract type AbstractVariationalProblem end

struct VariationalProblem <: AbstractVariationalProblem
    layout::TensorLayout
    U_phy::TensorMap
    operators::NamedTuple
    hoppings::Tuple
    chi::Int
    settings::Any
end

"""
Differentiate the energy through a FIXED number of complete CTMRG sweeps,
including the environment response. This is unrolled AD, not a frozen-CTM
gradient or implicit fixed-point derivative. Check both chi and iteration
convergence. Like the legacy CTMRG, calls must be serialized.
"""
function VariationalProblem(state,parameters;chi=20,ctm_iterations=20,projector_tol=1e-8)
    chi > 0 && ctm_iterations > 0 || throw(ArgumentError("positive CTMRG settings required"))
    0 < projector_tol < 1 || throw(ArgumentError("projector_tol must lie in (0,1)"))
    _,layout = pack_state(state)
    U = _fusion()
    settings = _settings(iterations=ctm_iterations)
    settings.CTM_trun_tol = projector_tol
    return VariationalProblem(layout,U,KC._blocked_operators(U,parameters),
                               KF._hoppings(parameters,:dn),chi,settings)
end

function _ipeps(v,problem)
    l = problem.layout
    tensors = _tensors(v,l)
    return ntuple(l.Lx) do x
        ntuple(l.Ly) do y
            k = 5*(x-1+l.Lx*(y-1))
            _cell_tensor(tensors[k+1],tensors[k+2],(tensors[k+3],tensors[k+4],tensors[k+5]),problem.U_phy)
        end
    end
end

_problem_energy(problem::VariationalProblem,A,env) =
    _energy(A,env.AA,env.CTM,problem.settings,problem.operators,problem.hoppings)

function variational_energy(v,problem::AbstractVariationalProblem)
    ignore_derivatives() do
        _configure(problem.layout.Lx,problem.layout.Ly,problem.chi;trun_tol=problem.settings.CTM_trun_tol)
    end
    A = _ipeps(v,problem)
    env = _fixed_environment(A,problem.chi,problem.settings)
    return _problem_energy(problem,A,env)
end

function value_and_gradient(v,problem)
    value,back = Zygote.pullback(x -> variational_energy(x,problem),v)
    gradient = back(1.0)[1]
    isfinite(value) && all(isfinite,gradient) || error("nonfinite CTMRG energy/gradient")
    return value,gradient
end

function check_gradient(v,problem;epsilon=1e-5,rng=MersenneTwister(123))
    epsilon > 0 || throw(ArgumentError("positive epsilon required"))
    E,g = value_and_gradient(v,problem)
    direction = randn(rng,ComplexF64,length(v))
    direction /= norm(direction)
    fd = (variational_energy(v+epsilon*direction,problem)-variational_energy(v-epsilon*direction,problem))/(2epsilon)
    ad = real(dot(g,direction))
    return (;energy=E,ad,finite_difference=fd,absolute_error=abs(ad-fd),
             relative_error=abs(ad-fd)/max(abs(ad),abs(fd),1e-10))
end

"""Complex-block L-BFGS (or steepest descent) with real inner product and Armijo search."""
function optimize_variational(state,parameters;chi=20,ctm_iterations=20,maxiter=50,
                               memory=8,grad_tol=1e-6,step_size=1.0,max_linesearch=20,
                               projector_tol=1e-8,callback=nothing)
    maxiter >= 0 && memory >= 0 && step_size > 0 && max_linesearch > 0 ||
        throw(ArgumentError("invalid optimizer settings"))
    v,layout = pack_state(state)
    problem = VariationalProblem(state,parameters;chi,ctm_iterations,projector_tol)
    E,g = value_and_gradient(v,problem)
    history = [(;iteration=0,energy=E,gradient_norm=norm(g),step=0.0)]
    ss,ys = Vector{ComplexF64}[],Vector{ComplexF64}[]
    status = :maxiter
    for iteration in 1:maxiter
        if norm(g) <= grad_tol
            status = :converged
            break
        end
        q = copy(g)
        alpha = zeros(length(ss))
        rho = [inv(real(dot(s,y))) for (s,y) in zip(ss,ys)]
        for i in length(ss):-1:1
            alpha[i] = rho[i]*real(dot(ss[i],q))
            q -= alpha[i]*ys[i]
        end
        scale = isempty(ss) ? 1.0 : real(dot(ss[end],ys[end]))/real(dot(ys[end],ys[end]))
        direction = scale*q
        for i in eachindex(ss)
            direction += ss[i]*(alpha[i]-rho[i]*real(dot(ys[i],direction)))
        end
        direction = -direction
        slope = real(dot(g,direction))
        if slope >= 0
            direction,slope = -g,-real(dot(g,g))
            empty!(ss); empty!(ys)
        end
        step = step_size
        accepted = false
        for _ in 1:max_linesearch
            trial = v+step*direction
            Et = variational_energy(trial,problem)
            if isfinite(Et) && Et <= E+1e-4*step*slope
                En,gn = value_and_gradient(trial,problem)
                s,y = trial-v,gn-g
                if memory > 0 && real(dot(s,y)) > 1e-10*norm(s)*norm(y)
                    push!(ss,s); push!(ys,y)
                    length(ss)>memory && (popfirst!(ss); popfirst!(ys))
                end
                v,E,g = trial,En,gn
                accepted = true
                break
            end
            step /= 2
        end
        if !accepted
            status = :line_search_failed
            break
        end
        record = (;iteration,energy=E,gradient_norm=norm(g),step)
        push!(history,record)
        isnothing(callback) || callback(unpack_state(v,layout,typeof(state)),record)
    end
    norm(g)<=grad_tol && (status=:converged)
    return (;state=unpack_state(v,layout,typeof(state)),history,status,energy=E,gradient_norm=norm(g),problem)
end
