using Parameters, NaNMath

# TODO: Should we deprecate the interface that only uses the ϕ argument?
function (ls::BackTracking)(ϕ, αinitial::Tα, ϕ_0, dϕ_0) where Tα
    # @unpack c_1, ρ_hi, ρ_lo, iterations, order = ls
    c_1=ls.c_1;
    ρ_hi=ls.ρ_hi;
    ρ_lo=ls.ρ_lo;
    iterations=ls.iterations;
    order=ls.order;

    iterfinitemax = -log2(eps(real(Tα)))

    @assert order in (2,3)
    # Check the input is valid, and modify otherwise
    #backtrack_condition = 1.0 - 1.0/(2*ρ) # want guaranteed backtrack factor
    #if c_1 >= backtrack_condition
    #    warn("""The Armijo constant c_1 is too large; replacing it with
    #                   $(backtrack_condition)""")
    #   c_1 = backtrack_condition
    #end

    # Count the total number of iterations
    iteration = 0

    ϕx_0, ϕx_1 = ϕ_0, ϕ_0

    α_1, α_2 = αinitial, αinitial

    ϕx_1 = ϕ(α_1)

    # Hard-coded backtrack until we find a finite function value
    iterfinite = 0
    while !isfinite(ϕx_1) && iterfinite < iterfinitemax
        iterfinite += 1
        α_1 = α_2
        α_2 = α_1/2

        ϕx_1 = ϕ(α_2)
    end

    # Backtrack until we satisfy sufficient decrease condition
    while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
        # Increment the number of steps we've had to perform
        iteration += 1

        # Ensure termination
        if iteration > iterations
            # throw(LineSearchException("Linesearch failed to converge, reached maximum iterations $(iterations).",
            #                           α_2))
            break;
        end

        # Shrink proposed step-size:
        if order == 2 || iteration == 1
            # backtracking via quadratic interpolation:
            # This interpolates the available data
            #    f(0), f'(0), f(α)
            # with a quadractic which is then minimised; this comes with a
            # guaranteed backtracking factor 0.5 * (1-c_1)^{-1} which is < 1
            # provided that c_1 < 1/2; the backtrack_condition at the beginning
            # of the function guarantees at least a backtracking factor ρ.
            α_tmp = - (dϕ_0 * α_2^2) / ( 2 * (ϕx_1 - ϕ_0 - dϕ_0*α_2) )
        else
            div = one(Tα) / (α_1^2 * α_2^2 * (α_2 - α_1))
            a = (α_1^2*(ϕx_1 - ϕ_0 - dϕ_0*α_2) - α_2^2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div
            b = (-α_1^3*(ϕx_1 - ϕ_0 - dϕ_0*α_2) + α_2^3*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div

            if isapprox(a, zero(a), atol=eps(real(Tα)))
                α_tmp = dϕ_0 / (2*b)
            else
                # discriminant
                d = max(b^2 - 3*a*dϕ_0, Tα(0))
                # quadratic equation root
                α_tmp = (-b + sqrt(d)) / (3*a)
            end
        end

        α_1 = α_2

        α_tmp = NaNMath.min(α_tmp, α_2*ρ_hi) # avoid too small reductions
        α_2 = NaNMath.max(α_tmp, α_2*ρ_lo) # avoid too big reductions

        # Evaluate f(x) at proposed position
        ϕx_0, ϕx_1 = ϕx_1, ϕ(α_2)
    end

    return α_2, ϕx_1
end
