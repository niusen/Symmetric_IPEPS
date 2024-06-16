include("FinitePEPS.jl")
using OptimKit

# initial peps
function initial_peps()
    L1 = 4;
    L2 = 4;
    D = 2;
    As = Array{Any,2}(undef, L1, L2);
    p = ComplexSpace(2);
    v = ComplexSpace(D);
    v0 = ComplexSpace(1)
    ifset(ifx::Bool, a, b) = ifx ? a : b

    for x = 1:L1
        for y = 1:L2
            v1 = ifset(x == 1, v0, v)
            v2 = ifset(y == 1, v0, v)
            v3 = ifset(x == L1, v0, v)
            v4 = ifset(y == L2, v0, v)

            As[x, y] = TensorMap(rand, ComplexF64, v1 * v2 * v3' * v4', p)
        end
    end

    return FinitePEPS(As)
end


function cfun1(x)

    fun(T) = real(dot(T,T))

    ∂E = fun'(x)
    
    E = fun(x)

    @assert !isnan(norm(∂E))
    return E,∂E
end


O = TensorMap(rand, ComplexF64, ℂ^2, ℂ^2)
O = O + O'
psi=initial_peps();

function cfun2(x1)
    As = copy(psi.A)
    x = copy(x1)

    #@show x
    @show typeof(x)
    @diffset As[2,2]=x;

    @tensor rho[-1;-5] := conj(As[1,1][1,2,3,4;-1])*As[1,1][1,2,3,4;-5]
    nn = @tensor rho[1,1]
    E = @tensor rho[2,1]*O[1,2]
    E = E / nn
    @show E

    return real.(E)
end

A=psi.A[2,2];
cfun1(A)
cfun2(A)
cfun2'(A)
