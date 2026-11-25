include("../run.jl")
using Test
const G=KagomeGFPEPSValidation
Random.seed!(68123)
V=Rep[ℤ₂](0=>2,1=>2); P=Rep[ℤ₂](0=>1,1=>1)
A=ntuple(x->(randn(ComplexF64,(V*V'*V'*V*P)←one(V)),),3)
B,info=G.balance_virtual_gauge(A;sweeps=4)
# Contract a closed three-site ring, including parity-twisted vertical traces.
# Normalization is immaterial; all physical amplitudes must be proportional.
function ring(A,twist)
    T=map(A) do col
        a=convert(Array,col[1]); t=zeros(ComplexF64,4,4,2)
        for d=1:4
            t .+= (twist && d>2 ? -1 : 1).*a[:,d,:,d,:]
        end
        t
    end
    a,b,c=T
    @tensor psi[-1,-2,-3] := a[1,2,-1]*b[2,3,-2]*c[3,1,-3]
    psi/norm(psi)
end
errors=Float64[]
@testset "Exact parity-even virtual gauge" begin
    @test info.identity_residual<1e-12
    for twist in (false,true)
        a,b=ring(A,twist),ring(B,twist)
        push!(errors,norm(b-a*dot(a,b)))
        @test errors[end]<1e-11
    end
end
open(io->TOML.print(io,Dict("identity_residual"=>info.identity_residual,
    "max_amplitude_error"=>maximum(errors),"max_gauge_condition"=>info.max_gauge_condition)),
    joinpath(ROOT,"Kagome/fermionic/iPESS/results/gfpeps_validation/gauge_test.toml"),"w")
println(info," amplitude errors: ",errors)
