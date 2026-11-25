include(joinpath(@__DIR__,"..","run.jl"))
using Test
const C=KagomeFermionicCTM.Legacy
isdefined(C,:compact_double_layer) || Base.include(C,joinpath(ROOT,"src/fermionic/kagome/gfpeps/ctm/compact_double_layer.jl"))
Random.seed!(91729)
errors=Float64[]
@testset "Compact double layer equals shared contraction" begin
    for (D,p) in ((2,4),(4,4),(4,64))
        V=Rep[ℤ₂](0=>D÷2,1=>D÷2);P=Rep[ℤ₂](0=>p÷2,1=>p÷2)
        A=KagomeFermionicZ2._allout(randn(ComplexF64,V'*V*V*V',P'))
        B=KagomeFermionicZ2._allout(randn(ComplexF64,V'*V*V*V',P'))
        A=A/norm(A);B=B/norm(B)
        old=invoke(C.build_double_layer_swap,Tuple{Any,Any},A',B)
        new=C.compact_double_layer(A',B)
        err=norm(old[1]-new[1])/norm(old[1]);push!(errors,err)
        @test err<2e-13
        @test all(old[j]==new[j] for j in 2:5)
        for q in (1,2)
            Q=Rep[ℤ₂](1=>q)
            O=randn(ComplexF64,P,P*Q')
            @tensor extra[:]:=B[-1,-2,-3,-4,1]*O[-5,1,-6]
            old=invoke(C.build_double_layer_swap_extra_leg,Tuple{Any,Any},A',extra)
            new=C.compact_double_layer(A',extra,true)
            err=norm(old[1]-new[1])/norm(old[1]);push!(errors,err)
            @test err<2e-13
            @test all(old[j]==new[j] for j in 2:5)
        end
        println("D=$D, p=$p, relative errors=",errors[end-2:end]);flush(stdout)
    end
    C.install_compact_double_layer!()
    A=convert_iPESS_to_iPEPS(KagomeGFPEPSValidation.import_gaussian_ipess(
        joinpath(ROOT,"Kagome/fermionic/iPESS/results/gfpeps_validation/ipess_D4/tensors.mat"))[1])[1][1]
    old=invoke(C.build_double_layer_swap,Tuple{Any,Any},A',A)[1]
    new=C.build_double_layer_swap(A',A)[1]
    @test norm(old-new)/norm(old)<2e-13
end
open(io->TOML.print(io,Dict("maximum_relative_tensor_error"=>maximum(errors),"cases"=>length(errors))),
    joinpath(ROOT,"Kagome/fermionic/iPESS/results/gfpeps_validation/compact_double_layer_test.toml"),"w")
