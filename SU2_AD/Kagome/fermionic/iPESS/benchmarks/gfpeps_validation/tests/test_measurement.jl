include("../run.jl")
# Full shared/compact comparison in exactly the same saved CTM environment.
using Test
let G=KagomeGFPEPSValidation
    directory=joinpath(ROOT,"Kagome/fermionic/iPESS/results/gfpeps_validation/ipeps_D4")
    previous=load(joinpath(directory,"ctm_chi32_cut1.0e-10.jld2"),"result")
    A,U,fields=G.import_gaussian_ipeps(joinpath(directory,"tensors.mat"))
    full=G.measure_gaussian_cell(A,U,fields;chi=32,maxiter=0,verbose=false,
        projector_tol=1e-10,CTM0=previous.CTM)
    energy_error=maximum(abs(getproperty(full,k)-getproperty(previous,k)) for k in
        (:grand_per_site,:kinetic_per_site,:n_up_per_site,:n_down_per_site))
    correlation_error=maximum(maximum(abs.(getproperty(full,k)-getproperty(previous,k))) for k in
        (:hoppings,:up_hoppings,:density,:pair,:mix))
    @testset "Shared/compact fixed-environment observables" begin
        # Reassociation roundoff can be amplified by this ill-conditioned CTM.
        # Require differences well below the scientific benchmark accuracy.
        @test energy_error<1e-10
        @test correlation_error<1e-10
    end
    report=Dict("max_energy_density_difference"=>energy_error,
                "max_correlation_difference"=>correlation_error)
    println(report)
    open(io->TOML.print(io,report),
        joinpath(ROOT,"Kagome/fermionic/iPESS/results/gfpeps_validation/measurement_test.toml"),"w")
end
