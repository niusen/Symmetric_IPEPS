using Test
include(joinpath(@__DIR__,"..","benchmarks","pairing_D4","common.jl"))
include(joinpath(@__DIR__,"..","benchmarks","flux_D4","reference.jl"))
include("fixtures.jl")
const KF=KagomeFermionicZ2

@testset "Inherited flux: Bloch and many-body Hamiltonians" begin
    # Match the user's existing triangular Hofstadter gauge before deleting
    # any bonds. m=px+1 is the source x coordinate of its diagonal bond.
    old=KagomeFermionicCTM.Legacy.get_Hofstadter_coefficients(2,1,
        Dict("t1"=>1.0,"t2"=>1.0,"U"=>0.0,"μ"=>0.0),(;Magnetic_cell=2))
    for px in 1:2
        m=px+1
        @test old["tx_coe_set"][px,1] ≈ -1 atol=1e-12
        @test old["ty_coe_set"][px,1] ≈ -(-1)^m atol=1e-12
        @test old["t2_coe_set"][px,1] ≈ -im*(-1)^m atol=1e-12
    end
    for chirality in (-1,1)
        p=Dict{String,Any}("t_up"=>flux_hoppings(chirality),"t_dn"=>flux_hoppings(chirality),"U"=>0.0,"μ"=>-sqrt(3)/2)
        for o in (:up,:dn)
            h=reshape(convert(Array,Hamiltonian_triangle(p;orientation=o)),64,64)
            @test h ≈ exact_triangle(p,o) atol=1e-12
            @test h ≈ h'
        end
        for (k1,k2) in ((0.0,0.0),(4pi/3,2pi/3),(.37,-1.9))
            r=sqrt(6+2cos(k1)+2cos(k2)+2cos(k1-k2))
            @test eigvals(Hermitian(flux_bloch(k1,k2;chirality))) ≈ [-r,0,r] atol=1e-12
        end
    end
    a,b=flux_reference(mesh=120),flux_reference(mesh=600)
    @test a["energy_per_site"] ≈ b["energy_per_site"] atol=1e-10
    @test b["energy_per_site"] ≈ -1.024004856691006 atol=1e-10
end

@testset "Complex flux trimers: both CTM orientations" begin
    for o in (:up,:dn)
        p=Dict{String,Any}("t_up"=>o==:up ? flux_hoppings() : (0.,0.,0.),
                          "t_dn"=>o==:dn ? flux_hoppings() : (0.,0.,0.),"U"=>0.0,"μ"=>-sqrt(3)/2)
        H=exact_triangle(p,o;onsite_fraction=1.0)
        e=eigen(Hermitian(H)); psi=e.vectors[:,1]
        @test e.values[1]/3 ≈ -sqrt(3)/3 atol=1e-12
        s=initial_iPESS(1,1,Rep[ℤ₂](0=>1);rng=MersenneTwister(9))
        W0=triangle_wavefunction(s,1,1,o;weighted=false)
        W=TensorMap(reshape(psi,1,4,1,4,1,4),codomain(W0),domain(W0))
        core,Us,Ss,_=KF._hosvd(W,4,1e-12)
        (o==:up ? s.T_u : s.T_d)[1,1]=core
        for b in 1:3
            s.B[1,1,b]=KF.fpermute(Us[b],o==:up ? (1,3,2) : (3,1,2))
            (o==:up ? s.lambda_u : s.lambda_d)[1,1,b]=KF._normalized(TensorKit.permute(Ss[b],((2,),(1,))))
        end
        r=measure_ctmrg(s,p;chi=16,maxiter=30,tol=1e-9,verbose=false)
        @test r.converged
        @test r.observables.energy_per_site ≈ e.values[1]/3 atol=1e-9
        @test r.observables.density ≈ 2/3 atol=1e-9
        @test r.observables.double_occupancy ≈ 1/9 atol=1e-9
        @test abs(r.observables.pair_amplitude)<1e-10
    end
end

@testset "Flux full-network reblocking and AD" begin
    p=Dict{String,Any}("t_up"=>flux_hoppings(),"t_dn"=>flux_hoppings(),"U"=>0.0,"μ"=>-sqrt(3)/2)
    s=initial_iPESS(1,1;rng=MersenneTwister(29))
    a=measure_ctmrg(s,p;chi=12,maxiter=60,tol=1e-9,verbose=false)
    b=measure_ctmrg(reverse_triangles(s),p;chi=12,maxiter=60,tol=1e-9,verbose=false)
    @test a.converged && b.converged
    @test a.observables.energy_per_site ≈ b.observables.energy_per_site atol=1e-7
    v,_=pack_state(s)
    pr=VariationalProblem(s,p;chi=4,ctm_iterations=4,projector_tol=1e-6)
    report=check_gradient(v,pr;epsilon=1e-5)
    @test report.relative_error<1e-3 || report.absolute_error<1e-7
    result=optimize_variational(s,p;chi=4,ctm_iterations=4,projector_tol=1e-6,maxiter=3,step_size=.2)
    @test result.energy<report.energy
end
println("FLUX TESTS COMPLETE")
