using Test
include(joinpath(@__DIR__,"..","benchmarks","pairing_D4","common.jl"))
const KF = KagomeFermionicZ2
const KO = KagomeFermionicOptimization
include("fixtures.jl")

function independent_pair_triangle(parameters;orientation=:up,onsite_fraction=1.0)
    p = copy(parameters)
    for key in ("t_up","t_dn")
        t = get(p,key,0.0); p[key] = t isa Number ? (t,t,t) : t
    end
    H = exact_triangle(p,orientation;onsite_fraction)
    occ=(0,3,1,2)
    bits=[occ[a]+4occ[b]+16occ[c] for c in 1:4 for b in 1:4 for a in 1:4]
    lookup=Dict(b=>i for (i,b) in enumerate(bits))
    delta=get(p,"pairing",0.0)
    for (column,b) in enumerate(bits),site in 0:2
        for (create,modes,amplitude) in ((true,(2site+1,2site),delta),(false,(2site,2site+1),conj(delta)))
            state,sign,valid=b,1,true
            for mode in modes
                occupied=!iszero(state & (1<<mode))
                if occupied==create; valid=false; break; end
                sign*=(-1)^count_ones(state & ((1<<mode)-1))
                state=state ⊻ (1<<mode)
            end
            valid && (H[lookup[state],column]+=onsite_fraction*amplitude*sign)
        end
    end
    H
end

@testset "Pairing matrix, parity, and zero-pairing compatibility" begin
    p=Dict{String,Any}("t_up"=>(1.0+0.2im,0.7,0.9),"t_dn"=>(0.4,0.6im,0.2),"U"=>1.2,"μ"=>-0.7,"pairing"=>0.6+0.3im)
    for orientation in (:up,:dn),fraction in (0.5,1.0)
        H=reshape(convert(Array,Hamiltonian_triangle(p;orientation,onsite_fraction=fraction)),64,64)
        @test H ≈ independent_pair_triangle(p;orientation,onsite_fraction=fraction) atol=1e-12
        @test H ≈ H'
        parity=kron(KF._local_operators().parity,kron(KF._local_operators().parity,KF._local_operators().parity))
        @test norm(H*parity-parity*H)<1e-12
        N=sum(KF._triangle_operators().n)
        @test norm(H*N-N*H)>0.1
    end
    no_pair=copy(p); delete!(no_pair,"pairing")
    @test Hamiltonian_triangle(no_pair) == Hamiltonian_triangle(merge(no_pair,Dict("pairing"=>0.0)))
    @test_throws ArgumentError Hamiltonian_triangle(merge(p,Dict("pairing"=>NaN)))
end

@testset "BdG energy, spin count, and derivatives" begin
    for delta in (0.5,1.0,2.0)
        a=pairing_reference(;delta,mesh=120); b=pairing_reference(;delta,mesh=240)
        @test a["energy_per_site"] ≈ b["energy_per_site"] atol=1e-10
        @test b["quasiparticle_gap"] ≈ delta atol=1e-12
        @test b["density"] ≈ 1-inv(sqrt(9+delta^2)) atol=1e-11
        epsilon=1e-5
        @test (pairing_reference(delta=delta+epsilon,mesh=60)["energy_per_site"]-pairing_reference(delta=delta-epsilon,mesh=60)["energy_per_site"])/(2epsilon) ≈ 2b["pair_real"] atol=1e-8
        @test (pairing_reference(delta=delta,mu=-1+epsilon,mesh=60)["energy_per_site"]-pairing_reference(delta=delta,mu=-1-epsilon,mesh=60)["energy_per_site"])/(2epsilon) ≈ -b["density"] atol=1e-8
        k1,k2=0.31,1.43
        energies=eigvals(Hermitian(pairing_bdg(k1,k2;delta=delta*cis(0.4))))
        s=sqrt(3+2cos(k1)+2cos(k2)+2cos(k1-k2))
        expected=sort(vcat(-sqrt.([s^2+delta^2,s^2+delta^2,9+delta^2]),sqrt.([s^2+delta^2,s^2+delta^2,9+delta^2])))
        @test energies ≈ expected atol=1e-12
    end
end

@testset "Exact paired trimers in both orientations" begin
    delta=0.7+0.2im; mu=-0.4
    for orientation in (:up,:dn)
        p=Dict{String,Any}("t_up"=>orientation==:up ? 1.0 : 0.0,"t_dn"=>orientation==:dn ? 1.0 : 0.0,"U"=>0.0,"μ"=>mu,"pairing"=>delta)
        H=independent_pair_triangle(p;orientation)
        e=eigen(Hermitian(H)); psi=e.vectors[:,1]
        xi=[-2.0,1.0,1.0].-mu; qp=sqrt.(xi.^2 .+abs2(delta))
        @test e.values[1] ≈ sum(xi-qp) atol=1e-12
        density=sum(1 .-xi./qp)/3; F=-delta*sum(inv.(qp))/6
        state=initial_iPESS(1,1,Rep[ℤ₂](0=>1);rng=MersenneTwister(9))
        W0=triangle_wavefunction(state,1,1,orientation;weighted=false)
        W=TensorMap(reshape(psi,1,4,1,4,1,4),codomain(W0),domain(W0))
        core,Us,Ss,_=KF._hosvd(W,4,1e-12)
        (orientation==:up ? state.T_u : state.T_d)[1,1]=core
        for b in 1:3
            state.B[1,1,b]=KF.fpermute(Us[b],orientation==:up ? (1,3,2) : (3,1,2))
            (orientation==:up ? state.lambda_u : state.lambda_d)[1,1,b]=KF._normalized(TensorKit.permute(Ss[b],((2,),(1,))))
        end
        measured=measure_ctmrg(state,p;chi=16,maxiter=30,tol=1e-9,verbose=false).observables
        @test measured.energy_per_site ≈ e.values[1]/3 atol=1e-9
        @test measured.density ≈ density atol=1e-9
        @test measured.pair_amplitude ≈ F atol=1e-9
        @test measured.double_occupancy ≈ (density/2)^2+abs2(F) atol=1e-9
    end
end

@testset "Pairing SU, FU, and AD" begin
    p=Dict{String,Any}("t"=>0.0,"U"=>0.0,"μ"=>-0.7,"pairing"=>0.6+0.3im)
    state=initial_iPESS(1,1,Rep[ℤ₂](0=>1);rng=MersenneTwister(10))
    itebd!(state,p,15.0,0.1,4)
    reference=pairing_reference(t=0.0,mu=p["μ"],delta=p["pairing"],mesh=1)
    ob=measure_ctmrg(state,p;chi=2,maxiter=8,tol=1e-8,verbose=false).observables
    @test ob.energy_per_site ≈ reference["energy_per_site"] atol=1e-10
    @test ob.pair_amplitude ≈ reference["pair_real"]+im*reference["pair_imag"] atol=1e-9
    history=full_update!(state,p,0.02,0.01,4;chi=2,ctm_iterations=4,n_sweep=2)
    @test all(r -> abs(r.fidelity-1)<1e-9,history)
    @test measure_ctmrg(state,p;chi=2,maxiter=8,tol=1e-8,verbose=false).observables.energy_per_site ≈ reference["energy_per_site"] atol=1e-10
    random_state=initial_iPESS(1,1;rng=MersenneTwister(11))
    params=Dict("t_up"=>1.0,"t_dn"=>0.4,"U"=>0.0,"μ"=>-1.0,"pairing"=>0.5)
    v,_=pack_state(random_state)
    problem=VariationalProblem(random_state,params;chi=2,ctm_iterations=2,projector_tol=1e-6)
    report=check_gradient(v,problem;epsilon=1e-5)
    @test report.relative_error<1e-3 || report.absolute_error<1e-7
    result=optimize_variational(random_state,params;chi=2,ctm_iterations=2,projector_tol=1e-6,maxiter=3,step_size=0.2)
    @test result.energy<report.energy
end

# Existing Hubbard core checks also run with pairing absent.
include("test_core.jl")
println("PAIRING TESTS COMPLETE")
