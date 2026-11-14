using Test, TensorKit, LinearAlgebra, Random, JLD2
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","KagomeDMI.jl"))
using .KagomeDMI
const DMI=KagomeDMI
const N=DMI.N
BLAS.set_num_threads(2);N.prepare_device!("cpu")
report=Dict{String,Any}()
function torus(s)
    anchor=only(keys(s.auxiliary_charges))
    c1,c2,c3=s.cells[[mod1(anchor+k,3) for k in 0:2]]
    a1,b1,d1,t1=c1.B_a,c1.B_b,c1.B_c,c1.T_u
    a2,b2,d2,t2=c2.B_a,c2.B_b,c2.B_c,c2.T_u
    a3,b3,d3,t3=c3.B_a,c3.B_b,c3.B_c,c3.T_u
    @tensor f1[:] := t1[1,2,3,-7]*a1[-1,1,-4]*b1[-2,2,-5]*d1[-3,3,-6]
    @tensor f2[:] := t2[1,2,3]*a2[-1,1,-4]*b2[-2,2,-5]*d2[-3,3,-6]
    @tensor f3[:] := t3[1,2,3]*a3[-1,1,-4]*b3[-2,2,-5]*d3[-3,3,-6]
    td1,td2,td3=c1.T_d,c2.T_d,c3.T_d
    @tensoropt psi[:] := f1[1,2,3,-1,-2,-3,-10]*f2[4,5,6,-4,-5,-6]*f3[7,8,9,-7,-8,-9]*td1[4,2,9]*td2[7,5,3]*td3[1,8,6]
    vec(convert(Array,psi))
end
@testset "DMI Hamiltonian and signs" begin
    for rep in (:dense,:u1)
        H=triangle_hamiltonian(J=.7,Dz=.2,h=.38,representation=rep)
        a=reshape(convert(Array,H),8,8)
        Z=sum(N.embedded_operator(i,N.SZ) for i in 1:3)
        @test norm(a-a')<1e-13
        @test norm(a*Z-Z*a)<1e-13
        base=rep==:dense ? N.dense_tensor(DMI.BS.heisenberg_triangle()) : DMI.SU.u1_heisenberg_triangle()
        @test norm(triangle_hamiltonian(J=1.,Dz=0.,h=0.,representation=rep)-base)<1e-13
    end
end
@testset "Dense product reference and old-energy regression" begin
    for chirality in (1,-1)
        s=product_120(;chirality)
        r=measure(s;J=0.,Dz=1.,h=0.,chi=2,iterations=3,verbose=false)
        @test r["energy_per_spin"]≈-chirality*sqrt(3)/4 atol=1e-12
        for c in r["cells"],name in ("kappa_up_CCW","kappa_down_CCW")
            @test all(abs.(c[name].+chirality*sqrt(3)/8).<1e-12)
        end
        report["product_chirality_$(chirality)_energy"]=r["energy_per_spin"]
        f=N.build_factors(s);env,_=N.native_ctmrg(f;chi=2,iterations=3,verbose=false)
        old=N.native_observables(f,env;h=.38)
        new=DMI.dmi_observables(f,env;J=1.,Dz=0.,h=.38)
        @test abs(old["energy_per_spin"]-new["energy_per_spin"])<1e-12
    end
    s=product_120()
    for c in s.cells,name in DMI.BS.B_FIELDS
        t=getfield(c,name);setfield!(c,name,TensorMap(reshape(ComplexF64[1,0],1,1,2),codomain(t),domain(t)))
    end
    r=measure(s;J=.7,Dz=.3,h=.38,chi=2,iterations=2,verbose=false)
    @test r["energy_per_spin"]≈(.7/2+.38/2) atol=1e-12
end
@testset "Dense D2 complex SU and U1 projection" begin
    s=random_dense_state(2,91;Lx=1,Ly=1,pattern=:rectangular)
    initial=deepcopy(s)
    evolve!(s;schedule=[(dt=.01,tau=.02)],Dmax=2,J=0.,Dz=.3,h=0.)
    @test norm(s.cells[1].B_a-initial.cells[1].B_a)>1e-8
    r=measure(s;J=0.,Dz=.3,h=0.,chi=2,iterations=3,verbose=false)
    @test isfinite(r["energy_per_spin"])
    report["D2_dense_short_SU_energy"]=r["energy_per_spin"]
    # The projected U1 fixture is only a symmetry-path check, not a zero-field
    # ground-state search: it has the fixed 1/9 auxiliary charge sector.
    source=N.KagomeChargedSU2.initialize_charged(;rng=MersenneTwister(17))
    u=DMI.SU.to_u1(source)
    ud=N.to_dense(u)
    evolve!(u;schedule=[(dt=.001,tau=.001)],Dmax=3,J=1.,Dz=.1,h=.38)
    evolve!(ud;schedule=[(dt=.001,tau=.001)],Dmax=3,J=1.,Dz=.1,h=.38)
    @test DMI.SU.check_u1(u;Dmax=3)===u
    # The SAME initial ket, evolved with identical complex gates, must agree
    # as a closed nine-spin wavefunction (global SVD phases cancel).
    a=torus(u);b=torus(ud)
    infidelity=abs(1-abs(dot(a,b))^2/(real(dot(a,a))*real(dot(b,b))))
    @test infidelity<1e-8
    report["U1_dense_SU_nine_spin_infidelity"]=infidelity
    f=N.build_factors(u);env,_=N.native_ctmrg(f;chi=2,iterations=2,verbose=false)
    obs=DMI.dmi_observables(f,env;J=1.,Dz=.1,h=.38)
    @test isfinite(obs["energy_per_spin"])
    report["D3_U1_short_SU_energy"]=obs["energy_per_spin"]
    report["D3_U1_wavefunction_norm"]=norm(a)
    report["D3_U1_tiny_chi_hermiticity_residual"]=obs["max_hermitian_imaginary_part"]
    mktempdir() do folder
        path=joinpath(folder,"state.jld2")
        save_checkpoint(path,u;J=1.,Dz=.1,h=.38)
        restored,data=load_checkpoint(path)
        @test data["metadata"]["Dz"]==.1
        @test norm(restored.cells[1].B_a-u.cells[1].B_a)<1e-14
    end
end
@testset "Tiny complete dense workflow and saved observables" begin
    mktempdir() do folder
        cfg=(representation=:dense,input="",out=joinpath(folder,"run"),D=1,seed=91,
            Lx=1,Ly=1,pattern=:rectangular,J=0.,Dz=.3,h=0.,threads=2,trun_tol=1e-8,
            progress_every=1,schedule=[(dt=.01,tau=.01)],chi=2,ctm_iterations=2,ctm_tolerance=1e-6)
        result=DMI.run(cfg)
        @test isfinite(result["energy_per_spin"])
        @test all(isfile(joinpath(cfg.out,n)) for n in ("final.jld2","observables.toml","bond_energies.csv","site_magnetization.csv"))
    end
end
N.atomic_toml(joinpath(@__DIR__,"..","data","tests","last_result.toml"),report)
println("ALL SMALL JULIA DMI CHECKS PASSED: ",report)
