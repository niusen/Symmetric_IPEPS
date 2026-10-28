using Test, TensorKit, LinearAlgebra, Random
BLAS.set_num_threads(10)
include(joinpath(@__DIR__,"..","..","..","src","bosonic","kagome",
                 "simple_update_charged_cell.jl"))
using .KagomeChargedSU2
const CS = KagomeChargedSU2
const BS = CS.BaseSU
const Hcharged = BS.heisenberg_triangle()

function charged_cluster(core,bonds,weights)
    a,b,c = bonds
    wa,wb,wc = weights
    @tensor a[:] := a[1,-2,-3]*wa[1,-1]
    @tensor b[:] := b[1,-2,-3]*wb[1,-1]
    @tensor c[:] := c[1,-2,-3]*wc[1,-1]
    @tensor A[:] := core[1,2,3,-7]*a[-1,1,-2]*b[-3,2,-4]*c[-5,3,-6]
    return convert(Array,A)
end

function phase_distance(a,b)
    a,b = a/norm(a),b/norm(b)
    overlap = dot(vec(a),vec(b))
    return norm(a-b*conj(overlap)/abs(overlap))
end

function dense_projected_torus(s;component=1)
    cells = polarized_dense_cells(s;component)
    c1,c2,c3 = cells[[mod1(s.anchor+k,3) for k in 0:2]]
    a1,b1,d1,t1 = c1.B1,c1.B2,c1.B3,c1.Tup
    a2,b2,d2,t2 = c2.B1,c2.B2,c2.B3,c2.Tup
    a3,b3,d3,t3 = c3.B1,c3.B2,c3.B3,c3.Tup
    @tensor f1[:] := t1[1,2,3]*a1[-1,1,-4]*b1[-2,2,-5]*d1[-3,3,-6]
    @tensor f2[:] := t2[1,2,3]*a2[-1,1,-4]*b2[-2,2,-5]*d2[-3,3,-6]
    @tensor f3[:] := t3[1,2,3]*a3[-1,1,-4]*b3[-2,2,-5]*d3[-3,3,-6]
    td1,td2,td3 = c1.Tdn,c2.Tdn,c3.Tdn
    @tensoropt psi[:] := f1[1,2,3,-1,-2,-3]*f2[4,5,6,-4,-5,-6]*
        f3[7,8,9,-7,-8,-9]*td1[4,2,9]*td2[7,5,3]*td3[1,8,6]
    return psi
end

@testset "Auxiliary spin selects a physical multiplet, not a singlet" begin
    for j in (1//2,3//2), anchor in 1:3
        s = initialize_charged(;auxiliary_spin=j,anchor,rng=MersenneTwister(152))
        @test numind(s.raw.cells[anchor].T_u) == 4
        @test dim(space(s.raw.cells[anchor].T_u,4)) == 2j+1
        high = exact_torus_observables(s;h=.38)
        low = exact_torus_observables(s;component=Int(2j+1),h=.38)
        @test isapprox(high.total_Sz,-Float64(j);atol=3e-12)
        @test isapprox(low.total_Sz,Float64(j);atol=3e-12)
        @test isapprox(high.magnetization_normalized,2Float64(j)/9;atol=3e-12)
        @test high.variance_Sz < 3e-12
        @test isapprox(high.spin_squared,Float64(j*(j+1));atol=3e-12)
        @test isapprox(high.exchange_energy_per_site,low.exchange_energy_per_site;atol=3e-12)
        @test isapprox(high.field_energy_per_site-low.field_energy_per_site,
                       -.38*2Float64(j)/9;atol=3e-12)
    end
    @test_throws ArgumentError initialize_charged(;auxiliary_spin=0)
    @test_throws ArgumentError initialize_charged(;anchor=4)
end

@testset "Trivial spectator recovers original up-triangle update" begin
    s = BS.initialize_cells(BS.KagomeSUCellGeometry();rng=MersenneTwister(73))
    gate,half = CS.Kernel.trotter_gate(Hcharged,.001)
    BS.sweep!(s,half,:dn;Dmax=3)
    c = s.cells[1]
    Q0 = SU2Space(0=>1)
    q = TensorMap(ones(1),Q0,one(Q0))
    @tensor Tu4[:] := c.T_u[-1,-2,-3]*q[-4]
    original = CS.Kernel.Tri_T_up(c.T_u,c.B_a,c.B_b,c.B_c,
        c.lambda_d_a,c.lambda_d_b,c.lambda_d_c,gate,1e-6,3,false)
    charged = CS._charged_up(Tu4,c.B_a,c.B_b,c.B_c,
        c.lambda_d_a,c.lambda_d_b,c.lambda_d_c,gate,1e-6,3)
    @tensor refcore[:] := original[7][-1,-2,-3]*q[-4]
    weights = (c.lambda_d_a,c.lambda_d_b,c.lambda_d_c)
    refcluster = charged_cluster(refcore,original[1:3],weights)
    newcluster = charged_cluster(charged[7],charged[1:3],weights)
    @test phase_distance(refcluster,newcluster) < 3e-12
    for i in 4:6
        @test norm(original[i]-charged[i]) < 3e-12
    end
end

@testset "Untruncated charged update agrees with physical gate action" begin
    s = initialize_charged(;rng=MersenneTwister(45))
    gate,half = CS.Kernel.trotter_gate(Hcharged,.001)
    BS.sweep!(s.raw,half,:dn;Dmax=12)
    c = s.raw.cells[s.anchor]
    wa,wb,wc = c.lambda_d_a,c.lambda_d_b,c.lambda_d_c
    @tensor a[:] := c.B_a[1,-2,-3]*wa[1,-1]
    @tensor b[:] := c.B_b[1,-2,-3]*wb[1,-1]
    @tensor d[:] := c.B_c[1,-2,-3]*wc[1,-1]
    @tensor A[:] := c.T_u[1,2,3,-7]*a[-1,1,-2]*b[-3,2,-4]*d[-5,3,-6]
    @tensor expected[:] := gate[-2,-4,-6,1,2,3]*A[-1,1,-3,2,-5,3,-7]
    updated = CS._charged_up(c.T_u,c.B_a,c.B_b,c.B_c,wa,wb,wc,gate,1e-12,12)
    actual = charged_cluster(updated[7],updated[1:3],(wa,wb,wc))
    @test phase_distance(convert(Array,expected),actual) < 3e-11
    @test space(updated[7],4) == SU2Space(1/2=>1)
end

@testset "Polarized export and SU preserve the intended magnetization" begin
    for Dmax in (3,4)
        s = initialize_charged(;rng=MersenneTwister(1234))
        original = deepcopy(s)
        updated = itebd_charged(s,Hcharged,.002,.001;Dmax)
        @test check_charged(updated;Dmax) === updated
        @test norm(s.raw.cells[1].T_u-original.raw.cells[1].T_u) == 0
        obs = exact_torus_observables(updated;h=.38)
        @test isapprox(obs.total_Sz,-.5;atol=3e-12)
        @test isapprox(obs.magnetization_normalized,1/9;atol=3e-12)
        @test isapprox(obs.spin_squared,.75;atol=3e-12)
        @test isfinite(obs.exchange_energy_per_site)
        openpsi = closed_sqrt3_wavefunction(updated)
        for component in 1:2
            dense = dense_projected_torus(updated;component)
            @test phase_distance(dense,copy(selectdim(openpsi,10,component))) < 3e-12
        end
        # Summing over Q (instead of choosing a component) restores zero M.
        probabilities = reshape(abs2.(openpsi),512,2)
        mz = [4.5-count_ones(index) for index in 0:511]
        @test abs(sum(probabilities .* mz)/sum(probabilities)) < 3e-12
        mktempdir() do folder
            path = joinpath(folder,"charged.jld2")
            save_charged_checkpoint(path,updated,Hcharged;metadata=Dict("Dmax"=>Dmax))
            restored,data = load_charged_checkpoint(path)
            @test data["metadata"]["Dmax"] == Dmax
            @test phase_distance(closed_sqrt3_wavefunction(restored),openpsi) < 3e-12
            resumed = itebd_charged(restored,data["H"],.001,.0005;Dmax)
            direct = itebd_charged(updated,Hcharged,.001,.0005;Dmax)
            @test phase_distance(closed_sqrt3_wavefunction(resumed),
                                 closed_sqrt3_wavefunction(direct)) < 3e-12
        end
    end
end

println("Charged-cell checks completed: Julia ",VERSION," / TensorKit ",pkgversion(TensorKit))
