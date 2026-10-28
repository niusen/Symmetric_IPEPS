using Test, TensorKit, LinearAlgebra, Random
BLAS.set_num_threads(10)
include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","simple_update_cells.jl"))
using .KagomeSU2Cells

module OriginalAnsatz
using TensorKit, ChainRulesCore
const BOSONIC = normpath(joinpath(@__DIR__,"..","..","..","..","src","bosonic"))
include(joinpath(BOSONIC,"iPEPS_ansatz.jl"))
include(joinpath(BOSONIC,"kagome_AD_SU2_cell.jl"))
end

const RAW_FIELDS = fieldnames(KagomeSUTensors)
const H = heisenberg_triangle()
const V3 = SU2Space(0=>1,1/2=>1)
const G1 = KagomeSUCellGeometry()

function relative_difference(a,b)
    isnothing(a) && isnothing(b) && return 0.0
    return norm(a-b)/max(norm(a),norm(b),eps(Float64))
end

function compare_cells(a,b;tol=3e-12)
    for name in RAW_FIELDS
        @test relative_difference(getfield(a,name),getfield(b,name)) < tol
    end
end

@testset "Periodic kagome geometry" begin
    rect = KagomeSUCellGeometry(3,3)
    root3 = KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3)
    @test length(rect.representatives) == 9
    @test length(root3.representatives) == 3
    @test root3.site_map == [1 3 2; 2 1 3; 3 2 1]
    # Explicit expected down-triangle owners, independent of triangle_sites.
    @test triangle_sites(rect,1,:dn) == ((2,1),(1,2),(4,3))
    @test triangle_sites(root3,1,:dn) == ((2,1),(1,2),(3,3))
    for g in (rect,root3), x in -3:6, y in -3:6
        @test representative(g,x,y) == representative(g,x+3,y+3)
    end
    for x in -3:6, y in -3:6
        @test representative(root3,x,y) == representative(root3,x+1,y+1)
        @test representative(root3,x,y) == representative(root3,x-1,y+2)
    end
    for g in (G1,rect,root3), layer in (:up,:dn)
        sites = [s for r in eachindex(g.representatives) for s in triangle_sites(g,r,layer)]
        @test sort(sites) == [(r,b) for r in eachindex(g.representatives) for b in 1:3]
    end
    @test_throws ArgumentError KagomeSUCellGeometry(2,2;pattern=:sqrt3xsqrt3)
    @test_throws ArgumentError KagomeSUCellGeometry(0,3)
    @test_throws ArgumentError triangle_sites(rect,1,:bad)
end

@testset "1x1 matches original itebd, including boundary half steps" begin
    for complex in (false,true)
        initial = initialize_cells(G1,V3;rng=MersenneTwister(431),complex)
        before = deepcopy(initial)
        c = initial.cells[1]
        reference = KagomeSU2Cells.Legacy.itebd(
            (getfield(c,name) for name in RAW_FIELDS)...,H,1e-6,.002,.001,3,false)
        updated = itebd_cells(initial,H,.002,.001;Dmax=3)
        for (name,value) in zip(RAW_FIELDS,reference)
            @test relative_difference(getfield(updated.cells[1],name),value) < 3e-12
        end
        compare_cells(initial.cells[1],before.cells[1]) # nonmutating interface
        @test check_state(updated;Dmax=3) === updated
    end
end

@testset "Homogeneous lift preserves original state on both large cells" begin
    seed = initialize_cells(G1,V3;rng=MersenneTwister(51)).cells[1]
    reference = itebd_cells(initialize_cells(G1,V3;seed_cell=seed),H,.002,.001;Dmax=3)
    for pattern in (:rectangular,:sqrt3xsqrt3)
        g = KagomeSUCellGeometry(3,3;pattern)
        state = initialize_cells(g,V3;seed_cell=seed)
        @test state.cells[1].B_a !== state.cells[2].B_a
        updated = itebd_cells(state,H,.002,.001;Dmax=3)
        for c in updated.cells
            compare_cells(c,reference.cells[1])
        end
    end
end

@testset "Nonuniform sqrt3 quotient agrees with expanded 3x3 evolution" begin
    g = KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3)
    tied = initialize_cells(g,V3;rng=MersenneTwister(75),complex=true)
    rectg = KagomeSUCellGeometry(3,3)
    expanded = KagomeSUCell(rectg,
        [deepcopy(tied.cells[representative(g,x,y)]) for (x,y) in rectg.representatives])
    @test norm(tied.cells[1].B_a-tied.cells[2].B_a) > 1e-4
    tied_final = itebd_cells(tied,H,.002,.001;Dmax=4)
    full_final = itebd_cells(expanded,H,.002,.001;Dmax=4)
    for (r,(x,y)) in enumerate(rectg.representatives)
        compare_cells(full_final.cells[r],tied_final.cells[representative(g,x,y)])
    end
    @test check_state(tied_final;Dmax=4) === tied_final
    @test check_state(full_final;Dmax=4) === full_final
    # One tuple per CTMRG grid position; all five tensors exported in old order.
    exported = to_ipess_cells(tied_final,tuple)
    @test size(exported) == (3,3)
    @test exported[1,1][1] !== exported[2,2][1]
    @test norm(exported[1,1][1]-exported[2,2][1]) < 1e-13
    @test norm(exported[1,1][5]-tied_final.cells[1].T_d) < 1e-13
    # Exercise the REAL original mutable and AD-immutable ansatz constructors
    # and build_A_kagome, rather than only comparing a substitute tuple.
    for su_state in (tied_final,full_final), constructor in
        (OriginalAnsatz.Kagome_iPESS,OriginalAnsatz.Kagome_iPESS_immutable)
        x = to_ipess_cells(su_state,constructor)
        @test eltype(x) == constructor
        for cell in x
            unfused,fused,_ = OriginalAnsatz.build_A_kagome(cell)
            @test norm(unfused) > 0
            @test isapprox(norm(unfused),norm(fused);rtol=3e-12)
        end
    end
end

@testset "3x3 down layer uses the correct cross-cell bonds and lambdas" begin
    g = KagomeSUCellGeometry(3,3)
    state = initialize_cells(g,V3;rng=MersenneTwister(81))
    before = deepcopy(state)
    gate,half = KagomeSU2Cells.Legacy.trotter_gate(H,.001)
    # Triangle anchored at (1,1): B1 from (2,1), B2 onsite, B3 from (1,2).
    owners = (2,1,4)
    c = before.cells
    expected = KagomeSU2Cells.Legacy.Tri_T_dn(c[1].T_d,c[2].B_a,c[1].B_b,c[4].B_c,
        c[2].lambda_u_a,c[1].lambda_u_b,c[4].lambda_u_c,half,1e-6,3,false)
    sweep!(state,half,:dn;Dmax=3)
    @test norm(state.cells[1].T_u-before.cells[1].T_u) == 0
    for (b,owner) in enumerate(owners)
        @test relative_difference(getfield(state.cells[owner],(:B_a,:B_b,:B_c)[b]),
                                  expected[b]) < 3e-12
        @test relative_difference(getfield(state.cells[owner],
                                  (:lambda_d_a,:lambda_d_b,:lambda_d_c)[b]),
                                  expected[3+b]) < 3e-12
    end
    @test relative_difference(state.cells[1].T_d,expected[7]) < 3e-12
    @test check_state(state;Dmax=3) === state
end

@testset "Checkpoint round trip and continuation preserve raw SU(2) state" begin
    for pattern in (:rectangular,:sqrt3xsqrt3)
        g = KagomeSUCellGeometry(3,3;pattern)
        initial = initialize_cells(g,V3;rng=MersenneTwister(96))
        first_stage = itebd_cells(initial,H,.001,.001;Dmax=3)
        mktempdir() do folder
            path = joinpath(folder,"state.jld2")
            save_checkpoint(path,first_stage,H;metadata=Dict("stage"=>1))
            restored,data = load_checkpoint(path)
            @test data["metadata"]["stage"] == 1
            @test norm(data["H"]-H) == 0
            for r in eachindex(first_stage.cells)
                compare_cells(restored.cells[r],first_stage.cells[r])
            end
            continued = itebd_cells(restored,data["H"],.001,.0005;Dmax=3)
            direct = itebd_cells(first_stage,H,.001,.0005;Dmax=3)
            for r in eachindex(direct.cells)
                compare_cells(continued.cells[r],direct.cells[r])
            end
        end
        estimates = su_energy_estimates(first_stage,H)
        @test all(-.75-1e-12 <= e <= .75+1e-12 for e in [estimates.up;estimates.down])
        @test isfinite(estimates.per_site)
        @test length(bond_report(first_stage)) == 3length(g.representatives)
        @test_throws ArgumentError itebd_cells(initial,H,.001,0.;Dmax=3)
        @test_throws ArgumentError itebd_cells(initial,H,-.001,.001;Dmax=3)
        @test_throws ArgumentError sweep!(deepcopy(initial),H,:up;Dmax=3)
    end
end

println("Validation completed with Julia ",VERSION," / TensorKit ",pkgversion(TensorKit))
