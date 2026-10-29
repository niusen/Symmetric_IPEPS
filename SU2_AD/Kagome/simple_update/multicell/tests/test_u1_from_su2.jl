# Small D4 projection/charge/checkpoint regression; no large-D or CTMRG scan.
using Test, TensorKit, LinearAlgebra, SHA
include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","workflows","u1_from_su2_workflow.jl"))
BLAS.set_num_threads(10)

@testset "D4 SU2 seed continues in U1" begin
    seed=joinpath(@__DIR__,"..","data","su2_D4_seed.toml")
    digest=bytes2hex(sha256(read(seed)))
    source=server_load_seed(seed)
    initial=ServerU1.to_u1(source;component=1)
    @test ServerU1.check_u1(initial;Dmax=4)===initial
    saved=deepcopy(initial)
    H=ServerU1.u1_heisenberg_triangle()
    evolved=ServerU1.itebd_u1(initial,H,.002,.001;Dmax=4)
    @test ServerU1.check_u1(evolved;Dmax=4)===evolved
    @test evolved.auxiliary_charges==initial.auxiliary_charges==Dict(1=>1)
    for (a,b) in zip(saved.cells,initial.cells), name in fieldnames(ServerBase.KagomeSUTensors)
        x,y=getfield(a,name),getfield(b,name)
        @test isnothing(x) ? isnothing(y) : norm(x-y)<1e-12
    end
    # Exact finite-torus contraction checks physical charge independently of CTM.
    # Its energy is not used as an infinite-lattice energy estimate.
    obs=ServerU1.u1_torus_observables(evolved;h=.38)
    @test abs(obs.magnetization_normalized-1/9)<1e-12
    @test abs(obs.total_Sz+.5)<1e-12
    @test obs.variance_Sz<1e-12
    mktempdir() do dir
        file=joinpath(dir,"u1.jld2")
        ServerU1.save_u1_checkpoint(file,evolved,H)
        restored,_=ServerU1.load_u1_checkpoint(file)
        a=ServerU1.u1_torus_wavefunction(evolved)
        b=ServerU1.u1_torus_wavefunction(restored)
        @test norm(a-b)<1e-12*norm(a)
    end
    @test digest==bytes2hex(sha256(read(seed)))
end
println("D4 U1 regression: Julia ",VERSION,"; TensorKit ",pkgversion(TensorKit))
