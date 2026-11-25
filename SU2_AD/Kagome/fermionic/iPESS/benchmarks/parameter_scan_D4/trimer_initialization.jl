# Start close to the exact disconnected up-triangle state, embedded in 2+2
# virtual spaces, to test whether random initialization selects another basin.
include("run.jl")
const KF = KagomeFermionicZ2
function trimer_start(;noise=1e-3)
    rng = MersenneTwister(9)
    state = initial_iPESS(1,1,Rep[ℤ₂](0=>1);rng)
    ops = KF._triangle_operators()
    vacuum = zeros(ComplexF64,64); vacuum[1] = 1
    psi = sum(ops.c[k][1]' for k in 1:3)*sum(ops.c[k][2]' for k in 1:3)*vacuum/3
    template = triangle_wavefunction(state,1,1,:up;weighted=false)
    W = TensorMap(reshape(psi,1,4,1,4,1,4),codomain(template),domain(template))
    core,Us,Ss,_ = KF._hosvd(W,4,1e-12)
    state.T_u[1,1] = core
    V = Rep[ℤ₂](0=>2,1=>2)
    for b in 1:3
        B = KF.fpermute(Us[b],(1,3,2))
        embed_B = zeros(ComplexF64,V',space(B,1)); block(embed_B,Z2Irrep(0))[1,1] = 1
        state.B[1,1,b] = KF._leg_map(B,embed_B,1)
        embed_T = zeros(ComplexF64,V,space(state.T_d[1,1],b)); block(embed_T,Z2Irrep(0))[1,1] = 1
        state.T_d[1,1] = KF._leg_map(state.T_d[1,1],embed_T,b)
        state.lambda_u[1,1,b] = KF._normalized(TensorKit.permute(Ss[b],((2,),(1,))))
        state.lambda_d[1,1,b] = unitary(V',V')
    end
    for array in (state.T_u,state.T_d,state.B),i in eachindex(array)
        A = array[i]
        array[i] = KF._normalized(A+noise*randn(rng,ComplexF64,codomain(A),domain(A)))
    end
    check_state(state;Dmax=4)
end
parameters = Dict{String,Any}("t_up"=>1.0,"t_dn"=>0.2,"U"=>0.0,"μ"=>-0.6,"centered"=>true)
directory = joinpath(OUTPUT,"initialization","trimer"); mkpath(directory)
state = trimer_start()
for (stage,(tau,dt)) in enumerate(SU_SCHEDULE)
    elapsed = @elapsed itebd!(state,parameters,tau,dt,4;trun_tol=1e-12)
    note(directory,"SU trimer stage=$stage seconds=$elapsed sectors=$(sectors(state))")
end
save_checkpoint(joinpath(directory,"su.jld2"),state;parameters,metadata=Dict("initialization"=>"exact trimer plus noise","sectors"=>sectors(state)))
measure(directory,"su",state,parameters,exact_reference())
