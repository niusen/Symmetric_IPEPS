# Exact complex-hopping dimers check the physical network under the up/down
# reblocking used by FU, not only the algebraic round-trip of tensor axes.
function optimization_dimer(orientation,bond)
    i,j = ((1,2),(2,3),(3,1))[bond]
    t = cis(0.37)
    ops = KO.KF._triangle_operators()
    vacuum = zeros(ComplexF64,64)
    vacuum[1] = 1
    psi = (ops.c[i][1]'+conj(t)*ops.c[j][1]') *
          (ops.c[i][2]'+conj(t)*ops.c[j][2]') * vacuum/2
    state = initial_iPESS(1,1,Rep[ℤ₂](0=>1);rng=MersenneTwister(4))
    template = triangle_wavefunction(state,1,1,orientation;weighted=false)
    W = TensorMap(reshape(psi,1,4,1,4,1,4),codomain(template),domain(template))
    core,Us,Ss,_ = KO.KF._hosvd(W,4,1e-12)
    (orientation==:up ? state.T_u : state.T_d)[1,1] = core
    for b in 1:3
        state.B[1,1,b] = KO.KF.fpermute(Us[b],orientation==:up ? (1,3,2) : (3,1,2))
        (orientation==:up ? state.lambda_u : state.lambda_d)[1,1,b] =
            KO.KF._normalized(TensorKit.permute(Ss[b],((2,),(1,))))
    end
    ts = ntuple(k -> k==bond ? t : 0.0im,3)
    params = Dict("t_up"=>orientation==:up ? ts : (0,0,0),
                  "t_dn"=>orientation==:dn ? ts : (0,0,0),"U"=>0.0,"μ"=>0.0)
    return check_state(state),params
end

# All three hopping bonds must be active: separate dimers do not constrain
# the fermionic sign around the complete triangle.
function optimization_trimer(orientation,ts)
    h = zeros(ComplexF64,3,3)
    for ((i,j),t) in zip(((1,2),(2,3),(3,1)),ts)
        h[i,j] = -t
        h[j,i] = -conj(t)
    end
    eig = eigen(Hermitian(h))
    v = eig.vectors[:,1]
    ops = KO.KF._triangle_operators()
    vacuum = zeros(ComplexF64,64); vacuum[1] = 1
    psi = sum(v[k]*ops.c[k][1]' for k in 1:3)*sum(v[k]*ops.c[k][2]' for k in 1:3)*vacuum
    state = initial_iPESS(1,1,Rep[ℤ₂](0=>1);rng=MersenneTwister(9))
    template = triangle_wavefunction(state,1,1,orientation;weighted=false)
    W = TensorMap(reshape(psi,1,4,1,4,1,4),codomain(template),domain(template))
    core,Us,Ss,_ = KO.KF._hosvd(W,4,1e-12)
    (orientation==:up ? state.T_u : state.T_d)[1,1] = core
    for b in 1:3
        state.B[1,1,b] = KO.KF.fpermute(Us[b],orientation==:up ? (1,3,2) : (3,1,2))
        (orientation==:up ? state.lambda_u : state.lambda_d)[1,1,b] =
            KO.KF._normalized(TensorKit.permute(Ss[b],((2,),(1,))))
    end
    params = Dict("t_up"=>orientation==:up ? ts : (0,0,0),
                  "t_dn"=>orientation==:dn ? ts : (0,0,0),"U"=>0.0,"μ"=>0.0)
    return check_state(state),params,eig.values[1],v
end

@testset "Complete spinful triangles: loop signs" begin
    for orientation in (:up,:dn),ts in ((1.0,1.0,1.0),(1+0.2im,0.7-0.1im,0.9+0.3im))
        state,parameters,e,v = optimization_trimer(orientation,ts)
        result = measure_ctmrg(state,parameters;chi=16,maxiter=30,tol=1e-9,verbose=false)
        @test result.observables.energy_per_site ≈ 2e/3 atol=1e-9
        @test vec(result.observables.n) ≈ 2abs2.(v) atol=1e-9
        @test vec(result.observables.double_occupancy_set) ≈ abs2.(v).^2 atol=1e-9
        if orientation==:dn
            exact_hops = [2conj(v[i])*v[j] for (i,j) in ((1,2),(2,3),(3,1))]
            @test vec(result.observables.down_hopping) ≈ exact_hops atol=1e-9
        end
        rp = merge(parameters,Dict("t_up"=>parameters["t_dn"],"t_dn"=>parameters["t_up"]))
        reflected = measure_ctmrg(reverse_triangles(state),rp;chi=16,maxiter=30,tol=1e-9,verbose=false)
        @test reflected.observables.energy_per_site ≈ 2e/3 atol=1e-9
    end
end

@testset "Physical up/down reblocking: exact dimers" begin
    for orientation in (:up,:dn),bond in 1:3
        state,parameters = optimization_dimer(orientation,bond)
        reversed = reverse_triangles(state)
        rp = merge(parameters,Dict("t_up"=>parameters["t_dn"],"t_dn"=>parameters["t_up"]))
        ob = measure_ctmrg(reversed,rp;chi=16,maxiter=12,verbose=false).observables
        @test ob.energy_per_site ≈ -2/3 atol=1e-8
        @test ob.density ≈ 2/3 atol=1e-8
    end
end
