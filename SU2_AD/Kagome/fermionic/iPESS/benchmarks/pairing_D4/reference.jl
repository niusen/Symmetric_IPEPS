using LinearAlgebra

# H_pair = Delta*c_up'c_dn' + conj(Delta)*c_dn*c_up.
# With real spin-independent hopping and uniform singlet pairing, each
# normal band contributes xi-E (both spins already included) to E_ground.
function pairing_reference(;t=1.0,mu=-1.0,delta=1.0,mesh=300)
    isreal(t) && t>=0 && isreal(mu) && abs(delta)>0 || error("real t>=0, real mu, nonzero pairing required")
    energy,density,F,kinetic,gap = 0.0,0.0,0.0im,0.0,Inf
    for i in 0:mesh-1,j in 0:mesh-1
        k1,k2 = 2pi*i/mesh,2pi*j/mesh
        s = sqrt(max(0.0,3+2cos(k1)+2cos(k2)+2cos(k1-k2)))
        epsilon = (-t*(1+s),-t*(1-s),2t)
        for e in epsilon
            xi = e-mu
            E = sqrt(xi^2+abs2(delta))
            energy += xi-E
            density += 1-xi/E
            F -= delta/(2E)
            kinetic += e*(1-xi/E)
            gap = min(gap,E)
        end
    end
    scale = 3mesh^2
    energy/=scale; density/=scale; F/=scale; kinetic/=scale
    @assert isapprox(energy,kinetic-mu*density+2real(conj(delta)*F);atol=1e-10)
    return Dict("t"=>t,"mu"=>mu,"pairing_real"=>real(delta),"pairing_imag"=>imag(delta),
        "mesh"=>mesh,"energy_per_site"=>energy,"density"=>density,
        "pair_real"=>real(F),"pair_imag"=>imag(F),"double_occupancy"=>(density/2)^2+abs2(F),
        "kinetic_per_site"=>kinetic,"quasiparticle_gap"=>gap)
end

# Independent 6x6 Nambu spectrum for checks at generic k.
function pairing_bdg(k1,k2; t=1.0,mu=-1.0,delta=1.0)
    h = zeros(ComplexF64,3,3)
    h[1,2]=-t*(1+cis(-k1)); h[2,3]=-t*(1+cis(k2)); h[3,1]=-t*(1+cis(k1-k2))
    h = h+h'-mu*I
    [h delta*Matrix{ComplexF64}(I,3,3); conj(delta)*Matrix{ComplexF64}(I,3,3) -h]
end
