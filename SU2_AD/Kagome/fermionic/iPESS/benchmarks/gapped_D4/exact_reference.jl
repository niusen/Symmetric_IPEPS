using LinearAlgebra, TOML

# Per spin, using dn(x,y) = [a(x+1,y), b(x,y), c(x,y+1)].
function bloch(k1,k2,t_up,t_dn)
    h = zeros(ComplexF64,3,3)
    h[1,2] = -(t_up+t_dn*cis(-k1))
    h[2,3] = -(t_up+t_dn*cis(k2))
    h[3,1] = -(t_up+t_dn*cis(k1-k2))
    return Hermitian(h+h')
end

function bands(k1,k2,t_up,t_dn)
    s2 = max(0.0,3+2cos(k1)+2cos(k2)+2cos(k1-k2))
    r = sqrt(9(t_up-t_dn)^2+4t_up*t_dn*s2)
    return [-(t_up+t_dn+r)/2,-(t_up+t_dn-r)/2,t_up+t_dn]
end

function exact_reference(;t_up=1.0,t_dn=0.2,mu=-0.6,mesh=600)
    t_up>0 && 0<=t_dn<t_up || error("this reference fills the lowest band for 0<=t_dn<t_up")
    valence_top = -2t_up+t_dn
    conduction_bottom = t_up-2t_dn
    valence_top<mu<conduction_bottom || error("mu must lie inside the first band gap")
    for (k1,k2) in ((0.3,1.4),(2pi/3,4pi/3),(0.0,0.0))
        @assert eigvals(bloch(k1,k2,t_up,t_dn)) ≈ bands(k1,k2,t_up,t_dn)
    end
    # Kahan summation, including both spins and dividing by three physical sites.
    energy,correction = 0.0,0.0
    for i in 0:mesh-1,j in 0:mesh-1
        term = 2bands(2pi*i/mesh,2pi*j/mesh,t_up,t_dn)[1]/(3mesh^2)
        corrected = term-correction
        updated = energy+corrected
        correction = (updated-energy)-corrected
        energy = updated
    end
    return Dict("mesh"=>mesh,"t_up"=>t_up,"t_dn"=>t_dn,"U"=>0.0,"mu"=>mu,
                "band_gap"=>conduction_bottom-valence_top,"density"=>2/3,
                "double_occupancy"=>1/9,"kinetic_per_site"=>energy,
                "energy_per_site"=>energy-mu*2/3)
end

if abspath(PROGRAM_FILE)==@__FILE__
    records = [exact_reference(mesh=n) for n in (300,600)]
    output = normpath(joinpath(@__DIR__,"..","..","results","gapped_D4","exact.toml"))
    mkpath(dirname(output))
    open(io -> TOML.print(io,Dict("reference"=>records)),output,"w")
    foreach(println,records)
end
