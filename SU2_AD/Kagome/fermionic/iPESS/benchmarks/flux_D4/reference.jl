using LinearAlgebra

flux_hoppings(chirality=1) = (1.0+0im,1.0+0im,-1.0im*chirality)
function flux_bloch(k1,k2;chirality=1)
    h=zeros(ComplexF64,3,3)
    h[1,2]=-(1+cis(-k1))
    h[2,3]=-(1+cis(k2))
    h[3,1]=im*chirality*(1+cis(k1-k2))
    h+h'
end
function flux_reference(;mesh=600,mu=-sqrt(3)/2)
    -sqrt(3)<mu<0 || error("reference assumes only the lowest band is filled")
    kinetic=0.0
    for i in 0:mesh-1,j in 0:mesh-1
        k1,k2=2pi*i/mesh,2pi*j/mesh
        kinetic-=sqrt(6+2cos(k1)+2cos(k2)+2cos(k1-k2))
    end
    kinetic*=2/(3mesh^2) # two independent spins
    Dict("mesh"=>mesh,"energy_per_site"=>kinetic-2mu/3,
        "kinetic_per_site"=>kinetic,"density"=>2/3,"double_occupancy"=>1/9,
        "pair_real"=>0.0,"pair_imag"=>0.0,"band_gap"=>sqrt(3),
        "particle_or_hole_gap"=>min(-mu,sqrt(3)+mu),"mu"=>mu)
end
