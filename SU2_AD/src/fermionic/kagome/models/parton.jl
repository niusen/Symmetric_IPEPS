"""Magnetic-cell parton model in the imported b=(f↑,f↓†) basis.
mu and amplitude are ordered by *physical spin*, not triangle orientation.
"""
struct PartonModel
    variant::Symbol
    phi::NTuple{2,Float64}
    mu::NTuple{2,Float64}
    amplitude::NTuple{2,Float64}
    onsite_U::Float64
end

function parton_model(variant; mu, amplitude=(1.,1.))
    v=Symbol(variant)
    phi=v===:orig ? -pi/8 : v===:mild ? -pi/16 : v===:zero ? 0.0 :
        throw(ArgumentError("parton model must be orig, mild or zero"))
    length(mu)==length(amplitude)==2 && all(isfinite,mu) && all(isfinite,amplitude) ||
        throw(ArgumentError("expected two finite chemical potentials and hopping amplitudes"))
    PartonModel(v,(phi,-pi/8),Tuple(Float64.(mu)),Tuple(Float64.(amplitude)),0.0)
end

"""Coefficient of b†ᵢ bⱼ on oriented (ab,bc,ca), with (a,b,c)=(source B,A,C)."""
function parton_hoppings(m::PartonModel,orientation,x)
    orientation in (:up,:dn) || throw(ArgumentError("invalid triangle"))
    f=orientation===:up || m.variant===:hubbard ? 0.0 : 2pi/3*mod(1-x,3)
    ntuple(3) do k
        phase=(0.,f,-f)[k]
        (m.amplitude[1]*cis(m.phi[1]/3+phase),-m.amplitude[2]*cis(-m.phi[2]/3-phase))
    end
end

function Hamiltonian_triangle(m::PartonModel;orientation=:up,x=1,onsite_fraction=0.5)
    o=_triangle_operators(); co=parton_hoppings(m,orientation,x)
    H=zeros(ComplexF64,64,64)
    for (k,(i,j)) in enumerate(((1,2),(2,3),(3,1))), s in 1:2
        op=o.c[i][s]'*o.c[j][s]
        H += co[k][s]*op+conj(co[k][s])*op'
        n=o.c[i][s]'*o.c[i][s]
        H += onsite_fraction*(s==1 ? -m.mu[1]*n : m.mu[2]*(n-I))
    end
    if !iszero(m.onsite_U)
        for i in 1:3
            n1=o.c[i][1]'*o.c[i][1]; n2=o.c[i][2]'*o.c[i][2]
            H += onsite_fraction*m.onsite_U*n1*(I-n2)
        end
    end
    _physical_map(H)
end

magnetic_period(m)=1
magnetic_period(m::PartonModel)=m.variant===:hubbard ? 1 : 3

triangle_at(m,o,x)=Hamiltonian_triangle(m;orientation=o)
triangle_at(m::PartonModel,o,x)=Hamiltonian_triangle(m;orientation=o,x)

"""Uniform electron Hubbard in the same up/down-hole input basis as the MAT files."""
function hole_hubbard(;t=(1.,1.),U=0.,mu=(0.,0.))
    length(mu)==length(t)==2 && all(isfinite,mu) && all(isfinite,t) && isfinite(U) ||
        throw(ArgumentError("invalid Hubbard parameters"))
    PartonModel(:hubbard,(0.,0.),Tuple(Float64.(mu)),(-Float64(t[1]),-Float64(t[2])),Float64(U))
end