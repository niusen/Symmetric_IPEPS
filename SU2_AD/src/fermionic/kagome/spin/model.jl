"""Nearest-neighbor J S⋅S and a longitudinal field -h Sz, in spin-1/2 units."""
struct HeisenbergField
    J::Float64
    h::Float64
    function HeisenbergField(J::Real,h::Real)
        all(isfinite,(J,h)) || throw(ArgumentError("J and h must be finite"))
        new(J,h)
    end
end

function local_terms(U)
    spin = blocked_spin_operators(U)
    bonds = ((1,2),(2,3),(3,1))
    components = (spin.Sx,spin.Sy,spin.Sz)
    exchange = sum(s[i]*s[j] for s in components for (i,j) in bonds)
    return (;exchange,magnetization=sum(spin.Sz),components)
end

function KF.Hamiltonian_triangle(model::HeisenbergField;orientation=:up,onsite_fraction=0.5)
    orientation in (:up,:dn) || throw(ArgumentError("invalid triangle orientation"))
    U = Opt._fusion(projected_spin_space())
    terms = local_terms(U)
    # Each spin belongs to two triangles. The symmetric time step applies
    # each orientation for a total time dt.
    U'*(model.J*terms.exchange - onsite_fraction*model.h*terms.magnetization)*U
end

struct SpinVariationalProblem <: Opt.AbstractVariationalProblem
    layout::Opt.TensorLayout
    U_phy::TensorMap
    onsite::TensorMap
    components::Tuple
    J::Float64
    chi::Int
    settings::CTM.LS_CTMRG_settings
end

function Opt.VariationalProblem(state::ProjectedSpinIPESS,model::HeisenbergField;
                                chi=20,ctm_iterations=20,projector_tol=1e-8)
    chi>0 && ctm_iterations>0 && 0<projector_tol<1 || throw(ArgumentError("invalid spin AD settings"))
    _,layout = pack_state(state)
    U = Opt._fusion(projected_spin_space())
    terms = local_terms(U)
    settings = Opt._settings(iterations=ctm_iterations)
    settings.CTM_trun_tol = projector_tol
    # One blocked up triangle owns three spins and hence the full onsite field.
    onsite = model.J*terms.exchange-model.h*terms.magnetization
    SpinVariationalProblem(layout,U,onsite,terms.components,model.J,chi,settings)
end

"""Evaluate one copy of every bond using normalized 2×2 CTM patches.
The down bonds in a patch connect (LU.b,RU.a), (RU.b,RD.c), (LD.c,RU.a).
"""
function Opt._problem_energy(p::SpinVariationalProblem,A,env)
    Lx,Ly = p.layout.Lx,p.layout.Ly
    cells = map(CartesianIndices((Lx,Ly))) do index
        x,y = Tuple(index)
        right,lower = mod1(x+1,Lx),mod1(y+1,Ly)
        lu,ru,ld,rd = env.AA[x][y],env.AA[right][y],env.AA[x][lower],env.AA[right][lower]
        patch(a,b,c,d) = CTM.ob_2x2(env.CTM,a,b,c,d,x-1,y-1)
        denominator = patch(lu,ru,ld,rd)
        onsite = patch(GF._spin_layer(A[x][y],p.onsite),ru,ld,rd)
        down = map(p.components) do s
            rua = GF._spin_layer(A[right][y],s[1])
            lub = GF._spin_layer(A[x][y],s[2])
            rub = GF._spin_layer(A[right][y],s[2])
            ldc = GF._spin_layer(A[x][lower],s[3])
            rdc = GF._spin_layer(A[right][lower],s[3])
            patch(lub,rua,ld,rd) + patch(lu,rub,ld,rdc) + patch(lu,rua,ldc,rd)
        end
        real((onsite+p.J*sum(down))/denominator)
    end
    sum(cells)/(3Lx*Ly)
end
