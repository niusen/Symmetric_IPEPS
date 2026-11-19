function _triangle_expectation(W,O)
    return dot(W,_apply_gate(W,O))/dot(W,W)
end

"""Product-environment (simple-update) estimates; these are NOT CTMRG energies."""
function su_observables(state,parameters)
    Lx,Ly = size(state.T_u)
    energies = zeros(ComplexF64,Lx,Ly,2)
    n = zeros(ComplexF64,Lx,Ly,3,2)
    doublon = similar(n)
    ops = _triangle_operators()
    for (o,orientation) in enumerate((:up,:dn))
        H = Hamiltonian_triangle(parameters;orientation)
        for y in 1:Ly, x in 1:Lx
            W = triangle_wavefunction(state,x,y,orientation)
            energies[x,y,o] = _triangle_expectation(W,H)
            for k in 1:3
                n[x,y,k,o] = _triangle_expectation(W,_physical_map(ops.n[k]))
                doublon[x,y,k,o] = _triangle_expectation(W,_physical_map(ops.doublon[k]))
            end
        end
    end
    return (;energy_per_site=real(sum(energies))/(3Lx*Ly),triangle_energies=energies,
             density=real(sum(n))/length(n),double_occupancy=real(sum(doublon))/length(doublon),
             n,double_occupancy_set=doublon,method=:simple_update_product_environment)
end

"""
Contract the up triangle and its B_b-linked down simplex into (L,D,R,U,p).
Physical order is (a,b,c); use U_phy to transform every blocked operator.
Geometry matches the existing bosonic build_A_kagome.
"""
function build_A_kagome(state,x,y)
    W = triangle_wavefunction(state,x,y,:up;weighted=false) # ea,pa,eb,pb,ec,pc
    # Bring eb and down-simplex b to the contraction seam with Fermi swaps.
    W = fpermute(W,(1,2,4,5,6,3)) # ea,pa,pb,ec,pc,eb
    T = fpermute(state.T_d[x,y],(2,1,3)) # b,a,c
    M = TensorKit.permute(W,((1,2,3,4,5),(6,))) *
        TensorKit.permute(T,((1,),(2,3)))
    A = fpermute(M,(1,7,6,4,2,3,5)) # ea,c_dn,a_dn,ec,pa,pb,pc
    # Match the incoming-left-edge ordering of build_double_layer_swap and
    # hopping_x/y/diagonala. The attached down-simplex branch has parity
    # q_b = q_R + q_D; routing it past L contributes (-1)^(q_L*(q_R+q_D)).
    # This is a virtual crossing, not an extra physical Jordan-Wigner string.
    A = _crossing(_crossing(A,1,2),1,3)
    P = physical_space()
    U_phy = unitary(fuse(P*P*P),P*P*P)
    @tensor A_fused[-1,-2,-3,-4,-5] := A[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3]
    return A,_allout(A_fused),U_phy
end

function convert_iPESS_to_iPEPS(state)
    check_state(state)
    Lx,Ly = size(state.T_u)
    return ntuple(x -> ntuple(y -> _normalized(build_A_kagome(state,x,y)[2]),Ly),Lx)
end

# CTMRG routines are included in a separate adapter to keep simple update
# independent of the legacy AD/optimization globals and optional dependencies.
function evaluate_ob_cell end
