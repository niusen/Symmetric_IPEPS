function build_PEPS(bond_tensor,square_tensor)
    
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[-4,2,-6]*square_tensor[1,-2,-3,2];


    U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2]*U_phy[-5,1,2];

    return PEPS_tensor,A_fused,U_phy

end