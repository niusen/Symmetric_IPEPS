function build_PEPS(D,coe)
    
    A_set,E_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
    bond_tensor=A_set[1];
    tetrahedral_tensora=E_set[1];
    tetrahedral_tensorb=E_set[2];


    tetrahedral=tetrahedral_tensora*coe[1]+tetrahedral_tensorb*coe[2];

    U=unitary(space(tetrahedral_tensora,4)',space(tetrahedral_tensora,4));
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[-4,2,-6]*tetrahedral[1,-2,-3,2];



    U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2]*U_phy[-5,1,2];

    return PEPS_tensor,A_fused,U_phy
end