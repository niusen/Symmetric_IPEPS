function Hamiltonians(VV)
    if isa(VV,ComplexSpace)
        # Heisenberg interaction
        Vp=(ℂ^2);
        U_phy=unitary(fuse(Vp*Vp*Vp),Vp*Vp*Vp);
        Id=I(2);
        sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
        @tensor H12[:]:=sx[-1,-4]*sx[-2,-5]*Id[-3,-6]+sy[-1,-4]*sy[-2,-5]*Id[-3,-6]+sz[-1,-4]*sz[-2,-5]*Id[-3,-6];
        @tensor H31[:]:=sx[-1,-4]*Id[-2,-5]*sx[-3,-6]+sy[-1,-4]*Id[-2,-5]*sy[-3,-6]+sz[-1,-4]*Id[-2,-5]*sz[-3,-6];
        @tensor H23[:]:=Id[-1,-4]*sx[-2,-5]*sx[-3,-6]+Id[-1,-4]*sy[-2,-5]*sy[-3,-6]+Id[-1,-4]*sz[-2,-5]*sz[-3,-6];
        @tensor H123chiral[:]:=sx[-1,-4]*sy[-2,-5]*sz[-3,-6]-sx[-1,-4]*sz[-2,-5]*sy[-3,-6]+sy[-1,-4]*sz[-2,-5]*sx[-3,-6]-sy[-1,-4]*sx[-2,-5]*sz[-3,-6]+sz[-1,-4]*sx[-2,-5]*sy[-3,-6]-sz[-1,-4]*sy[-2,-5]*sx[-3,-6];
        H12_tensorkit=TensorMap(H12, domain(U_phy)' ← domain(U_phy)');
        H31_tensorkit=TensorMap(H31, domain(U_phy)' ← domain(U_phy)');
        H23_tensorkit=TensorMap(H23, domain(U_phy)' ← domain(U_phy)');
        H123chiral_tensorkit=TensorMap(H123chiral, domain(U_phy)' ← domain(U_phy)');
        # @tensor H12_tensorkit[:]:=U_phy'[4,5,6,-1]*H12_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
        # @tensor H31_tensorkit[:]:=U_phy'[4,5,6,-1]*H31_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
        # @tensor H23_tensorkit[:]:=U_phy'[4,5,6,-1]*H23_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
        #@tensor H123chiral_tensorkit[:]:=U_phy'[4,5,6,-1]*H123chiral_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];

        @tensor H_Heisenberg[:]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4];
        H_Heisenberg=TensorMap(H_Heisenberg, Vp'*Vp' ← Vp'*Vp');
        H_Heisenberg=permute(H_Heisenberg,(1,2,3,4,));
    elseif isa(VV,typeof(SU2Space(1/2=>1)))
        # Heisenberg interaction
        Vp=SU2Space(1/2=>1);
        U_phy=unitary(fuse(Vp*Vp*Vp),Vp*Vp*Vp);
        Id=I(2);
        sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
        @tensor H12[:]:=sx[-1,-4]*sx[-2,-5]*Id[-3,-6]+sy[-1,-4]*sy[-2,-5]*Id[-3,-6]+sz[-1,-4]*sz[-2,-5]*Id[-3,-6];
        @tensor H31[:]:=sx[-1,-4]*Id[-2,-5]*sx[-3,-6]+sy[-1,-4]*Id[-2,-5]*sy[-3,-6]+sz[-1,-4]*Id[-2,-5]*sz[-3,-6];
        @tensor H23[:]:=Id[-1,-4]*sx[-2,-5]*sx[-3,-6]+Id[-1,-4]*sy[-2,-5]*sy[-3,-6]+Id[-1,-4]*sz[-2,-5]*sz[-3,-6];
        @tensor H123chiral[:]:=sx[-1,-4]*sy[-2,-5]*sz[-3,-6]-sx[-1,-4]*sz[-2,-5]*sy[-3,-6]+sy[-1,-4]*sz[-2,-5]*sx[-3,-6]-sy[-1,-4]*sx[-2,-5]*sz[-3,-6]+sz[-1,-4]*sx[-2,-5]*sy[-3,-6]-sz[-1,-4]*sy[-2,-5]*sx[-3,-6];
        H12_tensorkit=TensorMap(H12, domain(U_phy)' ← domain(U_phy)');
        H31_tensorkit=TensorMap(H31, domain(U_phy)' ← domain(U_phy)');
        H23_tensorkit=TensorMap(H23, domain(U_phy)' ← domain(U_phy)');
        H123chiral_tensorkit=TensorMap(H123chiral, domain(U_phy)' ← domain(U_phy)');
        # @tensor H12_tensorkit[:]:=U_phy'[4,5,6,-1]*H12_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
        # @tensor H31_tensorkit[:]:=U_phy'[4,5,6,-1]*H31_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
        # @tensor H23_tensorkit[:]:=U_phy'[4,5,6,-1]*H23_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
        #@tensor H123chiral_tensorkit[:]:=U_phy'[4,5,6,-1]*H123chiral_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];

        @tensor H_Heisenberg[:]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4];
        H_Heisenberg=TensorMap(H_Heisenberg, Vp'*Vp' ← Vp'*Vp');
        H_Heisenberg=permute(H_Heisenberg,(1,2,3,4,));
    end
    return H_Heisenberg, H123chiral_tensorkit, H12_tensorkit, H31_tensorkit, H23_tensorkit 



end