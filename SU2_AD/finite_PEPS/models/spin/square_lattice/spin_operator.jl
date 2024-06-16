function Hamiltonians_spin_half(Symmetry)
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

    if Symmetry=="SU2"
        return H_Heisenberg, H123chiral_tensorkit, H12_tensorkit, H31_tensorkit, H23_tensorkit 
    elseif Symmetry==nothing
        return convert(Array,H_Heisenberg), convert(Array, H123chiral_tensorkit), convert(Array, H12_tensorkit), convert(Array, H31_tensorkit), convert(Array, H23_tensorkit)
    end
end



function  H_plaquatte(J1,J2,Jchi,x_range,y_range,Lx,Ly)


    if (1<x_range[1])&(x_range[2]<Lx)
        xp="bulk";
    elseif (x_range[1]==1)
        xp="left";
    elseif (x_range[2]==Lx)
        xp="right";
    end

    if (1<y_range[1])&(y_range[2]<Ly)
        yp="bulk";
    elseif (y_range[1]==1)
        yp="bot";
    elseif (y_range[2]==Ly)
        yp="top";
    end

    if xp=="bulk"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="top"
            J_12=J1;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1;
            J_41=J1/2;
        end
    elseif xp=="left"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1;
        elseif yp=="top"
            J_12=J1;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1;
            J_41=J1;
        end
    elseif xp=="right"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="top"
            J_12=J1;
            J_23=J1;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1;
            J_34=J1;
            J_41=J1/2;
        end
    end


    J_13=J2;
    J_24=J2;

    J_123=Jchi;
    J_234=Jchi;
    J_341=Jchi;
    J_412=Jchi;

    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");
    Id=unitary(space(H_Heisenberg,1),space(H_Heisenberg,1));
    U_ss=unitary(fuse(space(Id,1)*space(Id,2)), space(Id,1)*space(Id,2));

    @tensor op_12[:]:=H_Heisenberg[1,2,3,4]*U_ss[-1,1,3]*U_ss[-2,2,4]*Id[5,6]*U_ss[-3,5,6]*Id[7,8]*U_ss[-4,7,8];
    @tensor op_13[:]:=H_Heisenberg[1,2,3,4]*U_ss[-1,1,3]*U_ss[-3,2,4]*Id[5,6]*U_ss[-2,5,6]*Id[7,8]*U_ss[-4,7,8];
    @tensor op_14[:]:=H_Heisenberg[1,2,3,4]*U_ss[-1,1,3]*U_ss[-4,2,4]*Id[5,6]*U_ss[-3,5,6]*Id[7,8]*U_ss[-2,7,8];
    @tensor op_23[:]:=H_Heisenberg[1,2,3,4]*U_ss[-2,1,3]*U_ss[-3,2,4]*Id[5,6]*U_ss[-1,5,6]*Id[7,8]*U_ss[-4,7,8];
    @tensor op_24[:]:=H_Heisenberg[1,2,3,4]*U_ss[-2,1,3]*U_ss[-4,2,4]*Id[5,6]*U_ss[-3,5,6]*Id[7,8]*U_ss[-1,7,8];
    @tensor op_34[:]:=H_Heisenberg[1,2,3,4]*U_ss[-3,1,3]*U_ss[-4,2,4]*Id[5,6]*U_ss[-1,5,6]*Id[7,8]*U_ss[-2,7,8];


    @tensor op_123[:]:=H123chiral[1,3,5,2,4,6]*U_ss[-1,1,2]*U_ss[-2,3,4]*U_ss[-3,5,6]*Id[7,8]*U_ss[-4,7,8];
    @tensor op_234[:]:=H123chiral[1,3,5,2,4,6]*U_ss[-2,1,2]*U_ss[-3,3,4]*U_ss[-4,5,6]*Id[7,8]*U_ss[-1,7,8];
    @tensor op_341[:]:=H123chiral[1,3,5,2,4,6]*U_ss[-3,1,2]*U_ss[-4,3,4]*U_ss[-1,5,6]*Id[7,8]*U_ss[-2,7,8];
    @tensor op_412[:]:=H123chiral[1,3,5,2,4,6]*U_ss[-4,1,2]*U_ss[-1,3,4]*U_ss[-2,5,6]*Id[7,8]*U_ss[-3,7,8];

    h_plaquatte=J_12*op_12+J_13*op_13+J_41*op_14+J_23*op_23+J_24*op_24+J_34*op_34+J_123*op_123+J_234*op_234+J_341*op_341+J_412*op_412;
    return h_plaquatte
    
end