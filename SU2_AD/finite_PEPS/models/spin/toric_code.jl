function toric_code_terms()
    Vp=ℤ₂Space(0=>2);
    sigmax=[0 1.0;1.0 0];
    sigmaz=[1.0 0;0 -1.0];
    sigmax=TensorMap(sigmax,Vp,Vp);
    sigmaz=TensorMap(sigmaz,Vp,Vp);
    @tensor Ax[:]:=sigmax[-1,-5]*sigmax[-2,-6]*sigmax[-3,-7]*sigmax[-4,-8];
    @tensor Az[:]:=sigmaz[-1,-5]*sigmaz[-2,-6]*sigmaz[-3,-7]*sigmaz[-4,-8];

    return Ax,Az
end