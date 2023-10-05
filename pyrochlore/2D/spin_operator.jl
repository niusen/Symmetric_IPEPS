function spin1_op()
    sx=[0 1 0;1 0 1;0 1 0]/sqrt(2);
    sy=[0 1 0;-1 0 1;0 -1 0]/sqrt(2)/im;
    sz=[1 0 0;0 0 0;0 0 -1];
    I=[1 0 0;0 1 0;0 0 1]

    @tensor sxsx[:]:=sx[-1,-5]*sx[-2,-6]*I[-3,-7]*I[-4,-8];
    @tensor sysy[:]:=sy[-1,-5]*sy[-2,-6]*I[-3,-7]*I[-4,-8];
    @tensor szsz[:]:=sz[-1,-5]*sz[-2,-6]*I[-3,-7]*I[-4,-8];

    SS=sxsx+sysy+szsz;
    V=Rep[SU₂](1=>1);
    SS_op=TensorMap(SS,V*V*V*V,V*V*V*V);


    return SS_op

end


function plaquatte_Heisenberg(J1,J2)
    SS_op12=spin1_op();
    SS_op13=permute(SS_op12,(1,3,2,4,),(5,7,6,8,));
    SS_op14=permute(SS_op12,(1,4,3,2,),(5,8,7,6,));
    SS_op23=permute(SS_op12,(3,2,1,4,),(7,6,5,8,));
    SS_op24=permute(SS_op12,(4,2,3,1,),(8,6,7,5,));
    SS_op34=permute(SS_op12,(3,4,1,2,),(7,8,5,6,));

    check_hermitian(SS_op13);
    check_hermitian(SS_op14);
    check_hermitian(SS_op23);
    check_hermitian(SS_op24);
    check_hermitian(SS_op34);

    Sigma=J1*SS_op12+J2*SS_op13+J1*SS_op14+J1*SS_op23+J2*SS_op24+J1*SS_op34;
    return Sigma

end

function plaquatte_coupling()
    SS_op12=spin1_op();
    SS_op13=permute(SS_op12,(1,3,2,4,),(5,7,6,8,));
    SS_op14=permute(SS_op12,(1,4,3,2,),(5,8,7,6,));
    SS_op23=permute(SS_op12,(3,2,1,4,),(7,6,5,8,));
    SS_op24=permute(SS_op12,(4,2,3,1,),(8,6,7,5,));
    SS_op34=permute(SS_op12,(3,4,1,2,),(7,8,5,6,));

    check_hermitian(SS_op13);
    check_hermitian(SS_op14);
    check_hermitian(SS_op23);
    check_hermitian(SS_op24);
    check_hermitian(SS_op34);

    
    return SS_op12, SS_op13, SS_op14, SS_op23, SS_op24, SS_op34

end

function plaquatte_AKLT(Sigma)
    AKLT=Sigma+Sigma*Sigma*8/19+Sigma*Sigma*Sigma/19+12/19*unitary(domain(Sigma),domain(Sigma));
    return AKLT
end

function check_hermitian(M)
    @assert norm(M-M')/norm(M)<1e-12;
end