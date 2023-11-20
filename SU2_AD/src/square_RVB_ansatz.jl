

function D2_point_group_symmetric_tensors()
    Vp=SU2Space(1/2=>1);
    Vv=SU2Space(0=>1,1/2=>1);

    A=TensorMap(randn,Vv*Vv,Vv'*Vv'*Vp);
    A=permute(A,(1,2,3,4,5,));
    A=A+permute(A,(2,3,4,1,5,))+permute(A,(3,4,1,2,5,))+permute(A,(4,1,2,3,5,));
    A1=A+permute(A,(2,1,4,3,5,));
    A2=A-permute(A,(2,1,4,3,5,));

    A1=A1/norm(A1);
    A2=A2/norm(A2);

    U=unitary(Vv',Vv);
    @tensor A1_total[:]:=A1[-1,-2,1,2,-5]*U[-3,1]*U[-4,2];#there are two A1
    @tensor A2[:]:=A2[-1,-2,1,2,-5]*U[-3,1]*U[-4,2];#there is only one A2

    a1=convert(Array,A1);
    a1=a1*0;
    a1[2,1,1,1,1]=1;
    a1[3,1,1,1,2]=1;
    A=TensorMap(a1,Vv*Vv*Vv*Vv,Vp);
    A1a=permute(A,(1,2,3,4,5,))+permute(A,(2,3,4,1,5,))+permute(A,(3,4,1,2,5,))+permute(A,(4,1,2,3,5,));
    @tensor A1a[:]:=A1a[-1,-2,1,2,-5]*U[-3,1]*U[-4,2];

    A1a=A1a/norm(A1a);
    coe=dot(A1a,A1_total);
    A1b=A1_total-coe*A1a;
    A1b=A1b/norm(A1b);
    
    return A1a,A1b,A2
end

function RVB_ansatz(c1,c2,c3)
    A1a,A1b,A2=D2_point_group_symmetric_tensors();
    A=c1*A1a+c2*A1b+c3*A2;
    return A
end

