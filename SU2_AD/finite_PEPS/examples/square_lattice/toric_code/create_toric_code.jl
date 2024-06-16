using TensorKit
using JLD2

# Vv=ℤ₂Space(0=>1,1=>1);
# Vp=ℤ₂Space(0=>2);

# T1=zeros(2,2,2,2,2);
# T1[1,1,1,1,1]=1;
# T1[1,1,2,2,1]=1;
# T1[2,2,1,1,1]=1;
# T1[2,2,2,2,1]=1;

# T1[1,1,1,1,2]=1;
# T1[1,1,2,2,2]=-1;
# T1[2,2,1,1,2]=-1;
# T1[2,2,2,2,2]=1;

# T2=permutedims(T1,(1,4,2,3,5,));

# T1=TensorMap(T1,Vv*Vv*Vv'*Vv',Vp);
# T1=permute(T1,(1,2,3,4,5,));
# T2=TensorMap(T2,Vv*Vv*Vv'*Vv',Vp);
# T2=permute(T2,(1,2,3,4,5,));

# jldsave("Toric_code.jld2"; T1, T2)



#https://arxiv.org/pdf/0809.2821.pdf
Vv=ℤ₂Space(0=>1,1=>1);
Vp=ℤ₂Space(0=>2);
T=zeros(2,2,2,2);
for c1=1:2
    for c2=1:2
        for c3=1:2
            for c4=1:2
                if mod(c1+c2+c3+c4,2)==0
                    T[c1,c2,c3,c4]=1;
                end
            end
        end
    end
end
T=TensorMap(T,Vv*Vv',Vv*Vv');

Tg=zeros(2,2,2);
Tg[1,1,1]=1;
Tg[2,2,2]=1;
Tg=TensorMap(Tg,Vv*Vv',Vp);

@tensor A[:]:=T[-1,1,2,-4]*Tg[1,-2,-5]*Tg[2,-3,-6];
jldsave("Toric_code.jld2"; A)