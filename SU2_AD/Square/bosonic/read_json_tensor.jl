function find_space(str,c)
    pos=[];
    for cc=1:length(str)
        if str[cc]==c
            pos=vcat(pos,cc);
        end
    end
    return pos
end

filenm="TEST_D3.json";
data=read_json_state(filenm);
a=data["sites"][1]["entries"];
A=zeros(3,3,3,3,2)*im;
for cc=1:length(a)
    str=a[cc];
    pos=find_space(str,str[2]);
    num1=str[1:pos[1]-1];
    num2=str[pos[1]+1:pos[2]-1];
    num3=str[pos[2]+1:pos[3]-1];
    num4=str[pos[3]+1:pos[4]-1];
    num5=str[pos[4]+1:pos[5]-1];
    num6=str[pos[5]+1:pos[6]-1];
    num7=str[pos[6]+1:length(str)];

    ind1=parse(Int,num1)+1;
    ind2=parse(Int,num2)+1;
    ind3=parse(Int,num3)+1;
    ind4=parse(Int,num4)+1;
    ind5=parse(Int,num5)+1;
    function change_ind(x)
        if x==1
            return 2
        elseif x==2
            return 3
        elseif x==3
            return 1
        end
    end
    A[change_ind(ind2),change_ind(ind3),change_ind(ind4),change_ind(ind5),ind1]=parse(Float64,num6)+im*parse(Float64,num7);
    
end
Vv=SU2Space(0=>1,1/2=>1);
Vp=SU2Space(1/2=>1);
A=TensorMap(A,Vv*Vv,Vv'*Vv'*Vp);

singlet=unitary(Vv',Vv);
@tensor A[:]:=A[-1,-2,1,2,-5]*singlet[-3,1]*singlet[-4,2];

jldsave("didier.jld2"; A)
