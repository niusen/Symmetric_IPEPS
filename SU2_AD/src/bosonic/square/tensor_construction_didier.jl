Vv=Rep[SU₂](0=>1, 1/2=>1);
Vp=Rep[SU₂](1/2=>1);

A1=zeros(3,3,3,3,2);
A1[2,1,1,1,1]=-1;
A1[3,1,1,1,2]=-1;
A1[1,2,1,1,1]=1;
A1[1,3,1,1,2]=1;
A1[1,1,2,1,1]=-1;
A1[1,1,3,1,2]=-1;
A1[1,1,1,2,1]=1;
A1[1,1,1,3,2]=1;

A1=TensorMap(A1,Vv*Vv*Vv*Vv,Vp);


function lambda0(s1,s2)
    if s1==s2
        return (-1)^(s1)
    else
        return 0
    end
end
function lambda2(s1,s2)
    if s1==s2
        return (-1)^(s1+1)
    else
        return 0
    end
end
function bar(s)
    if s==1
        sbar=2;
    elseif s==2
        sbar=1;
    end
    return sbar;
end

A2=zeros(3,3,3,3,2);
for s=1:2
    for sp=1:2
        sbar=bar(s);
        A2[s+1,sbar+1,s+1,1,sp]=lambda0(s,sp)*2;
        A2[s+1,1,s+1,sbar+1,sp]=lambda0(s,sp)*2;
        A2[1,s+1,sbar+1,s+1,sp]=-lambda0(s,sp)*2;
        A2[sbar+1,s+1,1,s+1,sp]=-lambda0(s,sp)*2;

        A2[s+1,s+1,sbar+1,1,sp]=lambda2(s,sp);
        A2[sbar+1,1,s+1,s+1,sp]=lambda2(s,sp);

        A2[s+1,1,sbar+1,s+1,sp]=lambda2(s,sp);
        A2[sbar+1,s+1,s+1,1,sp]=lambda2(s,sp);
        A2[s+1,s+1,1,sbar+1,sp]=-lambda2(s,sp);
        A2[1,sbar+1,s+1,s+1,sp]=-lambda2(s,sp);
        A2[s+1,sbar+1,1,s+1,sp]=-lambda2(s,sp);
        A2[1,s+1,s+1,sbar+1,sp]=-lambda2(s,sp);
    end
end
A2=TensorMap(A2,Vv*Vv*Vv*Vv,Vp);

Achiral=zeros(3,3,3,3,2);
for s=1:2
    for sp=1:2

        sbar=bar(s);
        Achiral[s+1,s+1,sbar+1,1,sp]=lambda0(s,sp);
        Achiral[sbar+1,1,s+1,s+1,sp]=lambda0(s,sp);

        Achiral[s+1,1,sbar+1,s+1,sp]=-lambda0(s,sp);
        Achiral[sbar+1,s+1,s+1,1,sp]=-lambda0(s,sp);
        Achiral[s+1,s+1,1,sbar+1,sp]=lambda0(s,sp);
        Achiral[1,sbar+1,s+1,s+1,sp]=lambda0(s,sp);
        Achiral[s+1,sbar+1,1,s+1,sp]=-lambda0(s,sp);
        Achiral[1,s+1,s+1,sbar+1,sp]=-lambda0(s,sp);
    end
end
Achiral=TensorMap(Achiral,Vv*Vv*Vv*Vv,Vp);


jldsave("elementary_tensors.jld2";A1,A2,Achiral);