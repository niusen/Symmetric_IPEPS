using CSV,JLD2
using DataFrames
cd(@__DIR__)

Lx=4;
Ly=4;
D=2;
d=2;

df = CSV.File("Heisenberg_4x4_D=2.dat"; header=false)








function remove_space(st)
    for cc=1:100
        st=replace(st, "  " => " ");
    end
    return st
end


function read_vector(st)
    vec=split(st);
    Vec=Vector{Float64}(undef,length(vec));
    for cc=1:length(vec)
        Vec[cc]=parse(Float64,vec[cc])
    end
    return Vec
end


A_set=Matrix{Any}(undef,Lx,Ly);
step=0;
for cb=1:Ly
    for ca=1:Lx
    
    
        vec=(read_vector(remove_space(df[step+1].Column1)),read_vector(remove_space(df[step+2].Column1)),);
        A_set[ca,cb]=vec;
        println([ca,cb])
        step=step+2;
    end
end

psi=Matrix{Any}(undef,Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        if ca==1
            Dl=1;
            Dr=D;
            if cb==1
                Dd=1;
                Du=D;
                T=zeros(Dl,Du,Dr,Dd,d);
                T[:,:,:,:,1]=reshape(A_set[ca,cb][1],Dl,Du,Dr,Dd,1);
                T[:,:,:,:,2]=A_set[ca,cb][2];
            elseif 1<cb<Ly
                Dd=D;
                Du=D;
                T=zeros(Dl,Du,Dr,Dd,d);
                T[:,:,:,:,1]=A_set[ca,cb][1];
                T[:,:,:,:,2]=A_set[ca,cb][2];
            elseif cb==Ly
                Dd=D;
                Du=1;
                T=zeros(Dl,Du,Dr,Dd,d);
                T[:,:,:,:,1]=A_set[ca,cb][1];
                T[:,:,:,:,2]=A_set[ca,cb][2];
            end
            
        elseif 1<ca<Lx
            Dl=D;
            Dr=D;
            if cb==1
                Dd=1;
                Du=D;
                T=zeros(Dl,Du,Dr,Dd,d);
                T[:,:,:,:,1]=A_set[ca,cb][1];
                T[:,:,:,:,2]=A_set[ca,cb][2];
            elseif 1<cb<Ly
                Dd=D;
                Du=D;
                T=zeros(Dl,Du,Dr,Dd,d);
                T[:,:,:,:,1]=A_set[ca,cb][1];
                T[:,:,:,:,2]=A_set[ca,cb][2];
            elseif cb==Ly
                Dd=D;
                Du=1;
                T=zeros(Dl,Du,Dr,Dd,d);
                T[:,:,:,:,1]=A_set[ca,cb][1];
                T[:,:,:,:,2]=A_set[ca,cb][2];
            end
        elseif ca==Lx
            Dl=D;
            Dr=1;
            if cb==1
                Dd=1;
                Du=D;
                T=zeros(Dl,Du,Dr,Dd,d);
                T[:,:,:,:,1]=A_set[ca,cb][1];
                T[:,:,:,:,2]=A_set[ca,cb][2];
            elseif 1<cb<Ly
                Dd=D;
                Du=D;
                T=zeros(Dl,Du,Dr,Dd,d);
                T[:,:,:,:,1]=A_set[ca,cb][1];
                T[:,:,:,:,2]=A_set[ca,cb][2];
            elseif cb==Ly
                Dd=D;
                Du=1;
                T=zeros(Dl,Du,Dr,Dd,d);
                T[:,:,:,:,1]=A_set[ca,cb][1];
                T[:,:,:,:,2]=A_set[ca,cb][2];
            end
        end
        Tnew=TensorMap(T,(ℂ^Dl)*(ℂ^Du)*(ℂ^Dr)'*(ℂ^Dd)',(ℂ^d));
        Tnew=permute(Tnew,(1,4,3,2,5,));
        #Tnew=permute(Tnew,(1,2,3,4,5,));
        psi[ca,cb]=Tnew;
    end
end



filenm="WYLiu_D"*string(D)*".jld2";
jldsave(filenm;psi)