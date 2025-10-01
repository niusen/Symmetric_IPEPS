function SU2space_to_Z2space(V)
    Dim=dim(V);
    Parity=zeros(1,Dim);
    ind=0;
    for cc =1:length(V.dims.values)
        Spin=V.dims.keys[cc].j;
        Spin_dim=Int(2*Spin+1);
        if mod(Spin_dim,2)==1#integer spin
            Parity[ind+1:ind+Spin_dim*V.dims.values[cc]].=0
        elseif mod(Spin_dim,2)==0#half integer spin
            Parity[ind+1:ind+Spin_dim*V.dims.values[cc]].=1
        end
        ind=ind+Spin_dim*V.dims.values[cc];
    end
    dim_even=findall(x->x.==0,Parity[:]);
    dim_odd=findall(x->x.==1,Parity[:]);
    @assert length(dim_even)+length(dim_odd)==Dim;
    Vnew=Rep[ℤ₂](0=>length(dim_even),1=>length(dim_odd));
    if V.dual
        Vnew=Vnew';
    end
    return Vnew, dim_even,dim_odd
end

function convert_SU2_to_Z2(T::TensorMap)
    if Rank(T)==3
        v1,dim_even1,dim_odd1=SU2space_to_Z2space(space(T,1));
        v2,dim_even2,dim_odd2=SU2space_to_Z2space(space(T,2));
        v3,dim_even3,dim_odd3=SU2space_to_Z2space(space(T,3));

        T_dense=convert(Array,T);

        dim_even=deepcopy(dim_even1);
        dim_odd=deepcopy(dim_odd1);
        T_dense[1:length(dim_even),:,:]=T_dense[dim_even,:,:];
        T_dense[length(dim_even)+1:length(dim_even)+length(dim_odd),:,:]=T_dense[dim_odd,:,:];

        dim_even=deepcopy(dim_even2);
        dim_odd=deepcopy(dim_odd2);
        T_dense[:,1:length(dim_even),:]=T_dense[:,dim_even,:];
        T_dense[:,length(dim_even)+1:length(dim_even)+length(dim_odd),:]=T_dense[:,dim_odd,:];

        dim_even=deepcopy(dim_even3);
        dim_odd=deepcopy(dim_odd3);
        T_dense[:,:,1:length(dim_even)]=T_dense[:,:,dim_even];
        T_dense[:,:,length(dim_even)+1:length(dim_even)+length(dim_odd)]=T_dense[:,:,dim_odd];

        T_new=TensorMap(T_dense,v1*v2,v3');

        if (length(codomain(T))==2)&&(length(domain(T))==1)
            T_new=permute(T_new,(1,2,),(3,));
        else
            error("unknown case");
        end
    elseif Rank(T)==4
        v1,dim_even1,dim_odd1=SU2space_to_Z2space(space(T,1));
        v2,dim_even2,dim_odd2=SU2space_to_Z2space(space(T,2));
        v3,dim_even3,dim_odd3=SU2space_to_Z2space(space(T,3));
        v4,dim_even4,dim_odd4=SU2space_to_Z2space(space(T,4));

        T_dense=convert(Array,T);

        dim_even=deepcopy(dim_even1);
        dim_odd=deepcopy(dim_odd1);
        T_dense[1:length(dim_even),:,:,:]=T_dense[dim_even,:,:,:];
        T_dense[length(dim_even)+1:length(dim_even)+length(dim_odd),:,:,:]=T_dense[dim_odd,:,:,:];

        dim_even=deepcopy(dim_even2);
        dim_odd=deepcopy(dim_odd2);
        T_dense[:,1:length(dim_even),:,:]=T_dense[:,dim_even,:,:];
        T_dense[:,length(dim_even)+1:length(dim_even)+length(dim_odd),:,:]=T_dense[:,dim_odd,:,:];

        dim_even=deepcopy(dim_even3);
        dim_odd=deepcopy(dim_odd3);
        T_dense[:,:,1:length(dim_even),:]=T_dense[:,:,dim_even,:];
        T_dense[:,:,length(dim_even)+1:length(dim_even)+length(dim_odd),:]=T_dense[:,:,dim_odd,:];

        dim_even=deepcopy(dim_even4);
        dim_odd=deepcopy(dim_odd4);
        T_dense[:,:,:,1:length(dim_even)]=T_dense[:,:,:,dim_even];
        T_dense[:,:,:,length(dim_even)+1:length(dim_even)+length(dim_odd)]=T_dense[:,:,:,dim_odd];

        T_new=TensorMap(T_dense,v1,v2'*v3'*v4');

        if (length(codomain(T))==1)&&(length(domain(T))==3)
            T_new=permute(T_new,(1,),(2,3,4,));
        else
            error("unknown case");
        end
    end

    return T_new
end


