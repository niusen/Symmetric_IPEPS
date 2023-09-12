using MAT
using TensorKit
function construct_tensor(D)
    #D=3
    filenm="bond_tensors_D_"*string(D)*".mat"
    vars = matread(filenm)
    A_set=vars["A_set"][1,:]
    A_set_occu=Vector(undef,length(A_set))
    S_label=vars["S_label"][1,:]
    Sz_label=vars["Sz_label"][1,:]
    virtual_particle=vars["virtual_particle"][1,:]
    #typeof(A_set[1]["tensor"])


    filenm="square_tensors_D_"*string(D)*".mat"
    vars = matread(filenm)
    A1_set=vars["A1_set"]
    A1_set=A1_set[1,:]
    A1_set_occu=Vector(undef,length(A1_set))
    A2_set=vars["A2_set"]
    if length(A2_set)>0
        A2_set=A2_set[1,:]
    end
    A2_set_occu=Vector(undef,length(A2_set))

    B1_set=vars["B1_set"]
    B1_set=B1_set[1,:]
    B1_set_occu=Vector(undef,length(B1_set))
    B2_set=vars["B2_set"]
    if length(B2_set)>0
        B2_set=B2_set[1,:]
    end
    B2_set_occu=Vector(undef,length(B2_set))

    Va=[]
    Vb=[]

    for cm=1:length(A_set)
        A_set_occu[cm]=A_set[cm]["sectors"]
        T=A_set[cm]["tensor"]
        #size(T)
        #sizeof(T)
        V1=ℂ^(size(A_set[cm]["tensor"])[1])
        V2=ℂ^(size(A_set[cm]["tensor"])[2])
        V3=ℂ^(size(A_set[cm]["tensor"])[3])
        
        t1 = TensorMap(T, V1 ⊗ V2  ← V3)

        Va=SU2Space(0=>length(findall(x->x==0, S_label))/1, 1/2=>length(findall(x->x==1/2, S_label))/2,1=>length(findall(x->x==1, S_label))/3,3/2=>length(findall(x->x==3/2, S_label))/4)
        Vb=SU2Space(1=>1)
        t2 = TensorMap(T, Va ⊗ Va ← Vb);
        #print(convert(Array, t2))
        #display(convert(Array, t2))

        norm(convert(Array, t1)-convert(Array, t2))<1e-10  ? nothing : throw(AssertionError("Tensor converted incorrectly"));
        A_set[cm]=t2;
    end





        for cm=1:length(A1_set)
            A1_set_occu[cm]=A1_set[cm]["sectors"]
            T=A1_set[cm]["tensor"]

            V1=ℂ^(size(A1_set[cm]["tensor"])[1])
            V2=ℂ^(size(A1_set[cm]["tensor"])[2])
            V3=ℂ^(size(A1_set[cm]["tensor"])[3])
            V4=ℂ^(size(A1_set[cm]["tensor"])[4])
            t1 = TensorMap(T, V1 ←  V2 ⊗ V3 ⊗ V4)

            Va=SU2Space(0=>length(findall(x->x==0, S_label))/1, 1/2=>length(findall(x->x==1/2, S_label))/2,1=>length(findall(x->x==1, S_label))/3,3/2=>length(findall(x->x==3/2, S_label))/4)
            t2 = TensorMap(T, ProductSpace{GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}, 0}() ← Va ⊗ Va ⊗ Va ⊗ Va);
            
            norm(convert(Array, t1)-convert(Array, t2))<1e-10  ? nothing : throw(AssertionError("Tensor converted incorrectly"));
            A1_set[cm]=t2;
        end

        for cm=1:length(A2_set)
            A2_set_occu[cm]=A2_set[cm]["sectors"]
            T=A2_set[cm]["tensor"]

            V1=ℂ^(size(A2_set[cm]["tensor"])[1])
            V2=ℂ^(size(A2_set[cm]["tensor"])[2])
            V3=ℂ^(size(A2_set[cm]["tensor"])[3])
            V4=ℂ^(size(A2_set[cm]["tensor"])[4])
            t1 = TensorMap(T, V1 ←  V2 ⊗ V3 ⊗ V4)

            Va=SU2Space(0=>length(findall(x->x==0, S_label))/1, 1/2=>length(findall(x->x==1/2, S_label))/2,1=>length(findall(x->x==1, S_label))/3,3/2=>length(findall(x->x==3/2, S_label))/4)
            t2 = TensorMap(T, ProductSpace{GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}, 0}() ← Va ⊗ Va ⊗ Va ⊗ Va);
            
            norm(convert(Array, t1)-convert(Array, t2))<1e-10  ? nothing : throw(AssertionError("Tensor converted incorrectly"));
            A2_set[cm]=t2;
        end

        
        for cm=1:length(B1_set)
            B1_set_occu[cm]=B1_set[cm]["sectors"]
            T=B1_set[cm]["tensor"]

            V1=ℂ^(size(B1_set[cm]["tensor"])[1])
            V2=ℂ^(size(B1_set[cm]["tensor"])[2])
            V3=ℂ^(size(B1_set[cm]["tensor"])[3])
            V4=ℂ^(size(B1_set[cm]["tensor"])[4])
            t1 = TensorMap(T, V1 ←  V2 ⊗ V3 ⊗ V4)

            Va=SU2Space(0=>length(findall(x->x==0, S_label))/1, 1/2=>length(findall(x->x==1/2, S_label))/2,1=>length(findall(x->x==1, S_label))/3,3/2=>length(findall(x->x==3/2, S_label))/4)
            t2 = TensorMap(T, ProductSpace{GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}, 0}() ← Va ⊗ Va ⊗ Va ⊗ Va);
            
            norm(convert(Array, t1)-convert(Array, t2))<1e-10  ? nothing : throw(AssertionError("Tensor converted incorrectly"));
            B1_set[cm]=t2;
        end

        for cm=1:length(B2_set)
            B2_set_occu[cm]=B2_set[cm]["sectors"]
            T=B2_set[cm]["tensor"]

            V1=ℂ^(size(B2_set[cm]["tensor"])[1])
            V2=ℂ^(size(B2_set[cm]["tensor"])[2])
            V3=ℂ^(size(B2_set[cm]["tensor"])[3])
            V4=ℂ^(size(B2_set[cm]["tensor"])[4])
            t1 = TensorMap(T, V1 ←  V2 ⊗ V3 ⊗ V4)

            Va=SU2Space(0=>length(findall(x->x==0, S_label))/1, 1/2=>length(findall(x->x==1/2, S_label))/2,1=>length(findall(x->x==1, S_label))/3,3/2=>length(findall(x->x==3/2, S_label))/4)
            t2 = TensorMap(T, ProductSpace{GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}, 0}() ← Va ⊗ Va ⊗ Va ⊗ Va);
            
            norm(convert(Array, t1)-convert(Array, t2))<1e-10  ? nothing : throw(AssertionError("Tensor converted incorrectly"));
            B2_set[cm]=t2;
        end
        return A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb;
    
    
end