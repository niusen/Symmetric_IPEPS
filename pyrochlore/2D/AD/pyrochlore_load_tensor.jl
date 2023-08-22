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

    filenm="tetrahedral_tensors_D_"*string(D)*".mat"
    vars = matread(filenm)
    E_set=vars["E_set"][1,:]
    E_set_occu=Vector(undef,length(E_set))
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



    for cm=1:length(E_set)
        E_set_occu[cm]=E_set[cm]["sectors"]
        T=E_set[cm]["tensor"]

        V1=ℂ^(size(E_set[cm]["tensor"])[1])
        V2=ℂ^(size(E_set[cm]["tensor"])[2])
        V3=ℂ^(size(E_set[cm]["tensor"])[3])
        V4=ℂ^(size(E_set[cm]["tensor"])[4])
        t1 = TensorMap(T, V1 ←  V2 ⊗ V3 ⊗ V4)

        Va=SU2Space(0=>length(findall(x->x==0, S_label))/1, 1/2=>length(findall(x->x==1/2, S_label))/2,1=>length(findall(x->x==1, S_label))/3,3/2=>length(findall(x->x==3/2, S_label))/4)
        t2 = TensorMap(T, ProductSpace{GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}, 0}() ← Va ⊗ Va ⊗ Va ⊗ Va);
        
        norm(convert(Array, t1)-convert(Array, t2))<1e-10  ? nothing : throw(AssertionError("Tensor converted incorrectly"));
        E_set[cm]=t2;
    end

    
    return A_set,E_set, S_label, Sz_label, virtual_particle, Va, Vb;
end