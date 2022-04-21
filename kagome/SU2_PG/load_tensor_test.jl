using MAT
using TensorKit
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\kagome\\SU2_PG")

D=3
filenm="bond_tensors_D_"*string(D)*".mat"
vars = matread(filenm)
A_set=vars["A_set"]
#typeof(A_set[1]["tensor"])

A=A_set[1]["tensor"]
#size(A)
#sizeof(A)

V1=ℂ^3
V2=ℂ^3
V3=ℂ^2
t1 = TensorMap(A, V1 ⊗ V2, V3)

#A[1,2,1]=A[1,2,1]+1e-10
Va=SU2Space(0=>1, 1/2=>1)
Vb=SU2Space(1/2=>1)
t2 = TensorMap(A, Va ⊗ Va, Vb)
print(convert(Array, t2))
display(convert(Array, t2))
