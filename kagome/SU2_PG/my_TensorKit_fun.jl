using TensorKit

# file = matopen("matfile.mat", "w")
# write(file, "varname", variable)
# close(file)
# function fuse_legs(A,Ndim,ind1,ind2, Adjoint::Bool=false, Check::Bool=false)
#     if Check==true
#         A_dense=convert(Array,A);
#     end

#     if Adjoint==true
#         u=unitary(fuse(space(A, ind1) ⊗ space(A, ind2))', space(A, ind1) ⊗ space(A, ind2))
#     elseif Adjoint==false
#         u=unitary(fuse(space(A, ind1) ⊗ space(A, ind2)), space(A, ind1) ⊗ space(A, ind2))
#     end

#     @tensor PEPS_tensor[:] := A[-1,1,-5]*u[4,3,-6];




# end