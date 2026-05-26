using TensorKit
include("../../src/bosonic/square/square_RVB_ansatz.jl")
A1a,A1b,A2 = D2_point_group_symmetric_tensors()
B1a,B1b,B2 = D2_point_group_symmetric_tensors()
norm(A1b - B1b)