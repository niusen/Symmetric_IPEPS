using AutoFermionicPESS
using TensorKit

A_cell = Matrix{Any}(undef, 1, 1)
A_cell[1, 1] = pess_to_ipeps_tensor(random_triangle_pess(1, 1))

setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)

V = physical_spinful_space()
id3 = TensorKit.id(V * V * V)

println("triangle identity up = ", graded_ob_triangle_2x2(setup.CTM, id3, A_cell, setup.double_layer.AA, 1, 1; orientation=:up))
println("triangle identity down = ", graded_ob_triangle_2x2(setup.CTM, id3, A_cell, setup.double_layer.AA, 1, 1; orientation=:down))
