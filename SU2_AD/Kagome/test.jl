using Revise,TensorOperations
using LinearAlgebra, TensorKit, Zygote
using JLD2,ChainRulesCore
using JSON
using Random
using Zygote:@ignore_derivatives
using Dates
using TensorKitAD

A=TensorMap(randn, Rep[SU₂](0=>1),Rep[SU₂](0=>1))
# function Cos(A)
#   y=dot(A,A)
#   return real(y)
# end
function Cos(A)
  m=A.data.values[1];
  y=dot(m,m)
  return real(y)
end
∂E=Cos'(A)
typeof(∂E)
