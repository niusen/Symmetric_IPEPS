
abstract type iPEPS_ansatz end

mutable struct Kagome_iPESS <: iPEPS_ansatz
    B1::TensorMap
    B2::TensorMap
    B3::TensorMap
    Tup::TensorMap
    Tdn::TensorMap

end
function Base.similar(x::iPEPS_ansatz)
    return deepcopy(x)
end


