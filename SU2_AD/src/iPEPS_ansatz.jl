
abstract type iPEPS_ansatz end
abstract type iPEPS_ansatz_immutable end

mutable struct Kagome_iPESS <: iPEPS_ansatz #this is for line search. Don't use this for AD, otherwise the grad will be incorrect.
    B1::TensorMap
    B2::TensorMap
    B3::TensorMap
    Tup::TensorMap
    Tdn::TensorMap

end

struct Kagome_iPESS_immutable <: iPEPS_ansatz_immutable #this is for AD
    B1::TensorMap
    B2::TensorMap
    B3::TensorMap
    Tup::TensorMap
    Tdn::TensorMap

end

function Kagome_iPESS_convert(ansatz::Kagome_iPESS)
    ansatz_new=Kagome_iPESS_immutable(ansatz.B1,ansatz.B2,ansatz.B3,ansatz.Tup,ansatz.Tdn);
    return ansatz_new
end
# function Kagome_iPESS_convert(ansatz::Kagome_iPESS_immutable)
#     ansatz_new=Kagome_iPESS(ansatz.B1,ansatz.B2,ansatz.B3,ansatz.Tup,ansatz.Tdn);
#     return ansatz_new
# end



function Base.similar(x::iPEPS_ansatz)
    return deepcopy(x)
end


