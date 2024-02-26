
abstract type iPEPS_ansatz end
abstract type iPEPS_ansatz_immutable end

function Base.similar(x::iPEPS_ansatz)
    return deepcopy(x)
end

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




Base.@kwdef mutable struct Kagome_Energy_settings
    kagome_method :: String = "E_single_triangle";# "E_single_triangle", "E_triangle", "J2J3", "E_bond"
    E_up_method :: String = "1x1";#"1x1", "2x2"
    E_dn_method :: String = "simplified";#"open_leg", "simplfied"
    cal_chiral_order :: Bool = false;

end


##################################################

mutable struct Checkerboard_iPESS <: iPEPS_ansatz #this is for line search. Don't use this for AD, otherwise the grad will be incorrect.
    B_L::TensorMap
    B_U::TensorMap
    Tm::TensorMap

end

struct Checkerboard_iPESS_immutable <: iPEPS_ansatz_immutable #this is for AD
    B_L::TensorMap
    B_U::TensorMap
    Tm::TensorMap

end




function Checkerboard_iPESS_convert(ansatz::Checkerboard_iPESS)
    ansatz_new=Checkerboard_iPESS_immutable(ansatz.B_L,ansatz.B_U,ansatz.Tm);
    return ansatz_new
end

# Base.@kwdef mutable struct Checkerboard_Energy_settings
#     kagome_method :: String = "E_single_triangle";# "E_single_triangle", "E_triangle", "J2J3", "E_bond"
#     E_up_method :: String = "1x1";#"1x1", "2x2"
#     E_dn_method :: String = "simplified";#"open_leg", "simplfied"
#     cal_chiral_order :: Bool = false;

# end

##################################################

mutable struct Triangle_iPESS <: iPEPS_ansatz #this is for line search. Don't use this for AD, otherwise the grad will be incorrect.
    Bm::TensorMap
    Tm::TensorMap
end

struct Triangle_iPESS_immutable <: iPEPS_ansatz_immutable #this is for AD
    Bm::TensorMap
    Tm::TensorMap
end




function Triangle_iPESS_convert(ansatz::Triangle_iPESS)
    ansatz_new=Triangle_iPESS_immutable(ansatz.Bm,ansatz.Tm);
    return ansatz_new
end

##################################################

mutable struct Square_iPEPS <: iPEPS_ansatz #this is for line search. Don't use this for AD, otherwise the grad will be incorrect.
    T::TensorMap
end

struct Square_iPEPS_immutable <: iPEPS_ansatz_immutable #this is for AD
    T::TensorMap
end




function Square_iPEPS_convert(ansatz::Square_iPEPS)
    ansatz_new=Square_iPEPS_immutable(ansatz.T);
    return ansatz_new
end



Base.@kwdef mutable struct Square_Hubbard_Energy_settings
    model :: String = "spinless_Hubbard"

end

Base.@kwdef mutable struct Square_Energy_settings
    model :: String = "triangle_J1_J2_Jchi"

end

Base.@kwdef mutable struct Square_2site_Energy_settings
    model :: String = "triangle_J1_J2_Jchi"
    print_all_terms :: Bool=false

end








