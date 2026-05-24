using LinearAlgebra
using TensorKit
using AutoFermionicPESS

function TensorKit.permute(
    t::TensorKit.AbstractTensorMap,
    codomain_order::Tuple,
    domain_order::Tuple;
    copy::Bool=false,
)
    return TensorKit.permute(t, (codomain_order, domain_order); copy=copy)
end

@eval Main begin
    function eigh(t::TensorKit.AbstractTensorMap)
        return eigen(t)
    end
end

try
    using Zygote: @ignore_derivatives
catch err
    error("This comparison script needs Zygote because the legacy files use @ignore_derivatives. Run `import Pkg; Pkg.add(\"Zygote\")` in an environment that is allowed to use the old code. Original error: $err")
end

const AUTO_DIR = normpath(joinpath(@__DIR__, ".."))
const SU2_AD_SRC = normpath(joinpath(AUTO_DIR, ".."))

include(joinpath(SU2_AD_SRC, "fermionic", "swap_funs.jl"))
include(joinpath(SU2_AD_SRC, "fermionic", "square_Hubbard_model_cell.jl"))
include(joinpath(SU2_AD_SRC, "fermionic", "simple_update", "fermi_triangle_SimpleUpdate.jl"))

Base.@kwdef mutable struct LegacySpinHallSetting
    model::String = "Triangle_Hofstadter_Hubbard_spinHall"
    Magnetic_cell::Int = 2
end

function legacy_parameter_dict(p::SpinHallParameters)
    mu_key = String([Char(0x03BC)])
    return Dict(
        "t1" => p.t1,
        "t2" => p.t2,
        mu_key => p.mu,
        "U" => p.U,
        "B" => p.B,
        "mx" => p.mx,
        "mx_type" => p.mx_type,
    )
end

function many_site_dense_to_sector_basis(M::AbstractMatrix, nsites::Int)
    local_order = collect(AutoFermionicPESS.SPINFUL_SECTOR_ORDER)
    order = Int[]
    for sector_index0 in 0:(4^nsites - 1)
        dense_index0 = 0
        stride = 1
        x = sector_index0
        for _ in 1:nsites
            sector_digit = mod(x, 4) + 1
            dense_digit = local_order[sector_digit] - 1
            dense_index0 += dense_digit * stride
            stride *= 4
            x = fld(x, 4)
        end
        push!(order, dense_index0 + 1)
    end
    return M[order, order]
end

function compare_legacy_spin_hall_gate(;
    Lx::Int=2,
    Ly::Int=2,
    px::Int=1,
    py::Int=1,
    dt::Float64=0.01,
    pars::SpinHallParameters=SpinHallParameters(t1=1.0, t2=1.0, mu=0.0, U=0.0, mx=0.0, B=0.0),
)
    setting = SpinHallEnergySetting(Lx=Lx, Ly=Ly, magnetic_cell=2)
    coeffs = spin_hall_coefficients(setting, pars)
    new_dense = triangle_gate_dense(coeffs, px, py, dt)
    new_legacy_weight = triangle_gate_dense(
        coeffs,
        px,
        py,
        dt;
        hopping_weight=0.5,
        onsite_weight=1 / 3,
    )

    legacy_setting = LegacySpinHallSetting(Magnetic_cell=setting.magnetic_cell)
    legacy_params = legacy_parameter_dict(pars)
    legacy_space_type = GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
    legacy_gates = gate_RU_LD_RD_Hofstadter_spinHall(
        legacy_setting,
        legacy_params,
        dt,
        legacy_space_type,
        Lx,
        Ly,
    )
    legacy_dense = reshape(convert(Array, legacy_gates[px, py]), 64, 64)

    # TensorKit stores Z2/FermionParity sectors as even-even states followed by
    # odd states. The new dense Hamiltonian is written in physical Fock order.
    new_sector = many_site_dense_to_sector_basis(new_dense, 3)
    new_legacy_weight_sector = many_site_dense_to_sector_basis(new_legacy_weight, 3)

    println("legacy gate size: ", size(legacy_dense))
    println("new dense gate size: ", size(new_dense))
    println("norm(new_dense - legacy_dense): ", norm(new_dense - legacy_dense))
    println("norm(dense_to_sector_basis(new_dense) - legacy_dense): ", norm(new_sector - legacy_dense))
    println("norm(legacy_weight_sector - legacy_dense): ", norm(new_legacy_weight_sector - legacy_dense))
    println("relative sector-basis difference: ", norm(new_sector - legacy_dense) / max(norm(legacy_dense), eps()))
    println("relative legacy-weight difference: ", norm(new_legacy_weight_sector - legacy_dense) / max(norm(legacy_dense), eps()))

    return (
        legacy_dense = legacy_dense,
        new_dense = new_dense,
        new_sector = new_sector,
        new_legacy_weight = new_legacy_weight,
        new_legacy_weight_sector = new_legacy_weight_sector,
    )
end

compare_legacy_spin_hall_gate()
