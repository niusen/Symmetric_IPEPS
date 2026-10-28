using LinearAlgebra, TensorKit, TensorKitSectors, SUNRepresentations
using JLD2
include("../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
using .HoneycombSU4

function main(;symmetry=:dense)
    Lx=2; Ly=2;
    D_max=symmetry==:dense ? 4 : 15;
    parameters=Dict("J"=>1.0);
    convention=:traceless; # H = J sum(P_ij-I/4)

    if symmetry==:dense
        T_set,lambdax_set,lambday_set=initial_honeycomb_iPEPS(Lx,Ly;D=2,seed=1234);
    else
        # Initial quantum numbers and MULTIPLICITIES (not dimensions).
        # D_initial=1+4+6+4=15. All physical sites are the fundamental 4.
        V=honeycomb_virtual_space(["1"=>1,"4"=>1,"6"=>1,"4⁺"=>1]);
        # To choose each direction separately: V=(x=V,y=V,z=V).
        T_set,lambdax_set,lambday_set=initial_honeycomb_iPEPS(Lx,Ly;
            symmetry=:SU4,V=V,seed=1234);
    end

    for (tau,dt) in ((1.0,0.1),(1.0,0.02),(0.2,0.002))
        T_set,lambdax_set,lambday_set=itebd_honeycomb(parameters,
            T_set,lambdax_set,lambday_set,tau,dt,D_max;convention=convention);
        ob=honeycomb_energy_simple(parameters,T_set,lambdax_set,lambday_set;convention=convention);
        println("simple-lambda estimate: E/site=",ob.energy_per_site,", bonds=",ob.bond_energy);
    end

    filename=joinpath(@__DIR__,"SU_honeycomb_$(symmetry)_D$(D_max).jld2");
    jldsave(filename;T_set,lambdax_set,lambday_set,parameters,convention,symmetry,D_max);

    ctm_setting=LS_CTMRG_settings();
    ctm_setting.CTM_ite_nums=100;
    ctm_setting.CTM_conv_tol=1e-7;
    ctm_setting.grad_checkpoint=false;
    CTM0=nothing;
    for chi in (16,24,32)
        ob=honeycomb_energy_CTM(parameters,T_set,chi,ctm_setting;
            convention=convention,CTM0=CTM0);
        CTM0=ob.CTM;
        println("CTMRG: chi=",chi,", E/site=",ob.energy_per_site,
            ", bonds=",ob.bond_energy,", corner_converged=",ob.corner_converged);
    end
    return T_set,lambdax_set,lambday_set
end

if abspath(PROGRAM_FILE)==@__FILE__
    main(;symmetry=isempty(ARGS) ? :dense : Symbol(ARGS[1]));
end
