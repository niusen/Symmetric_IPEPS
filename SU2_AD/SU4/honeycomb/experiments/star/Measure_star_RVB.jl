using LinearAlgebra, Random, TensorKit, SUNRepresentations, JLD2, Test
include("../../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
using .HoneycombSU4

# Compatibility entry for the existing measurement / audit scripts.
function initial_star_RVB(;Lx=2,Ly=2,centers=nothing)
    T,_,_,info=initial_honeycomb_star_iPEPS(;Lx,Ly,centers);
    return T,info
end

function measure_star_RVB(;chis=(16,32,48),L=4)
    BLAS.set_num_threads(4);
    T,info=initial_star_RVB(Lx=L,Ly=L);
    println("STAR INITIAL ",info);flush(stdout);
    out=joinpath(@__DIR__,"../../data","star_RVB");mkpath(out);
    setting=LS_CTMRG_settings(CTM_ite_nums=100,CTM_conv_tol=1e-7,
        CTM_ite_info=true,CTM_conv_info=true,grad_checkpoint=false);
    results=NamedTuple[];
    for ch in chis
        println("START chi=",ch);flush(stdout);
        ob=honeycomb_energy_CTM(Dict("J"=>1.0),T,ch,setting;convention=:permutation);
        row=(L=L,chi=ch,E_P=ob.energy_per_site,P_x=ob.bond_energy.x,
            P_y=ob.bond_energy.y,P_z=ob.bond_energy.z,
            bond_spread=maximum(ob.energies)-minimum(ob.energies),
            min_bond_P=minimum(ob.energies),max_bond_P=maximum(ob.energies),
            physical_bond_bounds=all(-1-1e-9<=v<=1+1e-9 for v in ob.energies),
            corner_error=ob.corner_error,iterations=ob.iterations,converged=ob.corner_converged);
        push!(results,row);println("STAR RESULT ",row," bonds=",ob.energies);flush(stdout);
        jldsave(joinpath(out,"star_L$(L)_chi$(ch).jld2");T_set=T,info,results,CTM=ob.CTM);
        open(joinpath(out,"results_L$(L).csv"),"w") do io
            println(io,join(keys(first(results)),","));
            for r in results; println(io,join(values(r),","));end
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    measure_star_RVB();
end
