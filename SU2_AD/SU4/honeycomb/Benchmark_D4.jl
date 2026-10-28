using LinearAlgebra, JLD2, Printf
BLAS.set_num_threads(1);
include("../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
using .HoneycombSU4

function bond_spectra(lx,ly)
    return [sort(abs2.(diag(convert(Array,s)));rev=true) for s in vcat(vec(lx),vec(ly))]
end
function spectrum_change(a,b)
    return maximum(length(x)==length(y) ? norm(x-y) : Inf for (x,y) in zip(a,b))
end

function main(;seeds=(1234,2025,42),Lx=2,Ly=2)
    output=joinpath(@__DIR__,"benchmark_D4_$(Lx)x$(Ly)"); mkpath(output);
    parameters=Dict("J"=>1.0);
    trace_path=joinpath(output,"evolution.csv");
    ctm_path=joinpath(output,"ctm.csv");
    states=Dict{Int,Any}(); measurements=Dict{Int,Any}();
    open(trace_path,"w") do io
        println(io,"seed,dt,tau_total,E_site_traceless,E_site_permutation,delta_E,delta_spectrum");
    end
    open(ctm_path,"w") do io
        println(io,"seed,chi,iterations,corner_error,E_site_traceless,E_site_permutation,P_x,P_y,P_z");
    end
    for seed in seeds
        T,lx,ly=initial_honeycomb_iPEPS(Lx,Ly;D=2,seed=seed);
        previous=honeycomb_energy_simple(parameters,T,lx,ly).energy_per_site;
        elapsed_tau=0.0; trace=NamedTuple[];
        schedule=[(20.0,0.1),(10.0,0.05),(10.0,0.02),(10.0,0.01),
                  (5.0,0.005),(2.0,0.002),(1.0,0.001)];
        # Several final low-step-size chunks test remaining SU relaxation.
        append!(schedule,fill((2.0,0.001),5));
        for (stage,(tau,dt)) in enumerate(schedule)
            old_spectra=bond_spectra(lx,ly);
            itebd_honeycomb(parameters,T,lx,ly,tau,dt,4;verbose=false);
            ob=honeycomb_energy_simple(parameters,T,lx,ly);
            elapsed_tau+=tau;
            delta=abs(ob.energy_per_site-previous);
            ds=spectrum_change(old_spectra,bond_spectra(lx,ly));
            row=(seed=seed,dt=dt,tau_total=elapsed_tau,E_site_traceless=ob.energy_per_site,
                E_site_permutation=ob.energy_per_site+3/8,delta_E=delta,delta_spectrum=ds);
            push!(trace,row);
            open(trace_path,"a") do io
                println(io,join(values(row),","));
            end
            @printf("seed=%d tau=%.1f dt=%.3g local E(P)/site=%.10f dE=%.3g ds=%.3g\n",
                seed,elapsed_tau,dt,row.E_site_permutation,delta,ds);flush(stdout);
            previous=ob.energy_per_site;
            jldsave(joinpath(output,"seed_$(seed).jld2");T_set=T,lambdax_set=lx,lambday_set=ly,
                parameters,D_max=4,seed,Lx,Ly,trace);
            stage>length(schedule)-5 && delta<1e-7 && ds<1e-5 && break;
        end
        states[seed]=(T=T,lx=lx,ly=ly);
    end
    BLAS.set_num_threads(4);
    setting=LS_CTMRG_settings(CTM_ite_nums=80,CTM_conv_tol=1e-6,
        CTM_ite_info=false,CTM_conv_info=true,grad_checkpoint=false);
    function measure(seed,ch,env)
        ob=honeycomb_energy_CTM(parameters,states[seed].T,ch,setting;CTM0=env);
        total_iterations=ob.iterations;
        if !ob.corner_converged
            ob=honeycomb_energy_CTM(parameters,states[seed].T,ch,setting;CTM0=ob.CTM);
            total_iterations+=ob.iterations;
        end
        row=(seed=seed,chi=ch,iterations=total_iterations,corner_error=ob.corner_error,
            E_site_traceless=ob.energy_per_site,E_site_permutation=ob.energy_per_site+3/8,
            P_x=ob.bond_energy.x+1/4,P_y=ob.bond_energy.y+1/4,P_z=ob.bond_energy.z+1/4);
        open(ctm_path,"a") do io
            println(io,join(values(row),","));
        end
        println("CTM ",row);flush(stdout);
        return ob
    end
    for seed in seeds
        measurements[seed]=measure(seed,16,nothing);
    end
    best_seed=seeds[argmin([measurements[s].energy_per_site for s in seeds])];
    best=measurements[best_seed];
    for ch in (24,32,48)
        previous=best.energy_per_site;
        best=measure(best_seed,ch,best.CTM);
        ch>=32 && best.corner_converged && abs(best.energy_per_site-previous)<1e-7 && break;
    end
    jldsave(joinpath(output,"best_result.jld2");best_seed,ob=best,
        state=states[best_seed],parameters,D_max=4,Lx,Ly);
    println("BEST seed=",best_seed," E(P)/site=",best.energy_per_site+3/8);
    return best
end

"""Resume CTMRG from a saved SU state; SU evolution is not repeated.
Record every measurement so iteration and chi changes can both be checked.
"""
function measure_saved_D4(;seed=1234,chis=(8,16,24,32),iterations=80)
    BLAS.set_num_threads(4);
    output=joinpath(@__DIR__,"benchmark_D4_2x2");
    data=load(joinpath(output,"seed_$(seed).jld2"));
    T=data["T_set"]; parameters=data["parameters"];
    setting=LS_CTMRG_settings(CTM_ite_nums=iterations,CTM_conv_tol=1e-6,
        CTM_ite_info=false,CTM_conv_info=true,grad_checkpoint=false);
    path=joinpath(output,"ctm_refined.csv");
    open(path,"w") do io
        println(io,"seed,chi,round,iterations,corner_error,E_site_traceless,E_site_permutation,P_x,P_y,P_z");
    end
    env=nothing; ob=nothing; previous_chi=NaN;
    for ch in chis
        previous_round=NaN;
        for round=1:3
            ob=honeycomb_energy_CTM(parameters,T,ch,setting;CTM0=env);
            env=ob.CTM;
            row=(seed=seed,chi=ch,round=round,iterations=ob.iterations,corner_error=ob.corner_error,
                E_site_traceless=ob.energy_per_site,E_site_permutation=ob.energy_per_site+3/8,
                P_x=ob.bond_energy.x+1/4,P_y=ob.bond_energy.y+1/4,P_z=ob.bond_energy.z+1/4);
            open(path,"a") do io
                println(io,join(values(row),","));
            end
            println("REFINED ",row);flush(stdout);
            jldsave(joinpath(output,"ctm_refined.jld2");seed,ob,T_set=T,parameters);
            (ob.corner_converged || abs(ob.energy_per_site-previous_round)<1e-7) && break;
            previous_round=ob.energy_per_site;
        end
        ch>=24 && abs(ob.energy_per_site-previous_chi)<1e-6 && break;
        previous_chi=ob.energy_per_site;
    end
    return ob
end

if abspath(PROGRAM_FILE)==@__FILE__
    main();
end
