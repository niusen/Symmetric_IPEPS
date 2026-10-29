using LinearAlgebra, TensorKit, SUNRepresentations, JLD2, Printf
if !isdefined(@__MODULE__,:HoneycombSU4)
    include("../../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
end
using .HoneycombSU4
const HC=HoneycombSU4;

# Experimental fixed-space control. Reuse the original two-site contractions;
# only replace their total-rank SVD truncation by a per-irrep space bound.
if !isdefined(@__MODULE__,:FixedVirtualSpace)
    struct FixedVirtualSpace{S}
        V::S
    end
end
HC.truncdim(spec::FixedVirtualSpace)=TensorKit.truncspace(spec.V);

function fixed_space_steps(parameters,T,lx,ly,steps,dt,V)
    H=Hamiltonian_honeycomb_SU4(space(T[1],5);J=parameters["J"]);
    gate,half=HC.trotter_gate(H,dt);
    spec=FixedVirtualSpace(V);
    directions=(:x,:y,:z);
    for ct=1:steps
        a,b,c=ntuple(k->directions[mod1(k+ct-1,3)],3);
        for (direction,g) in ((a,half),(b,half),(c,gate),(b,half),(a,half))
            for bond in honeycomb_bonds(2,2)
                bond.direction==direction || continue;
                x,y=bond.site1;
                if bond.axis==:horizontal
                    HC.tebd_xbond(1,T,lx,ly,g,x+0.5,y,spec);
                else
                    HC.tebd_ybond(1,T,lx,ly,g,x,y-0.5,spec);
                end
            end
        end
    end
    return T,lx,ly
end

# Exploratory tests of the existing unrestricted multiplet truncation.
# No C3 projection or minimum quota for any center-charge class is imposed.
function torus_norm(T)
    @tensor psi[-1,-2,-3,-4] := T[1,1][b,e,a,g,-1]*T[2,1][a,h,b,f,-2]*
        T[1,2][d,g,c,e,-3]*T[2,2][c,f,d,h,-4];
    return norm(psi)
end

function real_bond_spectra(lx,ly)
    return [bond.axis==:horizontal ? lx[bond.site2...] : ly[bond.site1...]
        for bond in honeycomb_bonds(2,2)]
end

function reduced_spectra(lx,ly)
    return [Dict(Tuple(Int.(SUNRepresentations.dynkin_label(c)))=>sort(abs2.(diag(b));rev=true)
        for (c,b) in blocks(s)) for s in real_bond_spectra(lx,ly)]
end

function spectrum_error(a,b)
    error=0.0;
    for (x,y) in zip(a,b)
        keys(x)==keys(y) || return Inf;
        for key in keys(x)
            length(x[key])==length(y[key]) || return Inf;
            error=max(error,norm(x[key]-y[key]));
        end
    end
    return error
end

function record_sectors(path,label,seed,stage,lx,ly)
    for (bond,s) in zip(honeycomb_bonds(2,2),real_bond_spectra(lx,ly))
        total=sum(dim(c)*sum(abs2,abs2.(diag(b))) for (c,b) in blocks(s));
        for (c,b) in blocks(s)
            weight=dim(c)*sum(abs2,abs2.(diag(b)))/total;
            row=(label,seed,stage,bond.direction,bond.site1[1],bond.site1[2],
                dim(space(s,1)),join(Int.(SUNRepresentations.dynkin_label(c)),":"),
                SUNRepresentations.congruency(c),dim(c),length(diag(b)),weight);
            open(path,"a") do io
                println(io,join(row,","));
            end
        end
    end
end

function run_small_D_SU4(;seeds=(1234,2025),measure=true,truncation=:optimal,
                         selected=("D8","D9","D15"),pilot=false)
    truncation in (:optimal,:rank,:space) || throw(ArgumentError("use :optimal, :rank or :space"));
    BLAS.set_num_threads(1);
    folder=truncation==:optimal ? "small_D_SU4_optimal" :
        truncation==:rank ? "small_D_SU4" : "small_D_SU4_fixed";
    out=joinpath(@__DIR__,"../../data",folder); mkpath(out);
    tracefile=joinpath(out,"evolution.csv"); sectorfile=joinpath(out,"sectors.csv");
    ctmfile=joinpath(out,"ctm.csv"); errors=joinpath(out,"errors.txt");
    open(tracefile,"w") do io
        println(io,"case,seed,stage,dt,tau,E_P_local,delta_E,delta_spectrum,torus_norm,min_D,max_D,P_x,P_y,P_z");
    end
    open(sectorfile,"w") do io
        println(io,"case,seed,stage,direction,x,y,bond_D,dynkin,q,irrep_dim,multiplicity,schmidt_weight");
    end
    open(ctmfile,"w") do io
        println(io,"case,seed,chi,round,iterations,corner_error,E_P_CTM,P_x,P_y,P_z,bond_spread");
    end
    write(errors,"");
    cases=(("D8",8,["4"=>1,"4⁺"=>1]),
           ("D9",9,["1"=>1,"4"=>1,"4⁺"=>1]),
           ("D15",15,["1"=>1,"4"=>1,"6"=>1,"4⁺"=>1]));
    states=Dict{Tuple{String,Int},Any}();
    parameters=Dict("J"=>1.0);
    for (label,Dmax,irreps) in cases, seed in seeds
        label in selected || continue;
        println("START ",label," seed=",seed);flush(stdout);
        V=honeycomb_virtual_space(irreps);
        T,lx,ly=initial_honeycomb_iPEPS(2,2;symmetry=:SU4,V=V,seed=seed);
        previous=honeycomb_energy_simple(parameters,T,lx,ly).energy_per_site;
        time=0.0; completed=0; history=NamedTuple[];
        record_sectors(sectorfile,label,seed,0,lx,ly);
        println("INITIAL ",label," seed=",seed," local E(P)=",previous+3/8," torus norm=",torus_norm(T));flush(stdout);
        schedule=pilot ? [(3,0.1),(27,0.1),(60,0.05)] :
            vcat([(3,0.1),(27,0.1),(60,0.05),(150,0.02),(300,0.01),(300,0.005)],
                fill((300,0.005),4),fill((600,0.002),3));
        for (stage,(steps,dt)) in enumerate(schedule)
            old=deepcopy((T,lx,ly)); oldspec=reduced_spectra(lx,ly);
            try
                if truncation in (:rank,:optimal)
                    policy=truncation==:optimal ? :optimal : :legacy;
                    itebd_honeycomb(parameters,T,lx,ly,steps*dt,dt,Dmax;
                        verbose=false,truncation=policy);
                else
                    fixed_space_steps(parameters,T,lx,ly,steps,dt,V);
                end
                all(t->isfinite(norm(t)) && norm(t)>0,T) || error("nonfinite or zero tensor after SU");
                ob=honeycomb_energy_simple(parameters,T,lx,ly);
                isfinite(ob.energy_per_site) || error("nonfinite local energy");
                delta=abs(ob.energy_per_site-previous);
                ds=spectrum_error(oldspec,reduced_spectra(lx,ly));
                time+=steps*dt; completed=stage;
                dims=[dim(space(s,1)) for s in real_bond_spectra(lx,ly)];
                row=(case=label,seed=seed,stage=stage,dt=dt,tau=time,
                    E_P_local=ob.energy_per_site+3/8,delta_E=delta,delta_spectrum=ds,
                    torus_norm=torus_norm(T),min_D=minimum(dims),max_D=maximum(dims),
                    P_x=ob.bond_energy.x+1/4,P_y=ob.bond_energy.y+1/4,P_z=ob.bond_energy.z+1/4);
                push!(history,row);
                open(tracefile,"a") do io
                    println(io,join(values(row),","));
                end
                record_sectors(sectorfile,label,seed,stage,lx,ly);
                @printf("SU %s seed=%d stage=%d tau=%.2f dt=%.3g E(P)=%.10f dE=%.3g ds=%.3g D=%d..%d torus=%.3g\n",
                    label,seed,stage,time,dt,row.E_P_local,delta,ds,row.min_D,row.max_D,row.torus_norm);flush(stdout);
                jldsave(joinpath(out,"$(label)_seed$(seed).jld2");T_set=T,lambdax_set=lx,lambday_set=ly,
                    parameters,Dmax,seed,history,truncation,convention=:traceless);
                previous=ob.energy_per_site;
                # Always reach the smallest dt; require a subsequent equal-dt check.
                dt==0.002 && stage>=12 && delta<1e-6 && ds<1e-4 && break;
            catch err
                T,lx,ly=old;
                message=sprint(showerror,err,catch_backtrace());
                open(errors,"a") do io
                    println(io,"SU ",label," seed=",seed," stage=",stage,"\n",message,"\n");
                end
                println("FAILED ",label," seed=",seed," stage=",stage,": ",sprint(showerror,err));flush(stdout);
                break;
            end
        end
        completed>0 && (states[(label,seed)]=(T=T,lx=lx,ly=ly,history=history));
    end
    measure || return states;
    BLAS.set_num_threads(4);
    setting=LS_CTMRG_settings(CTM_ite_nums=60,CTM_conv_tol=1e-6,
        CTM_ite_info=false,CTM_conv_info=true,grad_checkpoint=false);
    for (label,_,_) in cases,seed in seeds
        haskey(states,(label,seed)) || continue;
        state=states[(label,seed)]; env=nothing; previous_chi=NaN;
        for chi in (16,32,48)
            last_E=NaN; ob=nothing;
            try
                for round=1:2
                    ob=honeycomb_energy_CTM(parameters,state.T,chi,setting;CTM0=env);
                    env=ob.CTM;
                    all(isfinite,ob.energies) || error("nonfinite CTMRG energy");
                    row=(case=label,seed=seed,chi=chi,round=round,iterations=ob.iterations,
                        corner_error=ob.corner_error,E_P_CTM=ob.energy_per_site+3/8,
                        P_x=ob.bond_energy.x+1/4,P_y=ob.bond_energy.y+1/4,P_z=ob.bond_energy.z+1/4,
                        bond_spread=maximum(ob.energies)-minimum(ob.energies));
                    open(ctmfile,"a") do io
                        println(io,join(values(row),","));
                    end
                    println("CTM ",row);flush(stdout);
                    jldsave(joinpath(out,"$(label)_seed$(seed)_CTM.jld2");ob,parameters);
                    (ob.corner_converged || abs(ob.energy_per_site-last_E)<1e-6) && break;
                    last_E=ob.energy_per_site;
                end
                chi>=32 && abs(ob.energy_per_site-previous_chi)<1e-5 && break;
                previous_chi=ob.energy_per_site;
            catch err
                open(errors,"a") do io
                    println(io,"CTM ",label," seed=",seed," chi=",chi,"\n",sprint(showerror,err,catch_backtrace()),"\n");
                end
                println("CTM FAILED ",label," seed=",seed," chi=",chi,": ",sprint(showerror,err));flush(stdout);
                # A small environment budget can be incompatible with a leading
                # irrep. Retry a larger chi from a fresh environment.
                env=nothing;
                continue;
            end
        end
    end
    return states
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_small_D_SU4();
end
