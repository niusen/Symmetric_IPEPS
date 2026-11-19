# Diagnostic: one complete star-SU sweep, then CTMRG on the saved tensors.
# Run from any directory: julia Trace_D9_CTM.jl [new_output_dir] [sweeps]
using LinearAlgebra, TensorKit, SUNRepresentations, JLD2
load_seconds=@elapsed include("../../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
using .HoneycombSU4

function trace_D9_CTM(out;sweeps=10,dt=.01,chis=(16,24),resume=false)
    ispath(out) && !resume && error("use a new output directory or --resume: $out");
    mkpath(out);BLAS.set_num_threads(1);
    T,lx,ly,info=initial_honeycomb_star_iPEPS(Lx=4,Ly=4);
    setting=LS_CTMRG_settings(CTM_ite_nums=100,CTM_conv_tol=1e-7,
        CTM_ite_info=true,CTM_conv_info=true,grad_checkpoint=false);
    println("MODULE LOAD SECONDS ",load_seconds,"; BLAS=1; independent PBC CTM for each measurement");flush(stdout);
    rows=NamedTuple[];previous=Dict{Int,Float64}();
    for step=0:sweeps
        su_seconds=step==0 ? 0. : @elapsed itebd_honeycomb(Dict("J"=>1.),T,lx,ly,dt,dt,9;
            convention=:permutation,step_offset=step-1,verbose=false);
        jldsave(joinpath(out,"state_step$(lpad(step,3,'0')).jld2");T,lx,ly,step,dt,info);
        dimensions=[dim(space(T[b.site1...],b.axis==:horizontal ? 3 : 2)) for b in honeycomb_bonds(4,4)];
        for chi in chis
            local row;
            resultpath=joinpath(out,"ctm_step$(lpad(step,3,'0'))_chi$chi.jld2");
            if resume && isfile(resultpath)
                row=load(resultpath,"row");push!(rows,row);
                if row.valid;previous[chi]=row.E_P;else;delete!(previous,chi);end
                println("REUSE SAVED MEASUREMENT ",row);flush(stdout);
                continue
            end
            println("MEASURE START step=$step chi=$chi D=$(extrema(dimensions))");flush(stdout);
            started=time();
            try
                timed=@timed honeycomb_energy_CTM(Dict("J"=>1.),T,chi,setting;convention=:permutation);
                ob=timed.value;
                physical=all(v->isfinite(v) && -1-1e-8<=v<=1+1e-8,ob.energies);
                valid=physical && ob.corner_converged;
                delta=valid && haskey(previous,chi) ? ob.energy_per_site-previous[chi] : NaN;
                row=(step=step,tau=step*dt,chi=chi,E_P=ob.energy_per_site,delta_E=delta,
                    ctm_seconds=timed.time,gc_seconds=timed.gctime,allocated_bytes=timed.bytes,
                    su_seconds=su_seconds,Dmin=minimum(dimensions),Dmax=maximum(dimensions),
                    iterations=ob.iterations,corner_error=ob.corner_error,physical=physical,valid=valid,
                    min_P=minimum(ob.energies),max_P=maximum(ob.energies));
                if valid;previous[chi]=ob.energy_per_site;else;delete!(previous,chi);end
                jldsave(resultpath;row,CTM=ob.CTM,bonds=ob.bonds,energies=ob.energies);
            catch err
                err isa InterruptException && rethrow();
                delete!(previous,chi);
                row=(step=step,tau=step*dt,chi=chi,E_P=NaN,delta_E=NaN,ctm_seconds=time()-started,
                    gc_seconds=NaN,allocated_bytes=0,su_seconds=su_seconds,
                    Dmin=minimum(dimensions),Dmax=maximum(dimensions),iterations=0,
                    corner_error=NaN,physical=false,valid=false,min_P=NaN,max_P=NaN);
                open(joinpath(out,"failures.log"),"a") do io
                    println(io,"step=$step chi=$chi\n",sprint(showerror,err,catch_backtrace()));
                end
            end
            push!(rows,row);println("MEASURE RESULT ",row);flush(stdout);
            open(joinpath(out,"energy_by_sweep.csv"),"w") do io
                println(io,join(keys(first(rows)),","));
                for r in rows;println(io,join(values(r),","));end
            end
        end
    end
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    out=isempty(ARGS) ? joinpath(@__DIR__,"../../logs/tests/D9_CTM_by_sweep") : abspath(ARGS[1]);
    trace_D9_CTM(out;sweeps=length(ARGS)>=2 ? parse(Int,ARGS[2]) : 10,resume="--resume" in ARGS);
end
