include("Audit_star_CTM.jl")

function star_lambdas(T)
    lx=Matrix{Any}(undef,size(T));ly=similar(lx);
    for i in eachindex(T)
        vl=space(T[i],1);vd=space(T[i],2);
        lx[i]=DiagonalTensorMap(unitary(vl,vl));
        ly[i]=DiagonalTensorMap(unitary(vd',vd'));
    end
    return lx,ly
end

function star_real_spectra(lx,ly)
    return [b.axis==:horizontal ? lx[b.site2...] : ly[b.site1...]
            for b in honeycomb_bonds(size(lx)...)];
end

function star_spectrum_signature(lx,ly)
    return [Dict(Tuple(Int.(SUNRepresentations.dynkin_label(c)))=>abs2.(diag(b))
                 for (c,b) in blocks(s)) for s in star_real_spectra(lx,ly)];
end

function star_spectrum_change(a,b)
    error=0.;
    for (x,y) in zip(a,b)
        keys(x)==keys(y) || return Inf;
        for k in keys(x)
            length(x[k])==length(y[k]) || return Inf;
            error=max(error,norm(x[k]-y[k]));
        end
    end
    return error
end

function run_star_simple_update(;Dmax=9,L=4,
        schedule=((20,.01),(80,.01),(100,.005),(200,.002),(200,.002)),
        identity_sweeps=3,measure=true)
    Dmax>=9 || throw(ArgumentError("the exact star initial state requires Dmax >= 9"));
    BLAS.set_num_threads(1);
    T,info=initial_star_RVB(Lx=L,Ly=L); lx,ly=star_lambdas(T);
    parameters=Dict("J"=>1.);
    out=joinpath(@__DIR__,"../../data","star_SU_D$(Dmax)_L$(L)");mkpath(out);
    trace=joinpath(out,"evolution.csv");sectors=joinpath(out,"sectors.csv");
    write(trace,"stage,dt,tau,E_P_simple,delta_spectrum,min_D,max_D,P_x,P_y,P_z\n");
    write(sectors,"stage,bond,direction,x,y,D,dynkin,irrep_dim,multiplicity,schmidt_weight\n");
    history=NamedTuple[]; tau=0.;previous=star_spectrum_signature(lx,ly);
    function record(stage,dt)
        ob=honeycomb_energy_simple(parameters,T,lx,ly;convention=:permutation);
        spectra=star_real_spectra(lx,ly); dims=[dim(space(s,1)) for s in spectra];
        signature=star_spectrum_signature(lx,ly);
        row=(stage=stage,dt=dt,tau=tau,E_P_simple=ob.energy_per_site,
            delta_spectrum=star_spectrum_change(previous,signature),
            min_D=minimum(dims),max_D=maximum(dims),P_x=ob.bond_energy.x,
            P_y=ob.bond_energy.y,P_z=ob.bond_energy.z);
        all(isfinite,(row.E_P_simple,row.P_x,row.P_y,row.P_z)) || error("nonfinite local energy");
        push!(history,row);open(io->println(io,join(values(row),",")),trace,"a");
        for (n,(bond,s)) in enumerate(zip(honeycomb_bonds(L,L),spectra))
            total=sum(dim(c)*sum(abs2,abs2.(diag(b))) for (c,b) in blocks(s));
            for (c,b) in blocks(s)
                weight=dim(c)*sum(abs2,abs2.(diag(b)))/total;
                r=(stage,n,bond.direction,bond.site1...,dim(space(s,1)),
                    join(Int.(SUNRepresentations.dynkin_label(c)),":"),dim(c),length(diag(b)),weight);
                open(io->println(io,join(r,",")),sectors,"a");
            end
        end
        previous=signature;
        jldsave(joinpath(out,"stage$(stage).jld2");T_set=T,lambdax_set=lx,
            lambday_set=ly,history,parameters,info,Dmax,L);
        println("STAR SU ",row);flush(stdout);
    end
    println("STAR SU START L=",L," Dmax=",Dmax," initial=full_star_RVB");flush(stdout);
    record(-1,0.);
    # Exact identity gates only equilibrate the SU gauge, before imaginary time.
    # Initial identity bond factors reproduce precisely the constructed PEPS.
    if identity_sweeps>0
        println("IDENTITY GAUGE sweeps=",identity_sweeps);flush(stdout);
        itebd_honeycomb(Dict("J"=>0.),T,lx,ly,identity_sweeps*.01,.01,Dmax;verbose=true);
    end
    record(0,0.);
    completed=true;
    for (stage,(steps,dt)) in enumerate(schedule)
        backup=deepcopy((T,lx,ly));
        previous_tau=tau;
        try
            itebd_honeycomb(parameters,T,lx,ly,steps*dt,dt,Dmax;verbose=true);
            tau+=steps*dt;record(stage,dt);
        catch err
            T,lx,ly=backup;tau=previous_tau;completed=false;
            write(joinpath(out,"failure.txt"),sprint(showerror,err,catch_backtrace()));
            println("STAR SU FAILED stage=",stage,": ",sprint(showerror,err));flush(stdout);
            break
        end
    end
    measure || return T,lx,ly,history;
    BLAS.set_num_threads(4);
    setting=LS_CTMRG_settings(CTM_ite_nums=100,CTM_conv_tol=1e-7,
        CTM_ite_info=true,CTM_conv_info=true,grad_checkpoint=false);
    env=nothing;ctm_rows=NamedTuple[];
    for ch in (16,32,48)
        try
            println("STAR SU CTM START chi=",ch," completed=",completed);flush(stdout);
            ob=honeycomb_energy_CTM(parameters,T,ch,setting;convention=:permutation,CTM0=env);
            iderror=identity_bond_checks(T,ob.CTM,setting);
            physical=all(-1-1e-8<=v<=1+1e-8 for v in ob.energies);
            row=(chi=ch,tau=tau,E_P_CTM=ob.energy_per_site,P_x=ob.bond_energy.x,
                P_y=ob.bond_energy.y,P_z=ob.bond_energy.z,
                min_P=minimum(ob.energies),max_P=maximum(ob.energies),
                corner_error=ob.corner_error,iterations=ob.iterations,
                physical=physical,identity_error=iderror,completed=completed);
            push!(ctm_rows,row);println("STAR SU CTM ",row);flush(stdout);
            jldsave(joinpath(out,"CTM_chi$(ch).jld2");row,CTM=ob.CTM,T_set=T,history,parameters);
            # Do not propagate an environment which failed the basic checks.
            env=physical && iderror<1e-8 && ob.corner_converged ? ob.CTM : nothing;
        catch err
            env=nothing;
            open(io->println(io,"chi=",ch," ",sprint(showerror,err,catch_backtrace())),
                joinpath(out,"ctm_failure.txt"),"a");
            println("STAR SU CTM FAILED chi=",ch,": ",sprint(showerror,err));flush(stdout);
        end
    end
    if !isempty(ctm_rows)
        open(joinpath(out,"ctm.csv"),"w") do io
            println(io,join(keys(first(ctm_rows)),","));
            for row in ctm_rows;println(io,join(values(row),","));end
        end
    end
    return T,lx,ly,history
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_star_simple_update();
end
