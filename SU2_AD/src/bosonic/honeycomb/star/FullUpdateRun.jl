module HoneycombFullRun
using TOML,Dates,LinearAlgebra,JLD2
include("FullRunConfig.jl")
include("StarSimpleUpdateRun.jl")
using .HoneycombStarRun.HoneycombSU4
using TensorKit,SUNRepresentations
const HC=HoneycombStarRun.HoneycombSU4
const SR=HoneycombStarRun
export read_full_config,describe_full_run,run_full_server

function _full_csv(path,rows)
    isempty(rows) && return;
    SR._atomic_write(path) do tmp
        open(tmp,"w") do io
            HC._star_audit_line(io,string.(keys(first(rows)))...);
            for r in rows
                HC._star_audit_line(io,(v isa Number ? v : string(v) for v in values(r))...);
            end
        end
    end
end

function _full_checkpoint(c,A,p,history,reports)
    SR._atomic_write(joinpath(c.output_dir,"state_latest.jld2")) do tmp
        jldsave(tmp;schema="honeycomb_two_site_fu_v1",physics=full_physics(c),A_set=A,
            progress=p,history,reports,coordinate_system="ctm_down");
    end
    _full_csv(joinpath(c.output_dir,"energy_by_sweep.csv"),history);
    _full_csv(joinpath(c.output_dir,"bond_updates.csv"),reports);
    SR._atomic_write(joinpath(c.output_dir,"status.toml")) do tmp
        open(io->TOML.print(io,Dict("progress"=>p,"saved_at"=>string(now()),"space_policy"=>String(c.space_policy))),tmp,"w");
    end
end

function _full_initial(c,path)
    if isnothing(path)
        T=c.initial==:star ? first(initial_honeycomb_star_iPEPS(Lx=c.Lx,Ly=c.Ly)) :
            first(initial_honeycomb_iPEPS(c.Lx,c.Ly;symmetry=:dense,D=c.dense_D,seed=c.seed));
        return HC.square_fu_cell_to_matrix(convert_honeycomb_to_iPEPS(T))
    end
    s=load(path);
    if haskey(s,"A_set")
        get(s,"coordinate_system","")=="ctm_down" || error("unknown checkpoint coordinate system");
        A=s["A_set"];
    elseif haskey(s,"T_set")
        A=HC.square_fu_cell_to_matrix(convert_honeycomb_to_iPEPS(s["T_set"]));
    else
        error("initial checkpoint needs A_set (FU) or T_set (SU)");
    end
    size(A)==(c.Lx,c.Ly) || error("initial checkpoint cell mismatch");
    return A
end

function _full_energy!(history,c,A,env,p,seconds)
    ob=honeycomb_energy_FU(A,env;J=c.J);
    groups=honeycomb_fu_bond_groups(A);
    dims=[dim(space(A[b.site1],b.direction==:x ? 3 : 2)) for b in reduce(vcat,groups)];
    row=(sweep=p["total_steps"],tau=p["tau"],chi=c.chi,E_P=ob.E_P,E_H=ob.E_H,
        min_P=minimum(r.P for r in ob.bonds),max_P=maximum(r.P for r in ob.bonds),
        Dmin=minimum(dims),Dmax=maximum(dims),corner_error=ob.corner_error,
        ctm_iterations=ob.iterations,sweep_seconds=seconds);
    push!(history,row);println("HONEYCOMB FU ENERGY ",row);flush(stdout);
    _full_csv(joinpath(c.output_dir,"bonds_sweep$(p["total_steps"]).csv"),ob.bonds);
    return row
end

"""One job evolves and measures; only completed, validated sweeps replace the
last checkpoint. Resume rebuilds CTM for the saved tensors, never imports lambda.
"""
function run_full_server(c;resume=false,initial_checkpoint=nothing,measure_only=false,stop_after_sweeps=typemax(Int))
    resume && !isnothing(initial_checkpoint) && error("resume and initial checkpoint are exclusive");
    describe_full_run(c);BLAS.set_num_threads(c.blas_threads);
    return SR._locked(c.output_dir) do
        latest=joinpath(c.output_dir,"state_latest.jld2");
        if resume || measure_only
            s=load(latest);get(s,"schema","")=="honeycomb_two_site_fu_v1" || error("not a FU checkpoint");
            s["physics"]==full_physics(c) || error("resume parameters differ; use a new output and --init-checkpoint");
            A=s["A_set"];p=s["progress"];history=s["history"];reports=s["reports"];
        else
            isfile(latest) && error("checkpoint exists; use --resume");
            A=_full_initial(c,initial_checkpoint);
            p=Dict{String,Any}("stage"=>1,"stage_step"=>0,"total_steps"=>0,"tau"=>0.,"finished"=>false);
            history=NamedTuple[];reports=NamedTuple[];
        end
        setting=LS_CTMRG_settings(CTM_ite_nums=c.ctm_iterations,CTM_conv_tol=c.ctm_tol,
            CTM_ite_info=true,CTM_conv_info=true,grad_checkpoint=false);
        settings=SquareJ1FullUpdateSettings(;Dmax=c.Dmax,multiplet_tol=0.,c.optimizer...);
        try
            all(dim(space(A[b.site1],b.direction==:x ? 3 : 2))<=c.Dmax
                for b in reduce(vcat,honeycomb_fu_bond_groups(A))) || error("input bond space exceeds configured Dmax");
            if isempty(history) || measure_only
                env=honeycomb_fu_environment(A,c.chi,setting);
                if measure_only
                    ob=honeycomb_energy_FU(A,env;J=c.J);
                    _full_csv(joinpath(c.output_dir,"remeasured_bonds.csv"),ob.bonds);
                    println("FU REMEASURE ",(E_P=ob.E_P,corner_error=ob.corner_error));return :measured;
                end
                _full_energy!(history,c,A,env,p,0.);_full_checkpoint(c,A,p,history,reports);
            end
            completed=0;
            while p["stage"]<=length(c.stages)
                stage=c.stages[p["stage"]];
                if p["stage_step"]>=stage.steps
                    p["stage"]+=1;p["stage_step"]=0;continue
                end
                completed>=stop_after_sweeps && return :checkpointed;
                pending=NamedTuple[];
                observer=r->begin
                    push!(pending,merge(r,(sweep=p["total_steps"]+1,)));
                    println("FU BOND ",(sweep=p["total_steps"]+1,layer=r.layer,direction=r.direction,
                        site1=r.site1,site2=r.site2,loss_initial=r.loss_initial,loss_final=r.loss_final,
                        old_space=string(r.old_bond_space),new_space=string(r.new_bond_space),corner_error=r.corner_error));flush(stdout);
                end;
                elapsed=@elapsed candidate,env,_=honeycomb_full_update_sweep(A,c.chi,stage.dt;J=c.J,Dmax=c.Dmax,
                    space_policy=c.space_policy,settings,ctm_setting=setting,step_offset=p["total_steps"],observer);
                nextp=copy(p);nextp["total_steps"]+=1;nextp["stage_step"]+=1;nextp["tau"]+=stage.dt;
                _full_energy!(history,c,candidate,env,nextp,elapsed);
                append!(reports,pending);A=candidate;p=nextp;
                _full_checkpoint(c,A,p,history,reports);completed+=1;
            end
            p["finished"]=true;_full_checkpoint(c,A,p,history,reports);
            println("Honeycomb full update schedule complete: ",latest);flush(stdout);
            return :finished
        catch err
            open(joinpath(c.output_dir,"failure.log"),"a") do io
                println(io,now(),"; last complete checkpoint: ",latest,"\n",sprint(showerror,err,catch_backtrace()));
            end
            rethrow()
        end
    end
end
end
