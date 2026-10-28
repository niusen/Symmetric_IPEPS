# Function entry points for run_kagome_settings.jl; numerical kernels are unchanged.
if !isdefined(@__MODULE__, :export_and_verify)
    include(joinpath(@__DIR__, "verify_charged_dense.jl"))
end
if !isdefined(@__MODULE__, :charged_trial)
    include(joinpath(@__DIR__, "run_charged_trial.jl"))
end
if !isdefined(@__MODULE__, :run_u1_options)
    include(joinpath(@__DIR__, "run.jl"))
end

function settings_schedule(schedule)
    isempty(schedule) && error("schedule cannot be empty")
    all(isfinite(s.dt) && s.dt>0 && isfinite(s.tau) && s.tau>=0 for s in schedule) ||
        error("schedule requires finite dt>0 and tau>=0")
    return schedule
end

function run_kagome_settings(task; su,charged,conversion)
    if task==:charged_trial
        c=charged
        return charged_trial(c.out;Dmax=c.Dmax,seed=c.seed,h=c.h,threads=c.threads,
            symmetry=c.symmetry,initial_state=isempty(c.initial_state) ? nothing : c.initial_state,
            schedule=settings_schedule(c.schedule))
    elseif task==:simple_update
        c=su
        # Adapt code settings to the existing runner; users do not enter arguments.
        values=(symmetry=string(c.symmetry),cell=c.cell,Dmax=string(c.Dmax),seed=string(c.seed),
            threads=string(c.threads),J1=string(c.J1),virtual=c.virtual_space,
            init=c.initial_state,component=string(c.component),resume=c.resume,out=c.out)
        args=String[]
        for name in keys(values)
            append!(args,["--"*string(name),getproperty(values,name)])
        end
        return main(args;schedule_override=settings_schedule(c.schedule))
    elseif task==:convert
        c=conversion
        isempty(c.input) && error("fill conversion_options.input in run_kagome_settings.jl")
        isfile(c.input) || error("conversion checkpoint does not exist: $(c.input)")
        return export_and_verify(c.input,c.out;h=c.h,component=c.component,
            threads=c.threads,representation=c.representation)
    end
    error("task must be :charged_trial, :simple_update or :convert")
end
