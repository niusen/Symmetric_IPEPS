module TensorMapSU4
using LinearAlgebra, Random, Serialization, Printf
using TensorKit, TensorKitSectors, ChainRulesCore, KrylovKit, Zygote
using Zygote: @ignore_derivatives

const SRC=normpath(joinpath(@__DIR__,"../.."))
include(joinpath(SRC,"Settings.jl"))
include(joinpath(SRC,"Settings_cell.jl"))
include(joinpath(SRC,"iPEPS_ansatz.jl"))
include(joinpath(SRC,"AD_lib.jl"))
include(joinpath(SRC,"SUN_spin_op.jl"))
include(joinpath(SRC,"CTMRG.jl"))
include(joinpath(SRC,"CTMRG_unitcell.jl"))
include(joinpath(SRC,"triangle/triangle_spin_model_cell.jl"))
include(joinpath(SRC,"triangle/triangle_iPESS_method.jl"))
include(joinpath(SRC,"triangle/simple_update/triangle_SimpleUpdate.jl"))
include(joinpath(SRC,"triangle/simple_update/triangle_SimpleUpdate_iPESS.jl"))

export init_state, model_settings, measure_state, save_state, load_state,
       nearest_neighbor_bonds, bond_table_matrix, bond_pattern_diagnostics,
       print_observables, save_observables, observable_mat_data

function model_settings(nx,ny)
    global Lx=nx
    global Ly=ny
    return Triangle_SUN_Spin_settings(model="Heisenberg",N=4,Lx=nx,Ly=ny)
end

function init_state(nx,ny;D=4,seed=1234)
    model_settings(nx,ny)
    Random.seed!(seed)
    B,T,l1,l2,l3=initial_iPESS(nx,ny,ComplexSpace(D),ComplexSpace(4))
    return (B=B,T=T,lambda1=l1,lambda2=l2,lambda3=l3)
end

function save_state(path,state;metadata=NamedTuple())
    serialize(path,(format=:SU4_TensorMap_iPESS_v1,state=state,metadata=metadata))
end
function load_state(path)
    data=deserialize(path)
    hasproperty(data,:format) && data.format==:SU4_TensorMap_iPESS_v1 ||
        error("Expected a TensorMap iPESS checkpoint; historical Array checkpoints are not resumed by this driver.")
    return data
end

# All one-site contractions also use the existing spin-model routine.
function site_densities(A,AA,env,setting)
    densities=Matrix{Matrix{ComplexF64}}(undef,Lx,Ly)
    for x=1:Lx,y=1:Ly
        cy=mod1(2-y,Ly)
        V=space(A[x][cy],5)
        rho=zeros(ComplexF64,4,4)
        for p=1:4,q=1:4
            E=zeros(ComplexF64,4,4); E[p,q]=1
            rho[q,p]=ob_onsite(env,TensorMap(E,V,V),A,AA,x-1,cy-1,setting)
        end
        densities[x,y]=rho
    end
    return densities
end

"""Original CTMRG_cell + evaluate_ob_cell, followed only by coordinate/format conversion.
The state remains TensorMap throughout SU and CTMRG. Array conversion is for
small observables/MAT export only. Shared src globals require sequential calls.
"""
function measure_state(state;J=1.0,convention=:traceless,chis=[16,32,48],
                       max_iterations=50,corner_tol=1e-7,chi_tol=1e-5,verbose=true)
    convention in (:traceless,:permutation) || throw(ArgumentError("unknown convention"))
    !isempty(chis) && issorted(chis) && allunique(chis) && all(>(0),chis) || throw(ArgumentError("invalid chis"))
    1<=max_iterations<=50 || throw(ArgumentError("CTMRG iterations must be between 1 and 50"))
    setting=model_settings(size(state.B)...)
    parameters=Dict("J"=>Float64(J),"K"=>0.0,"Φ"=>0.0)
    global multiplet_tol=0.0 # ordinary dense rank truncation; no multiplet library required
    global projector_trun_tol=1e-12
    global algrithm_CTMRG_settings=Algrithm_CTMRG_settings(CTM_cell_ite_method="continuous_update")
    ctm=LS_CTMRG_settings(CTM_conv_tol=corner_tol,CTM_ite_nums=max_iterations,
        CTM_trun_tol=projector_trun_tol,CTM_ite_info=verbose,CTM_conv_info=true,
        CTM_trun_svd=false,construct_double_layer=true,grad_checkpoint=false)
    original=convert_iPESS_to_iPEPS(state.B,state.T)
    # SU y increases toward B.U--T.D; CTMRG patch coordinates increase downward.
    A=ntuple(x->ntuple(cy->original[x][mod1(2-cy,Ly)],Ly),Lx)
    env=nothing; previous=nothing; obs=nothing; history=NamedTuple[]
    for ch in chis
        global chi=ch
        init=initial_condition(init_type="PBC",reconstruct_CTM=isnothing(env),reconstruct_AA=true)
        env,AA,_,_,_,_,iterations,error=CTMRG_cell(A,ch,init,env,ctm)
        energy,ex,ey,ed,_,_=evaluate_ob_cell(parameters,A,AA,env,ctm,setting)
        maps=map(M->[real(M[mod1(x-2,Lx),mod1(-y,Ly)]) for x=1:Lx,y=1:Ly],(ex,ey,ed))
        coupling=(x=maps[1],y=maps[2],diagonal=maps[3])
        exchange=map(M->M.+0.25,coupling)
        generator_dot=map(M->M./2,coupling)
        bond_energy=map(M->J.*M,convention==:traceless ? coupling : exchange)
        densities=site_densities(A,AA,env,ctm)
        connected=map(_->zeros(Lx,Ly),coupling)
        for x=1:Lx,y=1:Ly
            sites=((mod1(x-1,Lx),y),(x,y),(x,mod1(y+1,Ly)))
            for (d,(i,j)) in zip((:x,:y,:diagonal),((1,2),(2,3),(1,3)))
                getproperty(connected,d)[x,y]=(getproperty(exchange,d)[x,y]-real(tr(densities[sites[i]...]*densities[sites[j]...])))/2
            end
        end
        density=bond_energy.x+bond_energy.y+bond_energy.diagonal
        means=(exchange=map(M->sum(M)/length(M),exchange),coupling=map(M->sum(M)/length(M),coupling),
            generator_dot=map(M->sum(M)/length(M),generator_dot),connected_generator_dot=map(M->sum(M)/length(M),connected),
            bond_energy=map(M->sum(M)/length(M),bond_energy))
        obs=(environment=:ctmrg,J=Float64(J),convention=convention,cell=(Lx,Ly),exchange=exchange,
            coupling=coupling,generator_dot=generator_dot,connected_generator_dot=connected,
            bond_energy=bond_energy,energy_density=density,energy_per_site=sum(density)/(Lx*Ly),
            energy_per_bond=sum(density)/(3Lx*Ly),means=means)
        @assert isapprox(obs.energy_per_site,energy+(convention==:permutation ? 3J/4 : 0);atol=1e-10,rtol=1e-10)
        change=isnothing(previous) ? Inf : maximum(maximum(abs.(getproperty(coupling,d)-getproperty(previous,d))) for d in keys(coupling))
        push!(history,(chi=ch,iterations=iterations,corner_error=Float64(error),
            iteration_converged=error<corner_tol,energy=obs.energy_per_site,max_bond_change=change))
        previous=coupling
        verbose && println("chi=",ch," E/site=",obs.energy_per_site," max_bond_change=",change)
        flush(stdout)
    end
    converged=length(history)>1 && history[end].iteration_converged && history[end-1].iteration_converged && history[end].max_bond_change<chi_tol
    return merge(obs,(ctm=(chi=last(chis),corner_converged=history[end].iteration_converged,
        iteration_converged=history[end].iteration_converged,chi_converged=converged,
        corner_tol=corner_tol,chi_tol=chi_tol,history=history),))
end

# Reporting helpers only: no tensor-network contraction or update implementation.
include(joinpath(@__DIR__,"Observables_iPESS.jl"))

function observable_mat_data(obs)
    data=Dict{String,Any}("environment"=>"ctmrg","tensor_format"=>"TensorMap_ComplexSpace",
        "J"=>obs.J,"convention"=>string(obs.convention),"Lx"=>obs.cell[1],"Ly"=>obs.cell[2],
        "energy_per_site"=>obs.energy_per_site,"energy_per_bond"=>obs.energy_per_bond,
        "energy_density"=>obs.energy_density,"ctm_chi"=>obs.ctm.chi,
        "ctm_iteration_converged"=>obs.ctm.iteration_converged,"ctm_chi_converged"=>obs.ctm.chi_converged,
        "ctm_corner_tol"=>obs.ctm.corner_tol,"ctm_chi_tol"=>obs.ctm.chi_tol,
        "ctm_history_columns"=>string.(collect(keys(first(obs.ctm.history)))),
        "ctm_history"=>reduce(vcat,[permutedims(Float64.(collect(values(r)))) for r in obs.ctm.history]))
    for field in (:exchange,:coupling,:generator_dot,:connected_generator_dot,:bond_energy)
        for d in (:x,:y,:diagonal)
            data[string(field,"_",d)]=getproperty(getproperty(obs,field),d)
        end
        data["mean_"*string(field)]=reshape(collect(values(getproperty(obs.means,field))),1,3)
    end
    data["bond_table"],data["bond_table_columns"]=bond_table_matrix(obs)
    p=bond_pattern_diagnostics(obs)
    data["fourier_qx"]=p.qx; data["fourier_qy"]=p.qy
    data["direction_anisotropy"]=p.direction_anisotropy
    data["spatial_modulation_range"]=collect(values(p.modulation_range))
    for d in (:x,:y,:diagonal)
        data["coupling_origin_"*string(d)]=getproperty(p.origin_coupling,d)
        data["coupling_fourier_"*string(d)]=getproperty(p.fourier,d)
    end
    return data
end
end
