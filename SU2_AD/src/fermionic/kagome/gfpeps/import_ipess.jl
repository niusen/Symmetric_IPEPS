module KagomeGFPEPSValidation
using TensorKit, LinearAlgebra, MAT, JLD2, TOML, Dates
using ..KagomeFermionicZ2
using ..KagomeFermionicCTM
const KF=KagomeFermionicZ2
const KC=KagomeFermionicCTM
const CTM=KC.Legacy
include("ctm/balance_gauge.jl")
include("projection/gutzwiller.jl")
include("observables/heisenberg.jl")
export import_gaussian_ipess, import_gaussian_ipeps, measure_gaussian, measure_gaussian_cell, compare_reference

function import_gaussian_ipess(path;gutzwiller::Bool=false)
    data=matread(path); m=Int(data["m"]); D=2^m
    V=Rep[ℤ₂](0=>D÷2,1=>D÷2); P=KF.physical_space()
    state=initial_iPESS(3,1,V)
    for x in 1:3
        source=mod(1-x,3)
        for (b,s) in enumerate(("B","A","C"))
            a=KF._allout(TensorMap(data["site_$(source)_$s"],P,V*V))
            state.B[x,1,b]=KF.fpermute(a,(3,2,1))
        end
        for (field,label) in ((:T_u,"up"),(:T_d,"down"))
            a=KF._allout(TensorMap(data["triangle_$(source)_$label"],V,V'*V'))
            getfield(state,field)[x,1]=KF.fpermute(a,(2,1,3))
        end
    end
    check_state(state)
    return gutzwiller ? gutzwiller_project(state) : state,vec(data["fields"])
end

function spin_operators(U,fields; model=parton_model(:orig;mu=fields))
    dim(space(U,1))==64 || throw(ArgumentError("Gaussian hopping observables require the unprojected physical space; use spin operators after Gutzwiller projection"))
    ops=KF._triangle_operators(); F=space(U,1)
    fusion=reshape(convert(Array,U),64,64)
    even(M)=U*KF._physical_map(M)*U'
    Cs=Matrix{TensorMap}(undef,3,2); Cdags=similar(Cs)
    dummy=Rep[ℤ₂](1=>1)
    for site in 1:3,spin in 1:2
        cm=fusion*ops.c[site][spin]*fusion'
        cd=fusion*ops.c[site][spin]'*fusion'
        Cs[site,spin]=TensorMap(reshape(cm,1,64,64),dummy*F,F)
        d=TensorMap(permutedims(reshape(cd,1,64,64),(2,3,1)),F,F*dummy)
        Cdags[site,spin]=TensorKit.permute(d,((3,1),(2,)))
    end
    n=[ops.c[i][s]'*ops.c[i][s] for i in 1:3,s in 1:2]
    H=KF.Hamiltonian_triangle(model;orientation=:up,onsite_fraction=1.0)
    up_hop=[even(ops.c[i][spin]'*ops.c[j][spin]) for (i,j) in ((1,2),(2,3),(3,1)),spin in 1:2]
    return (;H=U*H*U',Cs,Cdags,n=map(even,n),up_hop,
        pair=[even(ops.c[i][2]*ops.c[i][1]) for i in 1:3],
        mix=[even(ops.c[i][1]'*ops.c[i][2]) for i in 1:3])
end

function measure_gaussian(state,fields;chi=16,maxiter=100,tol=1e-9,projector_tol=1e-10,verbose=true)
    A=convert_iPESS_to_iPEPS(state)
    _,_,U=build_A_kagome(state,1,1)
    measure_gaussian_cell(A,U,fields;chi,maxiter,tol,projector_tol,verbose)
end

function import_gaussian_ipeps(path;gutzwiller::Bool=false)
    data=matread(path); D=2^Int(data["m"])
    V=Rep[ℤ₂](0=>D÷2,1=>D÷2); P=KF.physical_space()
    U=unitary(fuse(P*P*P),P*P*P)
    A=ntuple(3) do x
        source=mod(1-x,3)
        t=KF._allout(TensorMap(data["cell_$source"],P*P*P,V*V'*V'*V))
        t=KF.fpermute(t,(4,5,6,7,1,2,3))
        t=KF._crossing(KF._crossing(KF._crossing(t,1,2),1,3),2,3)
        @tensor fused[-1,-2,-3,-4,-5] := t[-1,-2,-3,-4,1,2,3]*U[-5,1,2,3]
        (KF._normalized(KF._allout(fused)),)
    end
    if gutzwiller
        A,U=gutzwiller_project(A,U;normalize=true)
    end
    A,U,vec(data["fields"])
end

function measure_gaussian_cell(A,U,fields;chi=16,maxiter=100,tol=1e-9,projector_tol=1e-10,verbose=true,CTM0=nothing,model=parton_model(:orig;mu=fields),environment_path=nothing,update_method="continuous_update")
    dim(space(U,1))==64 || throw(ArgumentError("Gutzwiller-projected states are not Gaussian; the saved Gaussian energy reference and hopping evaluator do not apply"))
    update_method in ("continuous_update","together_update") || error("unknown CTMRG update method")
    CTM.algrithm_CTMRG_settings.CTM_cell_ite_method=update_method
    CTM.Lx,CTM.Ly=length(A),length(A[1]); CTM.chi=chi; CTM.projector_trun_tol=projector_tol
    settings=CTM.LS_CTMRG_settings()
    settings.CTM_conv_tol=tol; settings.CTM_ite_nums=maxiter
    settings.CTM_trun_tol=projector_tol; settings.CTM_ite_info=verbose
    settings.CTM_conv_info=true; settings.grad_checkpoint=false
    initial=CTM.initial_condition(init_type="PBC",reconstruct_CTM=isnothing(CTM0),reconstruct_AA=true)
    env,AA,_,_,_,_,iterations,error=CTM.Fermionic_CTMRG_cell(A,chi,initial,CTM0,settings)
    isnothing(environment_path) || jldsave(environment_path;CTM=env,chi,iterations,error)
    op=spin_operators(U,fields;model)
    nx,ny=length(A),length(first(A)); nsites=3nx*ny
    density=zeros(ComplexF64,nx*ny,3,2); pair=zeros(ComplexF64,nx*ny,3); mix=similar(pair)
    hoppings=zeros(ComplexF64,nx*ny,3,2); up_hoppings=similar(hoppings); energies=zeros(nx*ny)
    for y in 1:ny, x in 1:nx
        i=x+(y-1)*nx
        if verbose
            println("Measuring magnetic-cell block ($x,$y) (onsite and both-spin hoppings)")
            flush(stdout)
        end
        energies[i]=real(CTM.ob_onsite(env,op.H,A,AA,x-1,y-1,settings))
        for site in 1:3
            for spin in 1:2
                density[i,site,spin]=CTM.ob_onsite(env,op.n[site,spin],A,AA,x-1,y-1,settings)
            end
            pair[i,site]=CTM.ob_onsite(env,op.pair[site],A,AA,x-1,y-1,settings)
            mix[i,site]=CTM.ob_onsite(env,op.mix[site],A,AA,x-1,y-1,settings)
        end
        coefficients=KF.parton_hoppings(model,:dn,x)
        for spin in 1:2
            for edge in 1:3
                up_hoppings[i,edge,spin]=CTM.ob_onsite(env,op.up_hop[edge,spin],A,AA,x-1,y-1,settings)
            end
            hoppings[i,1,spin]=conj(CTM.hopping_x(env,op.Cdags[2,spin],op.Cs[1,spin],A,AA,x-1,y-1,settings))
            hoppings[i,2,spin]=CTM.hopping_y(env,op.Cdags[2,spin],op.Cs[3,spin],A,AA,x-2,y-1,settings)
            hoppings[i,3,spin]=-CTM.hopping_diagonala(env,op.Cdags[3,spin],op.Cs[1,spin],A,AA,x-1,y-1,settings)
            for edge in 1:3
                coefficient=coefficients[edge][spin]
                energies[i]+=2real(coefficient*hoppings[i,edge,spin])
            end
        end
    end
    nup=real(sum(density[:,:,1]))/nsites; ndn=1-real(sum(density[:,:,2]))/nsites
    grand=sum(energies)/nsites
    # Removing the field gives kinetic + interaction energy for Hubbard.
    nofield=grand+model.mu[1]*nup+model.mu[2]*ndn
    return (;grand_per_site=grand,energy_without_field_per_site=nofield,
             kinetic_per_site=iszero(model.onsite_U) ? nofield : missing,
             n_up_per_site=nup,n_down_per_site=ndn,density,pair,mix,hoppings,up_hoppings,
             chi,projector_tol,iterations,ctm_error=error,converged=error<tol,update_method,CTM=env)
end

function compare_reference(result,path)
    ref=matread(path); local_ref=ref["onsite_nambu"]; hop_ref=ref["hoppings"]
    density_error=0.;pair_error=0.;mix_error=0.;hopping_error=0.;up_hopping_error=0.
    for x in 1:3
        source=mod(1-x,3)+1
        if !isnothing(result.density)
            for (site,original) in enumerate((2,1,3))
                for spin in 1:2
                    density_error=max(density_error,abs(result.density[x,site,spin]-local_ref[source,original,spin,spin]))
                end
                pair_error=max(pair_error,abs(result.pair[x,site]-local_ref[source,original,1,4]))
                mix_error=max(mix_error,abs(result.mix[x,site]-local_ref[source,original,2,1]))
            end
        end
        hopping_error=max(hopping_error,maximum(abs.(result.hoppings[x,:,:]-hop_ref[source,2,:,:])))
        if hasproperty(result,:up_hoppings) && !isnothing(result.up_hoppings)
            up_hopping_error=max(up_hopping_error,maximum(abs.(result.up_hoppings[x,:,:]-hop_ref[source,1,:,:])))
        end
    end
    report=Dict{String,Any}(string(k)=>getproperty(result,k) for k in
        (:grand_per_site,:kinetic_per_site,:n_up_per_site,:n_down_per_site,:chi,:projector_tol,:iterations,:ctm_error,:converged))
    for key in ("grand_per_site","kinetic_per_site","n_up_per_site","n_down_per_site")
        report[key*"_difference"]=report[key]-ref[key]
        report[key*"_reference"]=ref[key]
    end
    # b=(up,down-hole): <b2*b1> is physical spin flip, while <b1'*b2>
    # is physical pair creation. Keep the physical names unambiguous.
    report["max_down_hopping_error"]=hopping_error
    if !isnothing(result.density)
        merge!(report,Dict("max_onsite_density_error"=>density_error,
            "max_physical_spin_flip_error"=>pair_error,"max_physical_pair_creation_error"=>mix_error))
    end
    hasproperty(result,:up_hoppings) && !isnothing(result.up_hoppings) && (report["max_up_hopping_error"]=up_hopping_error)
    report["detailed_observables"]=!isnothing(result.density)
    report
end
end
