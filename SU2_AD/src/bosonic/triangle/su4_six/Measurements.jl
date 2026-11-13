# Existing spin-model routines accept these three-leg operator-Schmidt factors.
function bond_factors(h)
    u,s,v=tsvd(permute(h,(1,3,),(2,4,));trunc=truncbelow(1e-13))
    left=permute(u*s,(3,1,),(2,))
    right=permute(v,(1,2,),(3,))
    @tensor reconstructed[-1,-2,-3,-4] := left[1,-1,-3]*right[1,-2,-4]
    norm(permute(reconstructed,(1,2,),(3,4,))-h)<1e-10 || error("Invalid operator-Schmidt factors")
    return left,right
end

function evaluate_bonds(A,AA,env,ctm,factors)
    result=(x=zeros(Lx,Ly),y=zeros(Lx,Ly),diagonal=zeros(Lx,Ly))
    for x=1:Lx,y=1:Ly
        cx=mod1(x-2,Lx); cy=mod1(-y,Ly)
        # hopping_x inserts on LU--RU; the simplex map uses LD--RD.
        result.x[x,y]=real(hopping_x_no_sign(env,factors...,A,AA,cx,cy+1,ctm))
        result.y[x,y]=real(hopping_y_no_sign(env,factors...,A,AA,cx,cy,ctm))
        result.diagonal[x,y]=real(hopping_diagonala_no_sign(env,factors...,A,AA,cx,cy,ctm))
    end
    return result
end

function local_densities(A,AA,env,ctm,model)
    if model.backend==:su4
        # An invariant one-site density matrix on this irreducible physical space
        # is exactly I6/6. Individual non-invariant operators are not TensorMaps.
        return [Matrix{ComplexF64}(I,6,6)/6 for x=1:Lx,y=1:Ly]
    end
    rho=Matrix{Matrix{ComplexF64}}(undef,Lx,Ly)
    for x=1:Lx,y=1:Ly
        grid_y=mod1(2-y,Ly); V=space(A[x][grid_y],5)
        r=zeros(ComplexF64,6,6)
        for p=1:6,q=1:6
            E=zeros(ComplexF64,6,6); E[p,q]=1
            r[q,p]=ob_onsite(env,TensorMap(E,V,V),A,AA,x-1,grid_y-1,ctm)
        end
        rho[x,y]=r
    end
    return rho
end

"""Pair the exact M bond used inside triangle_gate_iPESS_simplified.
Simplex B[x,y] connects T[x-1,y].R, T[x,y+1].D, T[x,y].M.
The physical y coordinate increases upward, opposite to the CTMRG patch rows.
"""
function measurement_cell(state)
    original=convert_iPESS_to_iPEPS(state.B,state.T)
    A=ntuple(x->ntuple(cy->original[x][mod1(2-cy,Ly)],Ly),Lx)
    for p=1:Lx,q=1:Ly
        space(A[p][q],3)'==space(A[mod1(p+1,Lx)][q],1) || error("Mismatched horizontal iPESS bond")
        space(A[p][q],2)'==space(A[p][mod1(q+1,Ly)],4) || error("Mismatched vertical iPESS bond")
    end
    return A
end

function trace_boundary(V,leg)
    W=fuse(V'⊗V); E=id(V)
    if leg in (1,2)
        Udag=unitary(W,V'⊗V)'
        @tensor v[-1] := Udag[1,2,-1]*E[1,2]
    else
        Udag=unitary(V⊗V',W)'
        @tensor v[-1] := Udag[-1,1,2]*E[1,2]
    end
    return permute(v/norm(v),(1,),())
end

function identity_environment(A,ch;verbose=false)
    # Old PBC initialization supplies the corner/boundary orientation convention.
    # Identity trace vectors are attached to the actual neighboring virtual legs;
    # no virtual-space padding and no isolated-site periodic closure is needed.
    a=A[1][1]
    V0=sectortype(a)==Trivial ? ComplexSpace(1) : Rep[SU{4}]((0,0,0,0)=>1)
    p=TensorMap{ComplexF64}(undef,V0,one(V0)); fill!(p,1)
    left=isdual(space(a,1)) ? V0' : V0
    down=isdual(space(a,2)) ? V0' : V0
    horizontal=id(left); vertical=id(down)
    @tensor seed[-1,-2,-3,-4,-5] := horizontal[-1,-3]*vertical[-2,-4]*p[-5]
    seed=permute(seed,(1,2,3,4,5,),())
    cell=ntuple(_->ntuple(_->seed,Ly),Lx)
    base=init_CTM_cell(ch,cell,"PBC",verbose)
    unit_vector(V)=TensorMap(ones(ComplexF64,1),V,one(V))
    Tset=ntuple(Lx) do x
        ntuple(Ly) do y
            positions=((x,mod1(y+1,Ly),4),(mod1(x-1,Lx),y,3),
                (x,mod1(y-1,Ly),2),(mod1(x+1,Lx),y,1))
            edges=ntuple(4) do direction
                sx,sy,leg=positions[direction]
                middle=trace_boundary(space(A[sx][sy],leg),leg)
                template=get_Tset(base.Tset[x][y],direction)
                l=unit_vector(space(template,1)); r=unit_vector(space(template,3))
                @tensor edge[-1,-2,-3] := l[-1]*middle[-2]*r[-3]
                permute(edge,(1,2,3,),())
            end
            Tset_struc(edges...)
        end
    end
    return (;Cset=base.Cset,Tset)
end

"""Energy and all nearest-neighbor invariants using the original CTMRG environment.
Only closed double layers with operator insertions are used. No six-dimensional
open-physical double-layer tensor or three-site density matrix is constructed.
"""
function measure_state(state;J=1.0,chis=[16,32,48],max_iterations=50,
    corner_tol=1e-7,chi_tol=1e-5,verbose=true,onsite=true)
    setup_cell(size(state.B)...)
    1<=max_iterations<=50 || throw(ArgumentError("at most 50 CTMRG iterations per chi"))
    !isempty(chis) && all(>(0),chis) && issorted(chis) && allunique(chis) || throw(ArgumentError("invalid chis"))
    backend=sectortype(state.T[1])==Trivial ? :dense : :su4
    verbose && println("Preparing six-irrep ",backend," model and CTMRG operator insertions")
    flush(stdout)
    model=six_model(;backend,J)
    global multiplet_tol=0.0
    global projector_trun_tol=1e-12
    global algrithm_CTMRG_settings=Algrithm_CTMRG_settings(CTM_cell_ite_method="continuous_update")
    ctm=LS_CTMRG_settings(CTM_conv_tol=corner_tol,CTM_ite_nums=max_iterations,
        CTM_trun_tol=projector_trun_tol,CTM_ite_info=verbose,CTM_conv_info=true,
        CTM_trun_svd=false,construct_double_layer=true,grad_checkpoint=false)
    A=measurement_cell(state)
    all(a->dim(space(a,5))==6,Iterators.flatten(A)) || error("Physical dimension must be six")
    hf=bond_factors(model.h); pf=bond_factors(model.swap)
    verbose && println("six-irrep ",backend," identity CTMRG environment initialization")
    flush(stdout); started=time()
    env=identity_environment(A,first(chis);verbose)
    seed_seconds=time()-started
    history=NamedTuple[]; previous=nothing; obs=nothing
    for (index,ch) in enumerate(chis)
        global chi=ch
        global multiplet_tol=0.0
        init=initial_condition(init_type="PBC",reconstruct_CTM=false,reconstruct_AA=true)
        verbose && println("six-irrep ",backend," chi=",ch," CTMRG start")
        flush(stdout); phases=Dict{Symbol,Float64}()
        env,AA,_,_,_,_,iterations,error=CTMRG_cell(A,ch,init,env,ctm;timings=phases)
        if index==1
            phases[:environment_init_seconds]+=seed_seconds
            phases[:total_seconds]+=seed_seconds
        end
        verbose && println("chi=",ch," CTMRG finished; nearest-neighbor observables start")
        flush(stdout); started=time()
        coupling=evaluate_bonds(A,AA,env,ctm,hf)
        verbose && println("chi=",ch," Hamiltonian bonds finished; swap/singlet observables start")
        flush(stdout)
        exchange=evaluate_bonds(A,AA,env,ctm,pf)
        all(v->all(isfinite,v),(coupling...,exchange...)) || error("Non-finite CTMRG bond observables at chi=$ch")
        bond_seconds=time()-started
        singlet=map((p,h)->(p-h)/6,exchange,coupling)
        p15=map(p->(1 .- p)/2,exchange)
        p20=map((p,s)->(1 .+ p)/2-s,exchange,singlet)
        generator_dot=map(h->h/2,coupling)
        connected=map(h->fill(NaN,size(h)),coupling)
        densities=nothing; onsite_seconds=0.0
        if index==length(chis) && (onsite || backend==:su4)
            verbose && println("chi=",ch," nearest-neighbor observables finished; onsite observables start")
            flush(stdout); started=time()
            densities=local_densities(A,AA,env,ctm,model)
            for x=1:Lx,y=1:Ly
                sites=((mod1(x-1,Lx),y),(x,y),(x,mod1(y+1,Ly)))
                for (dir,(i,j)) in zip((:x,:y,:diagonal),((1,2),(2,3),(1,3)))
                    r1=densities[sites[i]...]; r2=densities[sites[j]...]
                    product=sum(tr(r1*model.C[a,b])*tr(r2*model.C[b,a]) for a=1:4,b=1:4)
                    getproperty(connected,dir)[x,y]=(getproperty(coupling,dir)[x,y]-real(product))/2
                end
            end
            onsite_seconds=time()-started
        end
        bond_energy=map(h->J*h,coupling)
        density=bond_energy.x+bond_energy.y+bond_energy.diagonal
        means=(;exchange=map(h->sum(h)/length(h),exchange),coupling=map(h->sum(h)/length(h),coupling),
            generator_dot=map(h->sum(h)/length(h),generator_dot),
            connected_generator_dot=map(h->sum(h)/length(h),connected),
            bond_energy=map(h->sum(h)/length(h),bond_energy))
        obs=(;environment=:ctmrg,backend,physical_dimension=6,physical_irrep=:antisymmetric_two,
            particles_per_site=2,physical_basis=model.basis,cell=(Lx,Ly),J=Float64(J),convention=:scipost_2020,
            coupling,exchange,singlet_weight=singlet,adjoint_weight=p15,symmetric20_weight=p20,
            generator_dot,connected_generator_dot=connected,onsite_density=densities,
            bond_energy,energy_density=density,energy_per_site=sum(density)/(Lx*Ly),
            energy_per_bond=sum(density)/(3Lx*Ly),means)
        change=isnothing(previous) ? Inf : maximum(maximum(abs.(a-b))
            for (current,old) in zip((coupling,exchange),previous) for (a,b) in zip(current,old))
        push!(history,(chi=ch,iterations=iterations,corner_error=Float64(error),
            iteration_converged=error<corner_tol,energy=obs.energy_per_site,max_bond_change=change,
            double_layer_seconds=phases[:double_layer_seconds],environment_init_seconds=phases[:environment_init_seconds],
            iteration_seconds=phases[:iteration_seconds],ctmrg_seconds=phases[:total_seconds],
            bond_observables_seconds=bond_seconds,onsite_observables_seconds=onsite_seconds))
        previous=(coupling,exchange)
        verbose && @printf("chi=%d E/site=%.15g CTMRG=%.3fs bonds=%.3fs onsite=%.3fs\n",
            ch,obs.energy_per_site,phases[:total_seconds],bond_seconds,onsite_seconds)
        flush(stdout)
    end
    converged=length(history)>1 && history[end].iteration_converged &&
        history[end-1].iteration_converged && history[end].max_bond_change<chi_tol
    return merge(obs,(ctm=(chi=last(chis),corner_converged=history[end].iteration_converged,
        iteration_converged=history[end].iteration_converged,chi_converged=converged,
        max_iterations=max_iterations,corner_tol=corner_tol,chi_tol=chi_tol,history=history),))
end

# Reuse the full bond-coordinate/winding, Fourier, CSV and plain-text reporting.
include(joinpath(@__DIR__,"../su4_dense/Observables_iPESS.jl"))

function observable_mat_data(obs)
    table,columns=bond_table_matrix(obs)
    data=Dict{String,Any}("model"=>"triangular_SU4_antisymmetric_6", "particles_per_site"=>2,
        "physical_dimension"=>6,"physical_irrep"=>[1,1,0,0],"backend"=>string(obs.backend),
        "physical_basis"=>string(obs.physical_basis),"occupation_basis"=>["12","13","14","23","24","34"],
        "physical_to_ctm_coordinates"=>[1 0;0 -1],"physical_to_ctm_offset"=>[0,2],
        "hamiltonian"=>"J*(swap_6-6*P_singlet)","generator_normalization"=>"h/J=2*sum_A T_i^A*T_j^A; C2(6)=5/2",
        "environment"=>"ctmrg","J"=>obs.J,"Lx"=>obs.cell[1],"Ly"=>obs.cell[2],
        "energy_per_site"=>obs.energy_per_site,"energy_per_bond"=>obs.energy_per_bond,
        "energy_density"=>obs.energy_density,
        "bond_table"=>table,"bond_table_columns"=>columns,"ctm_chi"=>obs.ctm.chi,
        "ctm_max_iterations"=>obs.ctm.max_iterations,"ctm_corner_tol"=>obs.ctm.corner_tol,
        "ctm_chi_tol"=>obs.ctm.chi_tol,
        "ctm_chi_converged"=>obs.ctm.chi_converged,"ctm_iteration_converged"=>obs.ctm.iteration_converged,
        "ctm_history_columns"=>string.(collect(keys(first(obs.ctm.history)))),
        "ctm_history"=>reduce(vcat,[permutedims(Float64.(collect(values(r)))) for r in obs.ctm.history]))
    for field in (:coupling,:exchange,:singlet_weight,:adjoint_weight,:symmetric20_weight,
        :generator_dot,:connected_generator_dot,:bond_energy),d in (:x,:y,:diagonal)
        data[string(field,"_",d)]=getproperty(getproperty(obs,field),d)
    end
    pattern=bond_pattern_diagnostics(obs)
    data["direction_anisotropy"]=pattern.direction_anisotropy
    data["spatial_modulation_range"]=collect(values(pattern.modulation_range))
    data["fourier_qx"]=pattern.qx; data["fourier_qy"]=pattern.qy
    for d in (:x,:y,:diagonal)
        data["coupling_origin_"*string(d)]=getproperty(pattern.origin_coupling,d)
        data["coupling_fourier_"*string(d)]=getproperty(pattern.fourier,d)
    end
    if !isnothing(obs.onsite_density)
        data["onsite_density_occupation_basis"]=Matrix{Any}(obs.onsite_density)
    end
    for field in (:ctmrg_seconds,:bond_observables_seconds,:onsite_observables_seconds)
        data["timing_"*string(field)]=sum(r->getproperty(r,field),obs.ctm.history)
    end
    return data
end
