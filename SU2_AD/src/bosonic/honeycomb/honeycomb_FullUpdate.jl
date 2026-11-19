# Reuse the existing square two-site environment, reduced tensors and optimizer.
if !isdefined(@__MODULE__,:SquareJ1FullUpdateSettings)
    include("../square/full_update_J1.jl")
end
if !isdefined(@__MODULE__,:SquareJ1CellBond)
    include("../square/full_update_J1_cell.jl")
end

"""Honeycomb bonds in the DOWN-increasing CTMRG coordinates used by square FU.
The conversion from the SU cell is convert_honeycomb_to_iPEPS(). Missing
vertical edges are never updated and keep their one-dimensional spaces.
"""
function honeycomb_fu_bond_groups(A)
    nx,ny=_square_fu_validate_cell(A);
    iseven(nx) && iseven(ny) || throw(ArgumentError("use an even honeycomb cell"));
    groups=[SquareJ1CellBond[] for _=1:3];
    for x=1:nx,y=1:ny
        push!(groups[isodd(x) ? 1 : 2],SquareJ1CellBond(:x,CartesianIndex(x,y),CartesianIndex(mod1(x+1,nx),y)));
        active=iseven(x+y);D=dim(space(A[x,y],2));
        if active
            push!(groups[3],SquareJ1CellBond(:y,CartesianIndex(x,y),CartesianIndex(x,mod1(y+1,ny))));
        else
            D==1 || throw(ArgumentError("missing vertical bond ($x,$y) must stay one-dimensional"));
        end
    end
    for group in groups
        sites=[s for b in group for s in (b.site1,b.site2)];
        length(unique(sites))==length(sites) || error("overlapping honeycomb bond group");
    end
    return groups
end

"""Select the leading singular vectors IN EACH requested irrep, keeping its
original multiplicity. This fixes the bond representation, not its entries.
Zero singular values do not delete requested quantum-number sectors.
"""
function _honeycomb_fu_fixed_factor(M,target_space)
    U,S,V=tsvd(M);
    P=TensorMap{eltype(S)}(undef,space(S,1),target_space);fill!(P,0);
    for (c,b) in blocks(P)
        for k=1:min(size(b)...);b[k,k]=1;end
    end
    Iold=unitary(target_space,target_space);
    norm(P'*P-Iold)<1e-12*max(1,norm(Iold)) ||
        throw(SpaceMismatch("the reduced SVD cannot accommodate the requested fixed bond space"));
    u=U*P;s=P'*S*P;v=P'*V;
    norm(s)>0 || error("zero overlap with the fixed virtual space");
    return permute(u*sqrt(s),(1,),(3,2)),permute(sqrt(s)*v,(1,3),(2,)),s
end

function _honeycomb_fu_initializer(policy,Dmax)
    policy in (:tensorkit,:fixed_space) || throw(ArgumentError("space_policy must be :tensorkit or :fixed_space"));
    return function (M,k1,k2)
        if policy==:tensorkit
            return _square_fu_factor_bond(M;truncation=truncdim(Dmax));
        end
        target=domain(k1)[1];
        dim(target)<=Dmax || throw(ArgumentError("fixed input space exceeds Dmax"));
        return _honeycomb_fu_fixed_factor(M,target)
    end
end

"""One genuine environment-aware two-site FU; A is in CTMRG coordinates.
The untruncated gated state remains the fitting target for BOTH policies.
"""
function honeycomb_full_update_bond(A,environment,gate,bond;
        settings=SquareJ1FullUpdateSettings(multiplet_tol=0.),space_policy=:fixed_space)
    bond in reduce(vcat,honeycomb_fu_bond_groups(A)) ||
        throw(ArgumentError("attempt to update a missing honeycomb edge"));
    nx,ny=size(A);old1=A[bond.site1];old2=A[bond.site2];
    rho= _square_fu_two_site_density_cell(environment.CTM,old1,old1,old2,old2,bond,nx,ny);
    z=_square_fu_rho_trace(rho);
    isfinite(z) && abs(z)>0 || error("zero/nonfinite reference CTM norm at $bond");
    # Independently normalized CTM corners/edges can give an overall phase.
    # Apply ONE fixed scale to target norm, candidate norm AND cross overlap.
    # Do not take separate absolute values or change this scale in the solver.
    density_scale=inv(z);
    a,b,report=square_J1_full_update_cell_bond(A,environment,gate,bond;settings,
        bond_initializer=_honeycomb_fu_initializer(space_policy,settings.Dmax),
        compare_old_candidate=space_policy==:fixed_space,density_scale);
    isfinite(report.fidelity) && -1e-8<=report.fidelity<=1+1e-8 ||
        error("nonphysical CTM fidelity $(report.fidelity); candidate not accepted");
    report.loss_final<=report.loss_initial+1e-10 || error("FU optimizer increased its local loss");
    if space_policy==:fixed_space
        report.new_bond_space==report.old_bond_space || error("fixed-space FU changed the bond representation");
    end
    return a,b,merge(report,(space_policy=space_policy,))
end

function honeycomb_fu_environment(A,chi,setting;require_converged=true)
    global multiplet_tol=0.;global projector_trun_tol=setting.CTM_trun_tol;
    global algrithm_CTMRG_settings=Algrithm_CTMRG_settings(CTM_cell_ite_method="continuous_update");
    setting.CTM_trun_svd && throw(ArgumentError("use CTM_trun_svd=false for SU4 full update"));
    # The legacy CTMRG omits its iteration/error return values unless this
    # flag is enabled; convergence must not depend on the verbosity choice.
    local_setting=deepcopy(setting);local_setting.CTM_conv_info=true;
    env=_square_fu_environment_cell(A,chi,local_setting);
    if require_converged
        !ismissing(env.ite_err) && isfinite(env.ite_err) && env.ite_err<setting.CTM_conv_tol ||
            error("CTMRG did not converge: chi=$chi, corner error=$(env.ite_err)");
    end
    return env
end

"""Measure only the 3N/2 real honeycomb edges in the SAME CTM used by FU."""
function honeycomb_energy_FU(A,environment;J=1.,check_physical=true)
    H=Hamiltonian_honeycomb_SU4(space(A[1],5);J=1.,convention=:permutation);
    nx,ny=size(A);rows=NamedTuple[];
    for b in reduce(vcat,honeycomb_fu_bond_groups(A))
        a1=A[b.site1];a2=A[b.site2];
        rho=_square_fu_two_site_density_cell(environment.CTM,a1,a1,a2,a2,b,nx,ny);
        z=_square_fu_rho_trace(rho);
        isfinite(z) && abs(z)>0 || error("zero/nonfinite CTM bond norm at $b");
        value=(@tensor rho[1,2,3,4]*H[1,2,3,4])/z;
        physical=isfinite(value) && abs(imag(value))<=1e-8 && -1-1e-8<=real(value)<=1+1e-8;
        check_physical && !physical && error("nonphysical swap expectation $value at $b");
        push!(rows,(direction=b.direction,site1=Tuple(b.site1),site2=Tuple(b.site2),P=real(value),physical=physical));
    end
    e=sum(r.P for r in rows)/(nx*ny);
    return (E_P=e,E_H=J*e,bonds=rows,corner_error=environment.ite_err,iterations=environment.ite_num)
end

"""Second-order sweep: x1/2,x2/2,y,x2/2,x1/2 (cyclically rotated).
After EVERY accepted bond rebuild CTMRG, following the existing square FU.
An exception leaves the caller's tensors unchanged; no lambda is introduced.
"""
function honeycomb_full_update_sweep(A,chi,dt;J=1.,Dmax=9,space_policy=:fixed_space,
        settings=SquareJ1FullUpdateSettings(Dmax=Dmax,multiplet_tol=0.),
        ctm_setting=LS_CTMRG_settings(),step_offset=0,observer=nothing)
    settings.Dmax==Dmax || throw(ArgumentError("inconsistent Dmax and FU settings"));
    dt>0 || throw(ArgumentError("dt must be positive"));
    groups=honeycomb_fu_bond_groups(A);current=copy(A);
    H=Hamiltonian_honeycomb_SU4(space(A[1],5);J,convention=:permutation);
    if iszero(J)
        gate=unitary(codomain(H),domain(H));half=gate;
    else
        gate,half=trotter_gate(H,dt);
    end
    env=honeycomb_fu_environment(current,chi,ctm_setting);
    honeycomb_energy_FU(current,env;J); # Validate the initial measurement, including all real bonds.
    a,b,c=ntuple(k->mod1(k+step_offset,3),3);reports=NamedTuple[];
    for (layer,(group,G)) in enumerate(((a,half),(b,half),(c,gate),(b,half),(a,half)))
        for bond in groups[group]
            left,right,report=honeycomb_full_update_bond(current,env,G,bond;settings,space_policy);
            candidate=copy(current);candidate[bond.site1]=left;candidate[bond.site2]=right;
            nextenv=honeycomb_fu_environment(candidate,chi,ctm_setting);
            current=candidate;env=nextenv;
            item=merge(report,(layer=layer,group=group,corner_error=env.ite_err,ctm_iterations=env.ite_num));
            push!(reports,item);isnothing(observer) || observer(item);
        end
    end
    return current,env,reports
end
