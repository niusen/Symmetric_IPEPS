# The local estimate uses the same external lambda factors as the old SU.
# It is not a CTMRG expectation value or a variational energy bound.
function honeycomb_bond_wavefunction(Tset,lx,ly,bond)
    nx,ny=size(Tset); x,y=bond.site1; u,v=bond.site2;
    T1=Tset[x,y]; T2=Tset[u,v];
    if bond.axis==:horizontal
        a=lx[x,y]; b=ly[x,y]; c=ly[x,mod1(y+1,ny)];
        d=ly[u,v]; e=lx[mod1(u+1,nx),v]; f=ly[u,mod1(v+1,ny)];
        @tensor T1[:]:=T1[1,2,-3,3,-5]*a[-1,1]*b[2,-2]*c[-4,3];
        @tensor T2[:]:=T2[-1,1,2,3,-5]*d[1,-2]*e[2,-3]*f[-4,3];
        @tensor psi[:]:=T1[-1,-2,1,-3,-7]*T2[1,-4,-5,-6,-8];
    else
        a=lx[x,y]; b=lx[mod1(x+1,nx),y]; c=ly[x,mod1(y+1,ny)];
        d=lx[u,v]; e=ly[u,v]; f=lx[mod1(u+1,nx),v];
        @tensor T1[:]:=T1[1,-2,2,3,-5]*a[-1,1]*b[2,-3]*c[-4,3];
        @tensor T2[:]:=T2[1,2,3,-4,-5]*d[-1,1]*e[2,-2]*f[3,-3];
        @tensor psi[:]:=T1[-1,1,-2,-3,-7]*T2[-4,-5,-6,1,-8];
    end
    return psi
end

function honeycomb_energy_result(bonds,values,nsite,environment)
    direction_means=map(d->sum(values[k] for k in eachindex(bonds) if bonds[k].direction==d)/(nsite/2),(:x,:y,:z));
    return (environment=environment,energy_per_site=sum(values)/nsite,
        energy_per_bond=sum(values)/length(values),
        bond_energy=(x=direction_means[1],y=direction_means[2],z=direction_means[3]),
        bonds=bonds,energies=values)
end

function honeycomb_energy_simple(parameters,Tset,lambdaxset,lambdayset;convention=:traceless)
    nx,ny=size(Tset); bonds=honeycomb_bonds(nx,ny);
    H=Hamiltonian_honeycomb_SU4(space(Tset[1],5);J=parameters["J"],convention=convention);
    energies=Float64[];
    for bond in bonds
        psi=honeycomb_bond_wavefunction(Tset,lambdaxset,lambdayset,bond);
        @tensor Hpsi[:]:=psi[-1,-2,-3,-4,-5,-6,1,2]*H[-7,-8,1,2];
        denominator=real(dot(psi,psi));
        denominator>0 || error("zero norm on bond $bond");
        push!(energies,real(dot(psi,Hpsi))/denominator);
    end
    return honeycomb_energy_result(bonds,energies,nx*ny,:simple_lambda)
end

"""
CTMRG energy using the old CTMRG_cell, hopping_x/y_no_sign and ob_2x2.
Returns the environment and convergence report for reuse / chi convergence checks.
The old CTMRG cell rotation supports 2×2 and 4×4 brick-wall cells.
"""
function honeycomb_energy_CTM(parameters,Tset,ch,ctm_setting=LS_CTMRG_settings();
                               convention=:traceless,CTM0=nothing)
    nx,ny=size(Tset);
    (nx,ny) in ((2,2),(4,4)) || throw(ArgumentError("old CTMRG supports 2x2 or 4x4 honeycomb cells"));
    ch>0 || throw(ArgumentError("chi must be positive"));
    global Lx=nx; global Ly=ny; global chi=ch;
    global multiplet_tol=0.0;
    global projector_trun_tol=ctm_setting.CTM_trun_tol;
    global algrithm_CTMRG_settings=Algrithm_CTMRG_settings(CTM_cell_ite_method="continuous_update");
    ctm_setting.CTM_trun_svd && throw(ArgumentError("use CTM_trun_svd=false for SU(4)/dense CTMRG"));
    A_cell=convert_honeycomb_to_iPEPS(Tset);
    init=initial_condition(init_type="PBC",reconstruct_CTM=isnothing(CTM0),reconstruct_AA=true);
    CTM,AA,_,_,_,_,ite_num,ite_err=CTMRG_cell(A_cell,ch,init,CTM0,ctm_setting);
    H=Hamiltonian_honeycomb_SU4(space(Tset[1],5);J=parameters["J"],convention=convention);
    u,s,v=tsvd(permute(H,(1,3,),(2,4,)));
    O1=permute(u*s,(3,1,),(2,));
    O2=permute(v,(1,2,),(3,));
    bonds=honeycomb_bonds(nx,ny); energies=Float64[];
    for bond in bonds
        x,y=bond.site1;
        cy=mod1(2-y,ny);
        if bond.axis==:horizontal
            value=hopping_x_no_sign(CTM,O1,O2,A_cell,AA,x-1,cy-1,ctm_setting);
        else
            # SU downward edge becomes CTMRG RU--RD at cy,cy+1.
            value=hopping_y_no_sign(CTM,O1,O2,A_cell,AA,x-2,cy-1,ctm_setting);
        end
        push!(energies,real(value));
    end
    obs=honeycomb_energy_result(bonds,energies,nx*ny,:ctmrg);
    return merge(obs,(CTM=CTM,iterations=ite_num,corner_error=ite_err,
        corner_converged=ite_err<ctm_setting.CTM_conv_tol,chi=ch,))
end
