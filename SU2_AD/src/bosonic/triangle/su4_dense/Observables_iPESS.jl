# Output and bond-coordinate helpers; contractions live in src/bosonic.
"""
All 3*Lx*Ly independent bonds, oriented from a site in cell (0,0).
The physical lattice vectors enclose 120 degrees: x=(1,0), y=(-1/2,sqrt(3)/2),
diagonal=x+y. target_cell retains the winding lost by wrapped site labels.
For 2x2, A=(1,1), B=(2,1), C=(1,2), D=(2,2).
"""
function nearest_neighbor_bonds(obs)
    nx,ny=obs.cell
    label(x,y) = (nx,ny)==(2,2) ? string("ABCD"[x+nx*(y-1)]) : "s$(x)_$(y)"
    bonds=NamedTuple[]
    for y=1:ny,x=1:nx,d in (:x,:y,:diagonal)
        dx,dy=d==:x ? (1,0) : d==:y ? (0,1) : (1,1)
        tx,ty=mod1(x+dx,nx),mod1(y+dy,ny)
        sx,sy=mod1(x+dx,nx),y # Convert physical-origin indexing to B/simplex indexing.
        at(field)=getproperty(getproperty(obs,field),d)[sx,sy]
        push!(bonds,(id=length(bonds)+1,direction=d,source=(x,y),target=(tx,ty),
            source_label=label(x,y),target_label=label(tx,ty),
            target_cell=(fld(x+dx-1,nx),fld(y+dy-1,ny)),displacement=(dx,dy),simplex=(sx,sy),
            exchange=at(:exchange),coupling=at(:coupling),generator_dot=at(:generator_dot),
            connected_generator_dot=at(:connected_generator_dot),bond_energy=at(:bond_energy)))
    end
    return bonds
end

function bond_table_matrix(obs)
    bonds=nearest_neighbor_bonds(obs)
    columns=["bond_id","direction_code","source_x","source_y","target_x","target_y",
        "target_cell_x","target_cell_y","displacement_x","displacement_y","simplex_x","simplex_y",
        "exchange","coupling","generator_dot","connected_generator_dot","bond_energy"]
    A=zeros(length(bonds),length(columns))
    for b in bonds
        code=findfirst(==(b.direction),(:x,:y,:diagonal))
        A[b.id,:]=[b.id,code,b.source...,b.target...,b.target_cell...,b.displacement...,b.simplex...,
            b.exchange,b.coupling,b.generator_dot,b.connected_generator_dot,b.bond_energy]
    end
    return A,columns
end

"""Bond maps indexed by physical origin, and normalized spatial Fourier coefficients.
These are diagnostics, not automatic phase labels. Directional means are separated
from translation-breaking modulation. For 2x2, indices (2,1),(1,2),(2,2) are
q=(pi,0),(0,pi),(pi,pi), respectively; (1,1) is the directional mean.
"""
function bond_pattern_diagnostics(obs)
    nx,ny=obs.cell
    maps=(x=zeros(nx,ny),y=zeros(nx,ny),diagonal=zeros(nx,ny))
    for b in nearest_neighbor_bonds(obs)
        getproperty(maps,b.direction)[b.source...]=b.coupling
    end
    qx=2pi .* collect(0:nx-1)./nx
    qy=2pi .* collect(0:ny-1)./ny
    fourier=map(maps) do A
        [sum(A[x,y]*exp(-im*(kx*(x-1)+ky*(y-1))) for x=1:nx,y=1:ny)/(nx*ny) for kx in qx,ky in qy]
    end
    ranges=map(A -> maximum(A)-minimum(A),maps)
    means=collect(values(obs.means.coupling))
    return (;origin_coupling=maps,qx,qy,fourier,modulation_range=ranges,
            direction_anisotropy=maximum(means)-minimum(means))
end

function print_observables(io::IO,obs)
    if obs.environment==:ctmrg
        println(io,"Dense SU(4) triangle iPESS: CTMRG environment")
        println(io,"chi=",obs.ctm.chi," corner_converged=",obs.ctm.corner_converged,
                " iteration_converged=",obs.ctm.iteration_converged," chi_converged=",obs.ctm.chi_converged)
    else
        println(io,"Dense SU(4) triangle iPESS: simple-lambda environment (not CTMRG)")
    end
    println(io,"cell=",obs.cell," J=",obs.J," convention=",obs.convention)
    @printf(io,"energy/site = %.16g\nenergy/bond = %.16g\n",obs.energy_per_site,obs.energy_per_bond)
    println(io,"direction         <Pij>           <Pij>-1/4      generator_dot    connected_dot     bond_energy")
    for d in keys(obs.exchange)
        @printf(io,"%-9s %16.10g %16.10g %16.10g %16.10g %16.10g\n",string(d),
                getproperty(obs.means.exchange,d),getproperty(obs.means.coupling,d),
                getproperty(obs.means.generator_dot,d),
                getproperty(obs.means.connected_generator_dot,d),getproperty(obs.means.bond_energy,d))
    end
    println(io,"All independent bonds: source is in cell (0,0); target cell is explicit.")
    println(io," id  direction   source -> target[cell_dx,cell_dy]      <Pij>        <Pij>-1/4       bond_energy")
    for b in nearest_neighbor_bonds(obs)
        endpoint="$(b.source_label) -> $(b.target_label)[$(b.target_cell[1]),$(b.target_cell[2])]"
        @printf(io,"%3d  %-9s %-30s %14.9f %14.9f %14.9f\n",b.id,string(b.direction),endpoint,b.exchange,b.coupling,b.bond_energy)
    end
    diag=bond_pattern_diagnostics(obs)
    println(io,"Direction-mean anisotropy (max-min): ",diag.direction_anisotropy)
    println(io,"Within-direction spatial ranges: ",diag.modulation_range)
    if obs.cell==(2,2)
        println(io,"Spatial Fourier coefficients of coupling (physical-origin convention):")
        println(io,"wavevector          x              y             diagonal")
        for (i,j,q) in ((2,1,"(pi,0)"),(1,2,"(0,pi)"),(2,2,"(pi,pi)"))
            @printf(io,"%-10s %14.9f %14.9f %14.9f\n",q,real(diag.fourier.x[i,j]),real(diag.fourier.y[i,j]),real(diag.fourier.diagonal[i,j]))
        end
    end
    return nothing
end
print_observables(obs) = print_observables(stdout,obs)

"""Save full observables as .jls, human-readable summary as .txt and per-bond .csv."""
function save_observables(prefix::AbstractString,obs)
    paths = (data=prefix*".jls",summary=prefix*".txt",bonds=prefix*".csv",
             full_bonds=prefix*"_bonds.csv",pattern=prefix*"_pattern.csv")
    serialize(paths.data,obs)
    open(paths.summary,"w") do io
        print_observables(io,obs)
    end
    nx,ny = obs.cell
    open(paths.bonds,"w") do io
        println(io,"simplex_x,simplex_y,direction,site_i_x,site_i_y,site_j_x,site_j_y,exchange,coupling,generator_dot,connected_generator_dot,bond_energy")
        for y=1:ny,x=1:nx
            endpoints = (x=((mod1(x-1,nx),y),(x,y)),
                         y=((x,y),(x,mod1(y+1,ny))),
                         diagonal=((mod1(x-1,nx),y),(x,mod1(y+1,ny))))
            for d in keys(obs.exchange)
                (ix,iy),(jx,jy) = getproperty(endpoints,d)
                row = (x,y,string(d),ix,iy,jx,jy,
                       getproperty(obs.exchange,d)[x,y],getproperty(obs.coupling,d)[x,y],
                       getproperty(obs.generator_dot,d)[x,y],
                       getproperty(obs.connected_generator_dot,d)[x,y],getproperty(obs.bond_energy,d)[x,y])
                println(io,join(row,','))
            end
        end
    end
    open(paths.full_bonds,"w") do io
        println(io,"bond_id,direction,source_label,target_label,source_x,source_y,target_x,target_y,target_cell_x,target_cell_y,displacement_x,displacement_y,simplex_x,simplex_y,exchange,coupling,generator_dot,connected_generator_dot,bond_energy,environment")
        for b in nearest_neighbor_bonds(obs)
            println(io,join((b.id,string(b.direction),b.source_label,b.target_label,b.source...,b.target...,
                b.target_cell...,b.displacement...,b.simplex...,b.exchange,b.coupling,
                b.generator_dot,b.connected_generator_dot,b.bond_energy,string(obs.environment)),','))
        end
    end
    diag=bond_pattern_diagnostics(obs)
    open(paths.pattern,"w") do io
        println(io,"qx,qy,coupling_x_real,coupling_x_imag,coupling_y_real,coupling_y_imag,coupling_diagonal_real,coupling_diagonal_imag")
        for j in eachindex(diag.qy),i in eachindex(diag.qx)
            z=(diag.fourier.x[i,j],diag.fourier.y[i,j],diag.fourier.diagonal[i,j])
            println(io,join((diag.qx[i],diag.qy[j],real(z[1]),imag(z[1]),real(z[2]),imag(z[2]),real(z[3]),imag(z[3])),','))
        end
    end
    return paths
end
