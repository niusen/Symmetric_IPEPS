# Small-cell diagnostic for the first invalid bond; no CTMRG or long evolution.
using LinearAlgebra,TensorKit,SUNRepresentations,Dates
include("../../../src/bosonic/honeycomb/star/StarSimpleUpdateRun.jl")
using .HoneycombStarRun
using .HoneycombStarRun.HoneycombSU4
const HC=HoneycombStarRun.HoneycombSU4
BLAS.set_num_threads(1)

function bond_info(s)
    (D=dim(space(s,1)),stored=sum(length(diag(b)) for (_,b) in blocks(s);init=0),
        norm=norm(s),sectors=[(Tuple(Int.(SUNRepresentations.dynkin_label(c))),size(b,1)) for (c,b) in blocks(s)])
end

function diagnose_star_update(;L=2,Dmax=24,dt=.01,sweeps=1,truncation=:legacy)
    truncation in (:legacy,:optimal) || throw(ArgumentError("use :legacy or :optimal"));
    budget=truncation==:legacy ? Dmax : WholeMultipletBudget(Dmax);
    T,lx,ly,_=initial_honeycomb_star_iPEPS(Lx=L,Ly=L)
    H=Hamiltonian_honeycomb_SU4(space(T[1],5);convention=:permutation)
    gate,half=HC.trotter_gate(H,dt)
    for sweep=1:sweeps
        a,b,c=ntuple(k->(:x,:y,:z)[mod1(k+sweep-1,3)],3)
        for (direction,g) in ((a,half),(b,half),(c,gate),(b,half),(a,half))
            for bond in honeycomb_bonds(L,L)
                bond.direction==direction || continue
                x,y=bond.site1
                println("UPDATE ",(sweep=sweep,direction=direction,site=bond.site1));flush(stdout)
                try
                    if bond.axis==:horizontal
                        HC.tebd_xbond(1,T,lx,ly,g,x+.5,y,budget)
                    else
                        HC.tebd_ybond(1,T,lx,ly,g,x,y-.5,budget)
                    end
                    s=bond.axis==:horizontal ? lx[bond.site2...] : ly[bond.site1...]
                    println("OUTPUT ",bond_info(s));flush(stdout)
                    dim(space(s,1))>0 && norm(s)>0 || error("first empty output at $bond")
                catch
                    for i in CartesianIndices(lx)
                        println("BOND ",Tuple(i)," x=",bond_info(lx[i])," y=",bond_info(ly[i]));
                    end
                    flush(stdout);rethrow()
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    if !isempty(ARGS) && first(ARGS)=="--trace"
        folder=length(ARGS)>=2 ? ARGS[2] : joinpath(@__DIR__,"../logs/trace_"*Dates.format(now(),"yyyymmdd_HHMMSS"));
        sweeps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 100;
        result=trace_star_simple_update(;folder,sweeps);
        result.status==:finished || exit(2);
    else
        diagnose_star_update()
    end
end
