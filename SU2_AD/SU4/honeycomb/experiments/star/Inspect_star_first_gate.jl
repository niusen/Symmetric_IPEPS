using LinearAlgebra, TensorKit, SUNRepresentations, JLD2
BLAS.set_num_threads(1);

function first_gate_spectrum()
    T=load(joinpath(@__DIR__,"../../data","star_RVB","initial_tensors_L4.jld2"),"full");
    A=T[1,1];B=T[2,1];p=space(A,5);
    I2=unitary(p*p,p*p);P=TensorKit.permute(I2,((2,1),(3,4)));
    # Direct two-site wavefunction, with identity external lambda factors.
    @tensor psi[:]:=A[-1,-2,a,-3,-4]*B[a,-5,-6,-7,-8];
    for dt in (0.,.001,.01)
        gate=cosh(dt)*I2-sinh(dt)*P;
        @tensor evolved[:]:=psi[-1,-2,-3,a,-5,-6,-7,b]*gate[-4,-8,a,b];
        M=TensorKit.permute(evolved,((1,2,3,4),(5,6,7,8)));
        _,s,_=TensorKit.svd_compact(M);
        largest=maximum(abs(v) for (_,b) in blocks(s) for v in diag(b));
        println("FIRST GATE dt=",dt);flush(stdout);
        rank=0;
        for (c,b) in blocks(s)
            values=sort(abs.(diag(b));rev=true);
            significant=filter(v->v>1e-10*largest,values);
            isempty(significant) && continue;
            rank+=Int(dim(c))*length(significant);
            println("  dynkin=",Tuple(Int.(SUNRepresentations.dynkin_label(c))),
                " dim=",dim(c)," multiplicity=",length(significant)," singular_values=",significant);
            flush(stdout);
        end
        println("  significant_rank=",rank);flush(stdout);
    end
    c4=first(sectors(Rep[SU₄]("4"=>1)));
    c15=first(sectors(Rep[SU₄]("15"=>1)));
    println("FUSION 4 x 15:");
    for c in c4⊗c15
        println("  ",Tuple(Int.(SUNRepresentations.dynkin_label(c)))," dim=",dim(c));
    end
end

first_gate_spectrum();
