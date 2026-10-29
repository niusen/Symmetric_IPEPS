# Total-state budget, retaining whole SU(N) multiplets. The objective is the
# retained squared tensor norm: sum_r dim(r) * sum_k s[r,k]^2.
# A group-knapsack solution is needed; a singular-value threshold is not in
# general optimal when irreps have different dimensions.

const HoneycombTruncationBase=isdefined(TensorKit,:MatrixAlgebraKit) ?
    TensorKit.MatrixAlgebraKit.TruncationStrategy : TensorKit.TruncationScheme;

struct WholeMultipletBudget <: HoneycombTruncationBase
    maxdim::Int
    function WholeMultipletBudget(maxdim::Integer)
        maxdim>0 || throw(ArgumentError("multiplet dimension budget must be positive"));
        new(Int(maxdim))
    end
end

function optimal_multiplet_counts(spectra,maxdim)
    cs=collect(keys(spectra));
    isempty(cs) && throw(ArgumentError("cannot truncate an empty singular spectrum"));
    all(v->isfinite(v),Iterators.flatten(spectra[c] for c in cs)) ||
        throw(ArgumentError("nonfinite singular values"));
    scale=maximum(maximum(abs,spectra[c];init=0.0) for c in cs);
    scale>0 || throw(ArgumentError("cannot truncate a zero-norm bond"));
    capacity=min(maxdim,sum(Int(dim(c))*length(spectra[c]) for c in cs));
    score=fill(-Inf,capacity+1); score[1]=0.0;
    choices=zeros(Int,length(cs),capacity+1);
    for (j,c) in enumerate(cs)
        d=Int(dim(c));
        n=min(length(spectra[c]),div(capacity,d));
        benefits=vcat(0.0,cumsum([d*abs2(spectra[c][k]/scale) for k=1:n]));
        next_score=fill(-Inf,capacity+1);
        for cost=0:capacity
            for k=0:min(n,div(cost,d))
                candidate=score[cost-k*d+1]+benefits[k+1];
                if candidate>next_score[cost+1]
                    next_score[cost+1]=candidate;
                    choices[j,cost+1]=k;
                end
            end
        end
        score=next_score;
    end
    best,index=findmax(score);
    best>0 || throw(ArgumentError("Dmax=$maxdim cannot retain any nonzero complete multiplet"));
    # findmax chooses the smallest dimension on a tie, avoiding zero-weight padding.
    cost=index-1;
    counts=zeros(Int,length(cs));
    for j=length(cs):-1:1
        counts[j]=choices[j,cost+1];
        cost-=counts[j]*Int(dim(cs[j]));
    end
    return cs,counts
end

if isdefined(TensorKit,:MatrixAlgebraKit)
    function TensorKit.MatrixAlgebraKit.findtruncated(values::TensorKit.SectorVector,
                                                     trunc::WholeMultipletBudget)
        cs,counts=optimal_multiplet_counts(values,trunc.maxdim);
        return TensorKit.SectorDict(c=>Base.OneTo(k) for (c,k) in zip(cs,counts))
    end
else
    # TensorKit 0.14--0.16 interface. Same selector; no second factorization.
    function TensorKit._compute_truncdim(values,trunc::WholeMultipletBudget,p::Real=2)
        p==2 || throw(ArgumentError("WholeMultipletBudget optimizes the Frobenius norm (p=2)"));
        cs,counts=optimal_multiplet_counts(values,trunc.maxdim);
        return TensorKit.SectorDict(c=>k for (c,k) in zip(cs,counts))
    end
end

# The reused two-site functions call truncdim(Dmax). Our own argument type
# changes only honeycomb SU calls, not TensorKit's integer truncation policy.
if isdefined(TensorKit,:truncdim)
    @eval TensorKit.truncdim(trunc::WholeMultipletBudget)=trunc
else
    truncdim(trunc::WholeMultipletBudget)=trunc
end
