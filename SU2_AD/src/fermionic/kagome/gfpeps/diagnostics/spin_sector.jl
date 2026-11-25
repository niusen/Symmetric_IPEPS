# Independent finite-volume Gaussian/Fock audit; no CTMRG or TensorKit required.
# julia --startup-file=no audit_spin_sector.jl
using LinearAlgebra, Random, MAT, TOML, Test

"""Occupation probabilities of a pure full-Nambu covariance P=<z†z>.

Rows are (b,b†). Expand Pfaffians after a particle/hole chart change.
Only probabilities are needed here; Fock reordering phases drop out.
"""
function occupation_probabilities(P)
    n=size(P,1)÷2
    @assert size(P)==(2n,2n) && n<=20
    @assert norm(P-P')<1e-9 && norm(P*P-P)<1e-8
    swap=vcat(n+1:2n,1:n)
    @assert norm(P+conj(P[swap,swap])-I)<1e-8
    w=eigen(Hermitian(P)).vectors[:,n+1:2n]
    base=sum(1<<(i-1) for i in 1:n if real(P[i,i])>.5;init=0)
    rng=MersenneTwister(917)
    candidates=vcat([base,0],[xor(base,1<<i) for i in 0:n-1],rand(rng,0:(1<<n)-1,128))
    best=-Inf; reference=0; frame=w
    for mask in candidates
        order=collect(1:2n)
        for i in 1:n
            if !iszero(mask & (1<<(i-1)))
                order[i],order[n+i]=order[n+i],order[i]
            end
        end
        t=w[order,:]; score=minimum(svdvals(t[n+1:2n,:]))
        if score>best
            best=score;reference=mask;frame=t
        end
    end
    @assert best>1e-10
    lower=frame[n+1:2n,:]
    Z=frame[1:n,:]/lower
    @assert norm(Z+transpose(Z))<1e-7
    Z=(Z-transpose(Z))/2
    pf=zeros(ComplexF64,1<<n)
    pf[1]=sqrt(abs(det(lower))) # normalized reference-vacuum amplitude
    for mask in 1:(1<<n)-1
        isodd(count_ones(mask)) && continue
        first=mask & -mask; i=trailing_zeros(first)+1
        rest=xor(mask,first); walk=rest; sign=1
        while walk!=0
            bit=walk & -walk; j=trailing_zeros(bit)+1
            pf[mask+1]+=sign*Z[i,j]*pf[xor(rest,bit)+1]
            walk=xor(walk,bit);sign=-sign
        end
    end
    probability=[abs2(pf[xor(mask,reference)+1]) for mask in 0:(1<<n)-1]
    @assert abs(sum(probability)-1)<1e-8
    probability
end

function number_moments(P)
    n=size(P,1)÷2;C=P[1:n,1:n];F=P[1:n,n+1:2n]
    # Wick: Var(N_b)=Tr(C-C²)+Tr(F†F), including anomalous correlations.
    (;mean=real(tr(C)),variance=real(tr(C-C*C))+sum(abs2,F),anomalous_norm=norm(F))
end

function projected_sectors(probability)
    modes=trailing_zeros(length(probability));@assert iseven(modes)
    sites=modes÷2
    # b mode order = all up particles, then all down holes.
    # Physical up: both occupied; physical down: both empty.
    weights=zeros(sites+1)
    for spins in 0:(1<<sites)-1
        mask=spins | (spins<<sites)
        weights[count_ones(spins)+1]+=probability[mask+1]
    end
    success=sum(weights);@assert success>1e-14
    weights/=success
    ratio=sum(weights[k+1]*(2k-sites)/sites for k in 0:sites)
    (;weights,success,mz_over_saturation=ratio)
end


function audit_saved_sector(family,D;directory_override=nothing)
    directory=isnothing(directory_override) ? normpath(joinpath(@__DIR__,"../../../../../Kagome/fermionic/iPESS/results/gfpeps_validation","$(family)_D$D")) : directory_override
    input=matread(joinpath(directory,"charge_audit_input.mat"))
    P=input["rho_zero"];prob=occupation_probabilities(P);mom=number_moments(P)
    numbers=count_ones.(0:length(prob)-1)
    mean_fock=sum(prob.*numbers)
    var_fock=sum(prob.*(numbers.-mean_fock).^2)
    @test mean_fock≈mom.mean atol=1e-8
    @test var_fock≈mom.variance atol=1e-8
    sectors=projected_sectors(prob)
    # Resolve projection within each total-Sz sector on the SAME torus.
    nb_weights=[sum(prob[numbers.==q]) for q in 0:18]
    projected_nb_weights=zeros(19)
    projected_nb_weights[1:2:19]=sectors.success.*sectors.weights
    @test all(projected_nb_weights.<=nb_weights.+1e-13)
    # Independently select Nb=10 before projection. The physical single-
    # occupancy selection must leave it in Sz=1/2, i.e. M/Msat=1/9.
    fixed=prob.*(numbers.==10);fixed/=sum(fixed)
    fixed_sectors=projected_sectors(fixed)
    @test fixed_sectors.mz_over_saturation≈1/9 atol=1e-14
    @test fixed_sectors.weights[6]≈1 atol=1e-14
    q_values=(collect(0:18).-9)./9
    before_ratio=(mean_fock-9)/9
    joint_q=dot(q_values,projected_nb_weights)
    covariance_shift=(joint_q-before_ratio*sectors.success)/sectors.success
    @test sectors.mz_over_saturation-before_ratio≈covariance_shift atol=1e-13
    bulk=number_moments.(eachslice(input["rho_mesh"];dims=3))
    grid=size(input["rho_mesh"],3)
    nb=sum(r.mean for r in bulk)/grid
    variance=sum(r.variance for r in bulk)/grid
    reference=matread(joinpath(directory,"reference_N192.mat"))
    ratio=reference["n_up_per_site"]-reference["n_down_per_site"]
    report=Dict(
        "family"=>family,"D"=>D,"source_sha256"=>input["source_sha256"],
        "target_mz_over_saturation"=>1/9,
        "unprojected_bulk_mz_over_saturation_N192"=>ratio,
        "bulk_audit_mesh"=>Int(input["mesh_n"]),
        "bulk_Nb_per_magnetic_cell"=>nb,
        "bulk_Nb_variance_per_magnetic_cell"=>variance,
        "bulk_Sz_variance_per_magnetic_cell"=>variance/4,
        "bulk_max_anomalous_b_norm"=>maximum(r.anomalous_norm for r in bulk),
        "torus_sites"=>9,"torus_boundary"=>"one magnetic cell, periodic, k=(0,0)",
        "torus_unprojected_Nb_mean"=>mean_fock,
        "torus_unprojected_Nb_variance"=>var_fock,
        "torus_unprojected_Nb_sector_probabilities"=>nb_weights,
        "torus_unprojected_mz_over_saturation"=>before_ratio,
        "torus_projection_magnetization_shift"=>covariance_shift,
        "torus_fixed_Nb10_then_projected_mz_over_saturation"=>fixed_sectors.mz_over_saturation,
        "torus_gutzwiller_probability"=>sectors.success,
        "torus_projected_Nup_sector_probabilities"=>sectors.weights,
        "torus_projected_target_sector_weight"=>sectors.weights[6],
        "torus_projected_mz_over_saturation"=>sectors.mz_over_saturation,
        "torus_Nb_mean_fock_minus_wick"=>mean_fock-mom.mean,
        "torus_Nb_variance_fock_minus_wick"=>var_fock-mom.variance,
        "note"=>"Finite torus diagnoses charge sectors; its projected magnetization is not the infinite-lattice CTMRG reference.")
    open(io->TOML.print(io,report),joinpath(directory,"spin_sector_audit.toml"),"w")
    println(family," D=",D,": bulk M/Msat=",ratio,"; Var(Nb)/cell=",variance,
        "; torus projected M/Msat=",sectors.mz_over_saturation,
        "; target-sector weight=",sectors.weights[6]);flush(stdout)
    report
end
