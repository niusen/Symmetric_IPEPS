include("../run.jl")
using Test
const G=KagomeGFPEPSValidation
const KF=KagomeFermionicZ2

@testset "Physical electron single occupancy in the hole basis" begin
    Q=gutzwiller_projector(); q=convert(Array,Q); o=KF._local_operators()
    nu=o.c[1]'*o.c[1]; nd=o.id-o.c[2]'*o.c[2]
    single=nu+nd-2nu*nd
    @test q'q≈single
    @test q*q'≈Matrix{ComplexF64}(I,2,2)
    @test q[:,1]==[0,1] # physical down
    @test q[:,2]==[1,0] # physical up
    @test iszero(q[:,3:4]) # physical double occupancy and empty state
    s=projected_spin_operators(); sx,sy,sz=map(x->convert(Array,x),(s.Sx,s.Sy,s.Sz))
    @test sx*sy-sy*sx≈im*sz
    @test sx*sx+sy*sy+sz*sz≈3/4*Matrix{ComplexF64}(I,2,2)
    @test q*(nu-nd)*q'/2≈sz
    # Intertwining (stronger than the compressed operator identity): projection
    # cannot transfer amplitude between different total-Sz eigenvalues.
    sz_fermion=(nu-nd)/2
    @test norm(q*sz_fermion-sz*q)<1e-14
    @test norm(q*exp(0.731im*sz_fermion)-exp(0.731im*sz)*q)<1e-14
    @test q*(o.c[1]'*o.c[2]')*q'≈convert(Array,s.Sp)
    @test q*nu*q'+q*nd*q'≈Matrix{ComplexF64}(I,2,2)
end

@testset "Saved D=4,8 imports and projection/blocking commute" begin
    for D in (4,8)
        root=joinpath(ROOT,"Kagome/fermionic/iPESS/results/gfpeps_validation")
        path=joinpath(root,"ipess_D$D","tensors.mat")
        raw,fields=import_gaussian_ipess(path)
        before=deepcopy(raw.B)
        projected,fields2=import_gaussian_ipess(path;gutzwiller=true)
        @test projected isa ProjectedSpinIPESS
        @test fields==fields2
        @test raw.B==before
        @test projected.T_u==raw.T_u && projected.T_d==raw.T_d
        @test all(b->dim(space(b,3))==2,projected.B)
        @test KF.check_state(projected;Dmax=D)===projected
        # Compare amplitudes without normalizing away relative information.
        for x in 1:3
            _,a,U=build_A_kagome(raw,x,1)
            _,b,Us=build_A_kagome(projected,x,1)
            direct,Us2=gutzwiller_project(((a,),),U)
            @test Us==Us2
            @test norm(b-direct[1][1])/norm(b)<2e-12
        end
        A=convert_iPESS_to_iPEPS(projected)
        @test all(col->dim(space(col[1],5))==8,A)
        @test_throws ArgumentError measure_gaussian(projected,fields)
        path=joinpath(root,"ipeps_D$D","tensors.mat")
        A,U,fields=import_gaussian_ipeps(path)
        B,Us,fields2=import_gaussian_ipeps(path;gutzwiller=true)
        # Check the actual fused three-site map, including the importer's U.
        Q=gutzwiller_projector();K=Us*(Q⊗Q⊗Q)*U'
        o=KF._local_operators();nu=o.c[1]'*o.c[1];nd=o.id-o.c[2]'*o.c[2]
        P=KF.physical_space();Id=TensorMap(o.id,P,P);Z=TensorMap((nu-nd)/2,P,P)
        z_before=U*(Z⊗Id⊗Id+Id⊗Z⊗Id+Id⊗Id⊗Z)*U'
        spin=blocked_spin_operators(Us);z_after=sum(spin.Sz)
        residual=norm(K*z_before-z_after*K)
        println("D=",D," fused projection Sz intertwining residual=",residual)
        @test residual<1e-13
        expected,Us2=gutzwiller_project(A,U;normalize=true)
        @test fields==fields2 && Us==Us2
        @test all(x->norm(B[x][1]-expected[x][1])<2e-12,1:3)
        # Independent dense selection before refusion: up=b pair, down=b vacuum.
        for x in 1:3
            a=convert(Array,A[x][1]); fusion=reshape(convert(Array,U),64,64)
            unfused=reshape(reshape(a,:,64)*conj(fusion),D,D,D,D,4,4,4)
            selected=unfused[:,:,:,:,[2,1],[2,1],[2,1]]
            us=reshape(convert(Array,Us),8,8)
            selected=reshape(reshape(selected,:,8)*transpose(us),D,D,D,D,8)
            @test norm(selected/norm(selected)-convert(Array,B[x][1]))<2e-12
        end
        @test_throws ArgumentError measure_gaussian_cell(B,Us,fields)
        @test_throws SpaceMismatch gutzwiller_project(B,Us)
    end
end

@testset "Spin CTMRG product state" begin
    raw=initial_iPESS(1,1,Rep[ℤ₂](0=>1))
    for i in eachindex(raw.B)
        a=zeros(ComplexF64,1,1,4);a[1,1,2]=1
        raw.B[i]=TensorMap(a,codomain(raw.B[i]),domain(raw.B[i]))
    end
    state=gutzwiller_project(raw)
    A=convert_iPESS_to_iPEPS(state);_,_,U=build_A_kagome(state,1,1)
    r=measure_projected_spins(A,U;chi=2,maxiter=12,tol=1e-9,verbose=false)
    @test r.converged
    @test maximum(abs.(r.normalization.-1))<1e-10
    @test maximum(abs.(r.magnetization[:,:,:,1:2]))<1e-10
    @test maximum(abs.(r.magnetization[:,:,:,3].-0.5))<1e-10
    jldsave(joinpath(ROOT,"Kagome/fermionic/iPESS/results/gfpeps_validation/gutzwiller_product_test.jld2");result=r)
end
