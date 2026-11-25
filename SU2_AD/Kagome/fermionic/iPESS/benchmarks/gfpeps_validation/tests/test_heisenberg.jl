include("../run.jl")
using Test
const HG=KagomeGFPEPSValidation

@testset "Heisenberg geometry and all spin components" begin
    raw=initial_iPESS(3,2,Rep[ℤ₂](0=>1))
    expected=zeros(3,2,3,3)
    for x=1:3,y=1:2,s=1:3
        theta=0.19+0.27x+0.31y+0.22s;phi=0.17x-0.37y+0.53s
        expected[x,y,s,:]=[sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)]/2
        a=zeros(ComplexF64,1,1,4)
        a[1,1,2]=cos(theta/2);a[1,1,1]=cis(phi)*sin(theta/2)
        raw.B[x,y,s]=TensorMap(a,codomain(raw.B[x,y,s]),domain(raw.B[x,y,s]))
    end
    state=gutzwiller_project(raw);A=convert_iPESS_to_iPEPS(state);_,_,U=build_A_kagome(state,1,1)
    r=measure_heisenberg(A,U;chi=2,maxiter=12,tol=1e-9,verbose=false)
    @test r.converged
    @test maximum(abs.(r.magnetization-expected))<1e-10
    for x=1:3,y=1:2
        sites=((x,y,1),(x,y,2),(x,y,3))
        dn=((mod1(x+1,3),y,1),(x,y,2),(x,mod1(y+1,2),3))
        for (e,(i,j)) in enumerate(((1,2),(2,3),(3,1)))
            @test r.up_bonds[x,y,e]≈dot(expected[sites[i]...,:],expected[sites[j]...,:]) atol=1e-10
            @test r.down_bonds[x,y,e]≈dot(expected[dn[i]...,:],expected[dn[j]...,:]) atol=1e-10
        end
    end
    @test r.energy_per_site≈real(sum(r.up_bonds)+sum(r.down_bonds))/18
    @test r.mz_over_saturation≈2sum(expected[:,:,:,3])/18
end

@testset "Entangled intra-cell singlet" begin
    S=projected_spin_space();U=unitary(fuse(S*S*S),S*S*S);V=Rep[ℤ₂](0=>1)
    psi=zeros(ComplexF64,2,2,2);psi[1,2,1]=1/sqrt(2);psi[2,1,1]=-1/sqrt(2)
    p=reshape(convert(Array,U),8,8)*vec(psi)
    a=TensorMap(reshape(p,1,1,1,1,8),V*V'*V'*V*space(U,1),one(V))
    r=measure_heisenberg(((a,),),U;chi=2,maxiter=12,tol=1e-9,verbose=false)
    @test r.up_bonds[1,1,:]≈[-0.75,0,0] atol=1e-10
    @test maximum(abs.(r.down_bonds))<1e-10
    @test r.energy_per_site≈-0.25 atol=1e-10
    @test r.magnetization_per_site≈[0,0,1/6] atol=1e-10
    @test r.max_imag<1e-12
end
