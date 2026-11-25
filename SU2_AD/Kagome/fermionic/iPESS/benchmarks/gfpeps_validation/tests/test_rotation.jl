include("../run.jl")
using Test
let C=KagomeFermionicCTM.Legacy
    V=Rep[ℤ₂](0=>1,1=>1)
    a=TensorKit.permute(randn(ComplexF64,V*V,V*V),((1,2,3,4),()))
    @testset "Generic cell rotations preserve legacy results" begin
        for (nx,ny) in ((1,1),(2,1),(1,2),(2,2),(3,2),(3,3))
            C.Lx=nx; C.Ly=ny
            cell=ntuple(x->ntuple(y->(x+3y)*a,ny),nx)
            old=invoke(C.rotate_AA_cell,Tuple{Any,Any},cell,true)
            new=C.rotate_AA_cell(cell,true)
            @test all(norm(getfield(old,d)[x][y]-getfield(new,d)[x][y])==0
                      for d in 1:4,x in 1:nx,y in 1:ny)
        end
        C.Lx=3;C.Ly=1
        cell=ntuple(x->(x*a,),3)
        new=C.rotate_AA_cell(cell,true)
        @test all(norm(getfield(new,d)[x][1]-getfield(C.rotate_AA(x*a,true),d))==0
                  for d in 1:4,x in 1:3)
    end
end

# Regression: dynamic getfield(..., direction) had correct forward values but
# raised UndefVarError in Zygote. All four rotations preserve the tensor norm.
using Zygote
let C=KagomeFermionicCTM.Legacy
    V=Rep[ℤ₂](0=>1,1=>1)
    a=TensorKit.permute(randn(ComplexF64,V*V,V*V),((1,2,3,4),()))
    function rotation_loss(t)
        r=C.rotate_AA_cell(((t,), (2t,), (3t,)),true)
        squared_norm(cell)=sum(map(col->sum(map(x->norm(x)^2,col)),cell))
        squared_norm(r.T1)+squared_norm(r.T2)+squared_norm(r.T3)+squared_norm(r.T4)
    end
    @testset "Generic magnetic-cell rotation pullback" begin
        g=Zygote.gradient(rotation_loss,a)[1]
        @test g≈112a atol=1e-10 rtol=1e-10
        direction=randn(ComplexF64,space(a));direction/=norm(direction)
        epsilon=1e-5
        fd=(rotation_loss(a+epsilon*direction)-rotation_loss(a-epsilon*direction))/(2epsilon)
        @test real(dot(g,direction))≈fd atol=1e-7 rtol=1e-7
    end
end
