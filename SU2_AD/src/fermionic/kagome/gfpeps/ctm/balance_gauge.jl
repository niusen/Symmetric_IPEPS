# Optional, exact virtual-bond gauge conditioning for the forward benchmark.
# No singular values are truncated. On each bond we insert G and inv(G).
function _gauge_axis(a,M,axis)
    order=(axis,filter(!=(axis),Tuple(1:ndims(a)))...)
    b=permutedims(a,order)
    b=reshape(M*reshape(b,size(b,1),:),size(b))
    permutedims(b,invperm(collect(order)))
end

function _bond_balance(a,b,ia,ib;floor=1e-12)
    oa=(filter(!=(ia),Tuple(1:ndims(a)))...,ia)
    ob=(ib,filter(!=(ib),Tuple(1:ndims(b)))...)
    L=reshape(permutedims(a,oa),:,size(a,ia))
    R=reshape(permutedims(b,ob),size(b,ib),:)
    X=L'*L; Y=R*R'; D=size(X,1)
    G=zeros(ComplexF64,D,D)
    # Dense TensorKit Z2 bases list the even sector before the odd sector.
    for indices in (1:D÷2,D÷2+1:D)
        x=Hermitian(X[indices,indices]); y=Hermitian(Y[indices,indices])
        sx=tr(x)/length(indices); sy=tr(y)/length(indices)
        Rx=cholesky(Hermitian(Matrix(x)+floor*real(sx)*I)).U
        Ry=cholesky(Hermitian(Matrix(y)+floor*real(sy)*I)).L
        f=svd(Rx*Ry)
        G[indices,indices]=Rx\(f.U*Diagonal(sqrt.(f.S)))
    end
    # A common scalar cancels between the endpoints; keep both gauges moderate.
    G/=sqrt(norm(G)/norm(inv(G)))
    H=inv(G)
    return transpose(G),H,norm(G*H-I),cond(G)
end

"""Balance local bond Gramians using invertible, parity-even virtual gauges.

The state is unchanged: the two matrices on every contracted bond multiply
to identity. The small Gramian regularizer chooses the gauge only; it does
not truncate or regularize the physical tensor network.
"""
function balance_virtual_gauge(A;sweeps=8,floor=1e-12)
    Lx,Ly=length(A),length(A[1]); dense=[convert(Array,A[x][y]) for x=1:Lx,y=1:Ly]
    residual=0.;maxcondition=0.
    for sweep in 1:sweeps, y in 1:Ly, x in 1:Lx, (ia,ib,dx,dy) in ((3,1,1,0),(4,2,0,1))
        nx,ny=mod1(x+dx,Lx),mod1(y+dy,Ly)
        M,N,err,c=_bond_balance(dense[x,y],dense[nx,ny],ia,ib;floor)
        residual=max(residual,err);maxcondition=max(maxcondition,c)
        dense[x,y]=_gauge_axis(dense[x,y],M,ia)
        dense[nx,ny]=_gauge_axis(dense[nx,ny],N,ib)
    end
    result=ntuple(Lx) do x
        ntuple(Ly) do y
            a=TensorMap(dense[x,y],codomain(A[x][y]),domain(A[x][y]))
            a/norm(a)
        end
    end
    result,(;sweeps,floor,identity_residual=residual,max_gauge_condition=maxcondition)
end
