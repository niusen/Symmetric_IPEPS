# for (a,b) in blocks(W_set[1,1])
#     println(a)
#     println(b)
# end

function create_vaccum_mps(L)
    #create vaccum

    V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(1,1/2)=>1,(2,0)=>1);#element order after converting to dense: <0,0>, <up,down>, <up,0>, <0,down>, 
    V_R=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1);



    M=zeros(1,1,4)*im;M[1,1,1]=1;
    M=TensorMap(M, V_R ← V_R ⊗ V);
    mps_set=Array{Any}(undef, L)
    for cc=1:L
        mps_set[cc]=M;
    end
    return mps_set
end

function create_mpo(W)
    size1=size(W)[1]
    size2=size(W)[2]

    V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(1,1/2)=>1,(2,0)=>1);#element order after converting to dense: <0,0>, <up,down>, <up,0>, <0,down>, 

    W_L=zeros(1,4)*im;W_L[1,2]=1;
    V_L=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2,0)=>1);
    W_L=TensorMap(W_L, V_L ←  V);

    W_R=zeros(4,1)*im;W_R[1,1]=1;
    V_R=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1);
    W_R=TensorMap(W_R, V ← V_R);

    
    W_set=Array{Any}(undef, size2,size1)
    #element order after reshape: a[1,1]=<0,0>, a[2,1]=<up,0>, a[1,2]=<0,down>, a[2,2]=<up,down>
    Pm=zeros(4,4)*im;Pm[1,1]=1;Pm[2,4]=1;Pm[3,2]=1;Pm[4,3]=1;Pm=reshape(Pm,4,2,2);
    W1=zeros(2,2,2,2)*im;
    W_total=1;
    for bb=1:size2
        for cc=1:size1
            W1=zeros(2,2,2,2)*im;
            W1[1,:,1,:]=Matrix(I,2,2);
            W1[2,:,1,:]=[0 0;1 0]'*W[cc,bb];
            W1[2,:,2,:]=[1 0;0 -1];

            W2=zeros(2,2,2,2)*im;
            W2[1,:,1,:]=Matrix(I,2,2);
            W2[2,:,2,:]=[1 0;0 -1];
            
            @tensor W_total[:]:=W1[-1,1,4,-7]*W2[4,3,-5,-8]*W2[-2,-3,2,1]*W1[2,-4,-6,3];
            @tensor W_total[:]:=W_total[1,2,3,4,5,6,7,8]*Pm[-1,1,2]*Pm[-2,3,4]*Pm[-3,5,6]*Pm[-4,7,8];

            T=TensorMap(W_total,V ⊗ V ← V ⊗ V);
            if cc==1
                @tensor T[:]:=W_L[-1,1]*T[1,-2,-3,-4];
            elseif cc==(size1)
                @tensor T[:]:=T[-1,-2,1,-4]*W_R[1,-3];
            end
            W_set[bb,cc]=T;

        end
    end
    return W_set

end


function mpo_mps(mpo_set,mps_set)
    mpo_set=deepcopy(mpo_set);
    mps_set=deepcopy(mps_set);
    L=length(mps_set);
    @assert length(mpo_set)==length(mps_set)
    for cc=1:L
        @tensor mps[:]:=mpo_set[cc][-1,1,-3,-5]*mps_set[cc][-2,-4,1];
        mps_set[cc]=mps;
    end
    #fuse legs
    UL=unitary(fuse(space(mps_set[1],1)⊗space(mps_set[1],2)),space(mps_set[1],1)⊗space(mps_set[1],2));
    @tensor mps[:]:=mps_set[1][1,2,-3,-4,-5]*UL[-1,1,2];
    mps_set[1]=mps;

    for cc=1:L-1
        UL=unitary(fuse(space(mps_set[cc+1],1)⊗space(mps_set[cc+1],2)),space(mps_set[cc+1],1)⊗space(mps_set[cc+1],2));
        # println(UL)
        # println(UL')
        # println(UL*UL')
        @tensor mps[:]:=mps_set[cc][-1,1,2,-3]*UL'[1,2,-2];
        mps_set[cc]=mps;
        @tensor mps[:]:=mps_set[cc+1][1,2,-2,-3,-4]*UL[-1,1,2];
        mps_set[cc+1]=mps;
    end

    UR=unitary(fuse(space(mps_set[end],2)⊗space(mps_set[end],3)),space(mps_set[end],2)⊗space(mps_set[end],3));
    @tensor mps[:]:=mps_set[end][-1,1,2,-3]*UR[-2,1,2];
    mps_set[end]=mps;

    for cc=1:L
        mps_set[cc]=permute(mps_set[cc],(1,),(2,3,));
    end

    return mps_set
end

function overlap_1D(mps1,mps2)
    mps1=deepcopy(mps1);
    mps2=deepcopy(mps2);
    @tensor Left[:]:=mps1[1]'[-1,1,2]*mps2[1][2,-2,1];
    for cc=2:length(mps1)-1
        @tensor Left[:]:=Left[1,2]*mps1[cc]'[-1,3,1]*mps2[cc][2,-2,3];
    end
    @tensor Left[:]:=Left[1,2]*mps1[end]'[4,3,1]*mps2[end][2,4,3];

    return blocks(Left)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1]
end