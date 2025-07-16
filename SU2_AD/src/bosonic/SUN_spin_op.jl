
function SUN_spin(N::Int)
    @assert N>=2;
    if N==2
        Vp=Rep[SU{N}](1/2=>1);
        V_trivial=Rep[SU{N}](0=>1);
    elseif N>2
        Vp=Rep[SU{N}](string(N)=>1);
        V_trivial=Rep[SU{N}](string(1)=>1);
    end
    
    ###########################
    P_ij=zeros(1,N,N,N,N,1);#d1',d2',d1,d2
    for ca=1:N
        for cb=1:N
            P_ij[1,ca,cb,cb,ca,1]=1;
        end
    end
    P_ij=TensorMap(P_ij, V_trivial*Vp*Vp,Vp*Vp*V_trivial);

    P_ijk=zeros(1,N,N,N,N,N,N,1);#d1',d2',d3',d1,d2,d3
    for ca=1:N
        for cb=1:N
            for cc=1:N
                P_ijk[1,ca,cb,cc,cb,cc,ca,1]=1;
            end
        end
    end
    @assert norm(conj.(permutedims(P_ijk,(1,5,6,7,2,3,4,8)))-permutedims(P_ijk,(1,2,4,3,5,7,6,8)))<1e-10;
    P_ijk=TensorMap(P_ijk,V_trivial*Vp*Vp*Vp,Vp*Vp*Vp*V_trivial);
    Id_2site=unitary(V_trivial*Vp*Vp, V_trivial*Vp*Vp);
    Id_3site=unitary(V_trivial*Vp*Vp*Vp, V_trivial*Vp*Vp*Vp);
    Id_2site=permute(Id_2site,(1,2,3,),(5,6,4,));
    Id_3site=permute(Id_3site,(1,2,3,4,),(6,7,8,5,));
    ###############################    
    P_ij=permute(P_ij,(1,2,3,),(4,5,6,));
    P_ijk=permute(P_ijk,(1,2,3,4,),(5,6,7,8,));
    
    #P_ikj=permute(P_ijk,(1,2,4,3,),(5,7,6,8,));
    ######################################
    P_ij=P_ij-Id_2site/N;

    chirality_123=im*P_ijk-im*permute(P_ijk',(4,1,2,3,),(6,7,8,5,))
    ######################################

    SS=permute(P_ij,(1,2,4,),(3,5,6,));#V,s1',s1,s2',s2
    u0,s0,v0=tsvd(SS; trunc=truncerr(1e-12));
    @assert norm(u0*s0*v0-SS)<1e-14;
    Sa=permute(u0*s0,(1,4,),(2,3,));#virtual, d',d
    Sb=permute(v0,(1,4,),(2,3,));#virtual, d',d

    Id_SS=unitary(space(Sb,1)*space(Sb,4)',space(Sb,1)*space(Sb,4)');
    Id_SS=permute(Id_SS,(1,3,),(2,4,));#virtual1,virtual2, d',d
    @assert space(Id_SS,4)==space(Sb,4);



    u,s,v=tsvd(permute(P_ijk,(1,2,5,),(3,4,6,7,8)); trunc=truncerr(1e-12));
    P_123_a=permute(u,(1,4,),(2,3,));
    S2S3=s*v;
    u,s,v=tsvd(permute(S2S3,(1,2,4,),(3,5,6)); trunc=truncerr(1e-12));
    P_123_b=permute(u,(1,4,),(2,3,));
    P_123_c=permute(s*v,(1,4,),(2,3,));
    @tensor P123_[:]:=P_123_a[-1,2,-2,-5]*P_123_b[2,3,-3,-6]*P_123_c[3,-8,-4,-7];
    P123_=permute(P123_,(1,2,3,4,),(5,6,7,8,));
    @assert norm(P_ijk-P123_)/norm(P_ijk)<1e-12;


    Id_P123_ab=unitary(space(P_123_b,1)*space(P_123_b,4)',space(P_123_b,1)*space(P_123_b,4)');
    Id_P123_ab=permute(Id_P123_ab,(1,3,),(2,4,));
    @assert space(Id_P123_ab,4)==space(Sb,4);

    Id_P123_bc=unitary(space(P_123_c,1)*space(P_123_c,4)',space(P_123_c,1)*space(P_123_c,4)');
    Id_P123_bc=permute(Id_P123_bc,(1,3,),(2,4,));
    @assert space(Id_P123_bc,4)==space(Sb,4);


    u,s,v=tsvd(permute(P_ijk,(1,1+1,4+1,),(3+1,2+1,6+1,5+1,8)); trunc=truncerr(1e-12));
    P_132_a=permute(u,(1,4,),(2,3,));
    S2S3=s*v;
    u,s,v=tsvd(permute(S2S3,(1,2,4,),(3,5,6)); trunc=truncerr(1e-12));
    P_132_b=permute(u,(1,4,),(2,3,));
    P_132_c=permute(s*v,(1,4,),(2,3,));
    @tensor P132_[:]:=P_132_a[-1,2,-2,-5]*P_132_b[2,3,-3,-6]*P_132_c[3,-8,-4,-7];
    P132_=permute(P132_,(1,1+1,3+1,2+1,),(4+1,6+1,5+1,8,));
    @assert norm(P_ijk-P132_)/norm(P_ijk)<1e-12;

    @assert space(Id_P123_ab,1)==space(P_132_b,1);
    @assert space(Id_P123_bc,1)==space(P_132_c,1);
    ################################
    u,s,v=tsvd(permute(chirality_123,(1,2,5,),(3,4,6,7,8)); trunc=truncerr(1e-12));
    chirality_123_a=permute(u,(1,4,),(2,3,));
    S2S3=s*v;
    u,s,v=tsvd(permute(S2S3,(1,2,4,),(3,5,6)); trunc=truncerr(1e-12));
    chirality_123_b=permute(u,(1,4,),(2,3,));
    chirality_123_c=permute(s*v,(1,4,),(2,3,));
    @tensor chirality_123__[:]:=chirality_123_a[-1,2,-2,-5]*chirality_123_b[2,3,-3,-6]*chirality_123_c[3,-8,-4,-7];
    chirality_123__=permute(chirality_123__,(1,2,3,4,),(5,6,7,8,));
    @assert norm(chirality_123-chirality_123__)/norm(chirality_123)<1e-12;


    Id_chirality_ab=unitary(space(chirality_123_b,1)*space(chirality_123_b,4)',space(chirality_123_b,1)*space(chirality_123_b,4)');
    Id_chirality_ab=permute(Id_chirality_ab,(1,3,),(2,4,));
    @assert space(Id_chirality_ab,4)==space(Sb,4);

    Id_chirality_bc=unitary(space(chirality_123_c,1)*space(chirality_123_c,4)',space(chirality_123_c,1)*space(chirality_123_c,4)');
    Id_chirality_bc=permute(Id_chirality_bc,(1,3,),(2,4,));
    @assert space(Id_chirality_bc,4)==space(Sb,4);
    

    return Sa,Sb, Id_SS,  P_123_a,P_123_b,P_123_c,  P_132_a,P_132_b,P_132_c,   Id_P123_ab, Id_P123_bc,   chirality_123_a,chirality_123_b,chirality_123_c, Id_chirality_ab,Id_chirality_bc
end