
using LinearAlgebra:diag,I,diagm 

function initial_SU2_PESS(filenm,init_noise,is_complex)
    data=load(filenm);
    if (haskey(data, "T_set"))&&(haskey(data, "B_set"))
        Lx,Ly=size(data["T_set"]);
        Tset=data["T_set"];
        Bset=data["B_set"];
        psi=Matrix{Triangle_iPESS}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                psi[cx,cy]=Triangle_iPESS(Tset[cx,cy],Bset[cx,cy]);
            end
        end
    elseif (haskey(data, "psi"))
        psi=data["psi"];
    elseif (haskey(data, "x"))
        psi=data["x"];
    end
    psi=add_noise(psi,init_noise,is_complex);
    return psi

end

function add_noise(psi::Matrix,init_noise,is_complex)
    Lx,Ly=size(psi);
    for cx=1:Lx
        for cy=1:Ly
            PESS=psi[cx,cy];
            Bm=PESS.Bm;
            Tm=PESS.Tm;
            if is_complex
                Bm_noise=TensorMap(randn,codomain(Bm),domain(Bm))+TensorMap(randn,codomain(Bm),domain(Bm))*im;
                Tm_noise=TensorMap(randn,codomain(Tm),domain(Tm))+TensorMap(randn,codomain(Tm),domain(Tm))*im;
            else
                Bm_noise=TensorMap(randn,codomain(Bm),domain(Bm));
                Tm_noise=TensorMap(randn,codomain(Tm),domain(Tm));
            end
            Bm_noise=Bm_noise/norm(Bm_noise);
            Tm_noise=Tm_noise/norm(Tm_noise);
            Bm=Bm/norm(Bm);
            Tm=Tm/norm(Tm);
            Bm_=Bm+Bm_noise*init_noise;
            Tm_=Tm+Tm_noise*init_noise;

            psi[cx,cy]=Triangle_iPESS(Bm_,Tm_);
        end
    end
    return psi
end

function iPESS_to_iPEPS_tensor(Bm,Tm)
    #|LU><M|
    #|Md><|RD
    Tm=permute(Tm,(1,2,),(3,));
    Bm=permute(Bm,(1,),(2,3,4,));
    T=permute(Tm*Bm,(1,5,4,2,3,));#L,D,R,U,d,
    return T
end


function PESS_to_PEPS_matrix(A_set::Matrix{Triangle_iPESS})    
    Lx,Ly=size(A_set)
    A_cell=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=iPESS_to_iPEPS_tensor(A_set[cx,cy].Bm,A_set[cx,cy].Tm);
            A_cell[cx,cy]=A;
        end
    end
    return A_cell
end

function PESS_to_PEPS_matrix(A_set::Matrix{Triangle_iPESS},A_cell::Matrix)    
    Lx,Ly=size(A_set)
    A_cell=deepcopy(A_cell);
    #provide initial PEPS matrix for AD 
    for cx=1:Lx
        for cy=1:Ly
            A=iPESS_to_iPEPS_tensor(A_set[cx,cy].Bm,A_set[cx,cy].Tm);
            A_cell=matrix_update(A_cell,cx,cy,A);
        end
    end
    return A_cell
end

function B_T_sets_to_PESS(Bset::Matrix,Tset::Matrix)    
    Lx,Ly=size(B_set)
    A_cell=Matrix{Triangle_iPESS}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A_cell[cx,cy]=Triangle_iPESS(Tset[cx,cy],Bset[cx,cy]);
        end
    end
    return A_cell
end



function convert_PESS_to_PEPS(Bset,Tset)
    global Lx,Ly
    
    A_cell_iPEPS=initial_tuple_cell(Lx,Ly);
    for ca=1:Lx
        for cb=1:Ly
            A_A=iPESS_to_iPEPS(Triangle_iPESS(Tset[ca,cb],Bset[ca,cb]));
            A_cell_iPEPS=fill_tuple(A_cell_iPEPS, A_A.T, ca,cb);
        end
    end
    return A_cell_iPEPS
end

function initial_iPESS_uniform(Lx,Ly,V,Vp)
    Bset=Matrix{Any}(undef,Lx,Ly);
    Tset=Matrix{Any}(undef,Lx,Ly);
    lambdaset1=Matrix{Any}(undef,Lx,Ly);
    lambdaset2=Matrix{Any}(undef,Lx,Ly);
    lambdaset3=Matrix{Any}(undef,Lx,Ly);
    BA=permute(TensorMap(randn,V'*Vp,V*V),(1,),(2,3,4,));
    TA=TensorMap(randn,V*V,V');
    for ca=1:Lx
        for cb=1:Ly
            Tset[ca,cb]=BA;
            Bset[ca,cb]=TA;
            t_A=TA;
            λ_A_1=unitary(space(t_A,1)',space(t_A,1)');
            λ_A_2=unitary(space(t_A,2)',space(t_A,2)');
            λ_A_3=unitary(space(t_A,3)',space(t_A,3)');
            lambdaset1[ca,cb]=λ_A_1;
            lambdaset2[ca,cb]=λ_A_2;
            lambdaset3[ca,cb]=λ_A_3;
        end
    end
    return Bset, Tset, lambdaset1, lambdaset2, lambdaset3
end



function initial_iPESS(Lx,Ly,V,Vp)
    Bset=Matrix{Any}(undef,Lx,Ly);
    Tset=Matrix{Any}(undef,Lx,Ly);
    lambdaset1=Matrix{Any}(undef,Lx,Ly);
    lambdaset2=Matrix{Any}(undef,Lx,Ly);
    lambdaset3=Matrix{Any}(undef,Lx,Ly);

    for ca=1:Lx
        for cb=1:Ly
            BA=permute(TensorMap(randn,V'*Vp,V*V),(1,),(2,3,4,))+permute(TensorMap(randn,V'*Vp,V*V),(1,),(2,3,4,))*im;
            TA=TensorMap(randn,V*V,V')+TensorMap(randn,V*V,V')*im;
            Tset[ca,cb]=BA;
            Bset[ca,cb]=TA;
            t_A=TA;
            λ_A_1=unitary(space(t_A,1)',space(t_A,1)');
            λ_A_2=unitary(space(t_A,2)',space(t_A,2)');
            λ_A_3=unitary(space(t_A,3)',space(t_A,3)');
            lambdaset1[ca,cb]=λ_A_1;
            lambdaset2[ca,cb]=λ_A_2;
            lambdaset3[ca,cb]=λ_A_3;
        end
    end
    return Bset, Tset, lambdaset1, lambdaset2, lambdaset3
end


function get_PESS_from_iPESS(T_phy_set,T_virt_set,Lx,Ly)
    @assert size(T_phy_set)==(2,2);

    #unit-cell of iPESS: 2x2
    iPESS_cell=[2,2];
    psi=Matrix{Triangle_iPESS}(undef,Lx,Ly);#PBC-PBC
    for cx=1:Lx
        for cy=1:Ly
            psi[cx,cy]=Triangle_iPESS(T_phy_set[mod1(cx,iPESS_cell[1]),mod1(cy,iPESS_cell[2])],T_virt_set[mod1(cx,iPESS_cell[1]),mod1(cy,iPESS_cell[2])]);
        end
    end

    #left boundary
    cx=1;
    for cy=1:Ly
        VL=Rep[SU₂](0=>1);
        V=space(psi[1,cy].Tm,1);
        iso=create_isometry(V,VL);
        T=psi[cx,cy].Tm;
        @tensor T[:]:=T[1,-2,-3]*iso'[-1,1];
        T=permute(T,(1,2,),(3,));
        psi[cx,cy].Tm=T;
    end

    #right boundary
    cx=Lx;
    for cy=1:Ly
        VL=Rep[SU₂](0=>1)';
        V=space(psi[cx,cy].Bm,3);
        iso=create_isometry(V,VL);
        T=psi[cx,cy].Bm;
        @tensor T[:]:=T[-1,-2,1,-4]*iso'[-3,1];
        T=permute(T,(1,),(2,3,4,));
        psi[cx,cy].Bm=T;
    end


    #bot boundary
    cy=1;
    for cx=1:Lx
        VL=Rep[SU₂](0=>1)';
        V=space(psi[cx,cy].Bm,4);
        iso=create_isometry(V,VL);
        T=psi[cx,cy].Bm;
        @tensor T[:]:=T[-1,-2,-3,1]*iso'[-4,1];
        T=permute(T,(1,),(2,3,4,));
        psi[cx,cy].Bm=T;
    end

    #top boundary
    cy=Ly;
    for cx=1:Lx
        VL=Rep[SU₂](0=>1);
        V=space(psi[cx,cy].Tm,2);
        iso=create_isometry(V,VL);
        T=psi[cx,cy].Tm;
        @tensor T[:]:=T[-1,1,-3]*iso'[-2,1];
        T=permute(T,(1,2,),(3,));
        psi[cx,cy].Tm=T;
    end

    return psi
end


function PESS_to_B_T_sets(psi)
    B_set=Matrix{TensorMap}(undef,Lx,Ly);
    T_set=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            B_set[cx,cy]=psi[cx,cy].Tm;
            T_set[cx,cy]=psi[cx,cy].Bm;
        end
    end
    return B_set,T_set
end

function get_trivial_lambda(Bset)
    lambdaset1=Matrix{Any}(undef,Lx,Ly);
    lambdaset2=Matrix{Any}(undef,Lx,Ly);
    lambdaset3=Matrix{Any}(undef,Lx,Ly);
    for ca=2:Lx
        for cb=1:Ly-1
            t_A=Bset[ca,cb];
            λ_A_1=unitary(space(t_A,1)',space(t_A,1)');
            λ_A_2=unitary(space(t_A,2)',space(t_A,2)');
            λ_A_3=unitary(space(t_A,3)',space(t_A,3)');
            lambdaset1[ca,cb]=λ_A_1;
            lambdaset2[ca,cb]=λ_A_2;
            lambdaset3[ca,cb]=λ_A_3;
        end
    end
    return lambdaset1,lambdaset2,lambdaset3
end