function group_numbers(E_set)
    Es=copy(E_set[1]);
    for cc=2:length(E_set)
        Es=vcat(Es,E_set[cc]);
    end
    return Es
end


function TransfOp_decom(A,O,total_space,total_sec,pow,Dtrun_init,Dtrun_max,trun_tol,method)
    # euL_set,evL_set,euR_set,evR_set,SPIN_eig_set=FLR_eig(Ag,OO,Dtrun,space_AOA,AOA_sec);

    # eul_set,evl_set,eur_set,evr_set,spin_eig_set=GLR_eig(Ag,Dtrun,space_AA,AA_sec);

    # S_set,U_set,Vh_set,SPIN_svd_set=FLR_svd(Ag,OO,pow,Dtrun,space_AOA,AOA_sec);

    # s_set,u_set,vh_set,spin_svd_set=GLR_svd(Ag,pow,Dtrun,space_AA,AA_sec);


    # euR_set,evL_set,evR_set,SPIN_eig_set=truncate_sectors(Dtrun,euR_set,evL_set,evR_set,SPIN_eig_set);

    # eur_set,evl_set,evr_set,spin_eig_set=truncate_sectors(Dtrun,eur_set,evl_set,evr_set,spin_eig_set);

    # S_set,U_set,Vh_set,SPIN_svd_set=truncate_sectors(Dtrun,S_set,U_set,Vh_set,SPIN_svd_set);

    # s_set,u_set,vh_set,spin_svd_set=truncate_sectors(Dtrun,s_set,u_set,vh_set,spin_svd_set);


    Dstep=50;
    Dtrun=Dtrun_init;
    if method=="eigenvalue_FLR"
        evL_set=Vector{Any}(undef, length(total_sec));
        euR_set=Vector{Any}(undef, length(total_sec));
        evR_set=Vector{Any}(undef, length(total_sec));
        SPIN_eig_set=Vector{Any}(undef, length(total_sec));
        for cs=1:Int(round((Dtrun_max-Dtrun_init)/Dstep))+1
            _,evL_set,euR_set,evR_set,SPIN_eig_set=FLR_eig(A,O,Dtrun,total_space,total_sec);
            euR_set,evL_set,evR_set,SPIN_eig_set=truncate_sectors(Dtrun,euR_set,evL_set,evR_set,SPIN_eig_set);
            eu=abs.(group_numbers(euR_set));
            eu_normed=eu/maximum(eu); 
            eu_normed=eu_normed.^pow;
            println("FLR_eig, DTrun="*string(Dtrun)*", SU2 reduced to "*string(length(group_numbers(euR_set)))*", minimal eigenvalue: "*string(minimum(eu_normed)));flush(stdout);
            if minimum(eu_normed)<trun_tol
                break;
            else
                if Dtrun>dim(total_space)
                    println("Dtrun exceeds matrix size");flush(stdout);
                    break;
                end
                Dtrun=Dtrun+Dstep;
            end
        end
        
        eu=abs.(group_numbers(euR_set));
        
        for cc=1:length(euR_set)
            euR=euR_set[cc];
            evL=evL_set[cc];
            evR=evR_set[cc];
            SPIN_eig=SPIN_eig_set[cc];
            for cs=length(euR):-1:1
                if ((abs(euR[cs]))/maximum(eu))^pow<trun_tol
                    deleteat!(euR, cs);
                    deleteat!(evR, cs);
                    deleteat!(evL, cs);
                    deleteat!(SPIN_eig, cs);
                end
            end
            euR_set[cc]=euR;
            evL_set[cc]=evL;
            evR_set[cc]=evR;
            SPIN_eig_set[cc]=SPIN_eig;
        end
        
        return euR_set,evL_set,evR_set,SPIN_eig_set

    elseif method=="eigenvalue_GLR"
        evl_set=Vector{Any}(undef, length(total_sec));
        eur_set=Vector{Any}(undef, length(total_sec));
        evr_set=Vector{Any}(undef, length(total_sec));
        spin_eig_set=Vector{Any}(undef, length(total_sec));
        for cs=1:Int(round((Dtrun_max-Dtrun_init)/Dstep))+1
            _,evl_set,eur_set,evr_set,spin_eig_set=GLR_eig(A,Dtrun,total_space,total_sec);
            eur_set,evl_set,evr_set,spin_eig_set=truncate_sectors(Dtrun,eur_set,evl_set,evr_set,spin_eig_set);
            eu=abs.(group_numbers(eur_set));
            eu_normed=eu/maximum(eu); 
            eu_normed=eu_normed.^pow;
            println("GLR_eig, Dtrun="*string(Dtrun)*", SU2 reduced to "*string(length(group_numbers(eur_set)))*", minimal eigenvalue: "*string(minimum(eu_normed)));flush(stdout);
            if minimum(eu_normed)<trun_tol
                break;
            else
                if Dtrun>dim(total_space)
                    println("Dtrun exceeds matrix size");flush(stdout);
                    break;
                end
                Dtrun=Dtrun+Dstep;
            end
        end

        eu=abs.(group_numbers(eur_set));
        
        for cc=1:length(eur_set)
            eur=eur_set[cc];
            evl=evl_set[cc];
            evr=evr_set[cc];
            spin_eig=spin_eig_set[cc];
            for cs=length(eur):-1:1
                if ((abs(eur[cs]))/maximum(eu))^pow<trun_tol
                    deleteat!(eur, cs);
                    deleteat!(evr, cs);
                    deleteat!(evl, cs);
                    deleteat!(spin_eig, cs);
                end
            end
            eur_set[cc]=eur;
            evl_set[cc]=evl;
            evr_set[cc]=evr;
            spin_eig_set[cc]=spin_eig;
        end
        
        return eur_set,evl_set,evr_set,spin_eig_set

    elseif method=="svd_FLR"
        S_set=Vector{Any}(undef, length(total_sec));
        U_set=Vector{Any}(undef, length(total_sec));
        Vh_set=Vector{Any}(undef, length(total_sec));
        SPIN_svd_set=Vector{Any}(undef, length(total_sec));
        for cs=1:Int(round((Dtrun_max-Dtrun_init)/Dstep))+1
            S_set,U_set,Vh_set,SPIN_svd_set=FLR_svd(A,O,pow,Dtrun,total_space,total_sec);
            S_set,U_set,Vh_set,SPIN_svd_set=truncate_sectors(Dtrun,S_set,U_set,Vh_set,SPIN_svd_set);  
            eu=abs.(group_numbers(S_set));
            eu_normed=eu/maximum(eu); 
            println("FLR_svd, DTrun="*string(Dtrun)*", SU2 reduced to "*string(length(group_numbers(S_set)))*", minimal eigenvalue: "*string(minimum(eu_normed)));flush(stdout);
            if minimum(eu_normed)<trun_tol
                break;
            else
                if Dtrun>dim(total_space)
                    println("Dtrun exceeds matrix size");flush(stdout);
                    break;
                end
                Dtrun=Dtrun+Dstep;
            end
        end
        
        eu=abs.(group_numbers(S_set));
        
        for cc=1:length(S_set)
            S=S_set[cc];
            U=U_set[cc];
            Vh=Vh_set[cc];
            SPIN_svd=SPIN_svd_set[cc];
            for cs=length(S):-1:1
                if ((abs(S[cs]))/maximum(eu))<trun_tol
                    deleteat!(S, cs);
                    deleteat!(U, cs);
                    deleteat!(Vh, cs);
                    deleteat!(SPIN_svd, cs);
                end
            end
            S_set[cc]=S;
            U_set[cc]=U;
            Vh_set[cc]=Vh;
            SPIN_svd_set[cc]=SPIN_svd;
        end
        
        return S_set,U_set,Vh_set,SPIN_svd_set
    elseif method=="svd_GLR"
        s_set=Vector{Any}(undef, length(total_sec));
        u_set=Vector{Any}(undef, length(total_sec));
        vh_set=Vector{Any}(undef, length(total_sec));
        spin_svd_set=Vector{Any}(undef, length(total_sec));
        for cs=1:Int(round((Dtrun_max-Dtrun_init)/Dstep))+1
            s_set,u_set,vh_set,spin_svd_set=GLR_svd(A,pow,Dtrun,total_space,total_sec);
            s_set,u_set,vh_set,spin_svd_set=truncate_sectors(Dtrun,s_set,u_set,vh_set,spin_svd_set);
            eu=abs.(group_numbers(s_set));
            eu_normed=eu/maximum(eu); 
            println("GLR_svd, Dtrun="*string(Dtrun)*", SU2 reduced to "*string(length(group_numbers(s_set)))*", minimal eigenvalue: "*string(minimum(eu_normed)));flush(stdout);
            if minimum(eu_normed)<trun_tol
                break;
            else
                if Dtrun>dim(total_space)
                    println("Dtrun exceeds matrix size");flush(stdout);
                    break;
                end
                Dtrun=Dtrun+Dstep;
            end
        end
    
        eu=abs.(group_numbers(s_set));
        
        for cc=1:length(s_set)
            s=s_set[cc];
            u=u_set[cc];
            vh=vh_set[cc];
            spin_svd=spin_svd_set[cc];
            for cs=length(s):-1:1
                if ((abs(s[cs]))/maximum(eu))<trun_tol
                    deleteat!(s, cs);
                    deleteat!(u, cs);
                    deleteat!(vh, cs);
                    deleteat!(spin_svd, cs);
                end
            end
            s_set[cc]=s;
            u_set[cc]=u;
            vh_set[cc]=vh;
            spin_svd_set[cc]=spin_svd;
        end
        return s_set,u_set,vh_set,spin_svd_set
    end
end
    




function FLR_eig(A,O,n,full_space,sector_set)
    SPIN_set=Vector{Any}(undef, length(sector_set));
    euL_set=Vector{Any}(undef, length(sector_set));
    evL_set=Vector{Any}(undef, length(sector_set));
    euR_set=Vector{Any}(undef, length(sector_set));
    evR_set=Vector{Any}(undef, length(sector_set));
    #println("FLR_eig: ")
    for cc=1:length(sector_set)
        sec=sector_set[cc];
        spin=(dim(sec)-1)/2;
        n_eff=Int(round(n/dim(sec)))+1
        #println("spin "*string(spin))

        FLR_eig_L(x)=HV_FL(x,A,O);
        vl_init=permute(TensorMap(randn, SU₂Space(spin=>1),space(A',1)*space(O,1)*space(A,1)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        try
            euL,evL,info=eigsolve(FLR_eig_L, vl_init, minimum([n_eff,dim(full_space,sec)]),:LM, Arnoldi(krylovdim=minimum([n_eff,dim(full_space,sec)])*4));
            @assert info.converged >= minimum([n_eff,dim(full_space,sec)])
        catch e
            println("Number of eigenvalues obtained are not enough, use smaller tol")
            euL,evL,info=eigsolve(FLR_eig_L, vl_init, minimum([n_eff,dim(full_space,sec)]),:LM, Arnoldi(krylovdim=minimum([n_eff,dim(full_space,sec)])*8, tol=1e-14));
            if minimum(abs.(euL))/maximum(abs.(euL))<1e-7
                println("minimal singular value in this sector is quite small, skip checking number of converged values");flush(stdout);
            else
                if  info.converged >= minimum([n_eff,dim(full_space,sec)])
                    @warn "number of values converged is not enough"
                end
            end
        end
        @assert abs.(euL)==sort(abs.(euL), rev=true)
        euL=euL[1:minimum([length(euL),n_eff,dim(full_space,sec)])];
        evL=evL[1:minimum([length(evL),n_eff,dim(full_space,sec)])];
        euL_set[cc]=euL;
        evLL=evL;
        evL=Vector{Any}(undef, length(evLL));
        for cs=1:length(evL)
            evL[cs]=permute(evLL[cs],(1,),(2,3,4,))
        end
        evL_set[cc]=evL;
        SPIN_set[cc]=ones(length(euL))*(dim(sec)-1)/2;
        
        FLR_eig_R(x)=HV_FR(x,A,O);
        vr_init=permute(TensorMap(randn, space(A',2)'*space(O,3)'*space(A,2)',SU₂Space(spin=>1)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        try
            euR,evR,info=eigsolve(FLR_eig_R, vr_init, minimum([n_eff,dim(full_space,sec)]),:LM, Arnoldi(krylovdim=minimum([n_eff,dim(full_space,sec)])*4));
            @assert info.converged >= minimum([n_eff,dim(full_space,sec)])
        catch e
            println("Number of eigenvalues obtained are not enough, use smaller tol")
            euR,evR,info=eigsolve(FLR_eig_R, vr_init, minimum([n_eff,dim(full_space,sec)]),:LM, Arnoldi(krylovdim=minimum([n_eff,dim(full_space,sec)])*4, tol=1e-14));
            if minimum(abs.(euR))/maximum(abs.(euR))<1e-7
                println("minimal singular value in this sector is quite small, skip checking number of converged values");flush(stdout);
            else
                if  info.converged >= minimum([n_eff,dim(full_space,sec)])
                    @warn "number of values converged is not enough"
                end
            end
        end
        @assert abs.(euR)==sort(abs.(euR), rev=true)
        euR=euR[1:minimum([length(euR),n_eff,dim(full_space,sec)])];
        evR=evR[1:minimum([length(evR),n_eff,dim(full_space,sec)])];
        euR_set[cc]=euR;
        evRR=evR;
        evR=Vector{Any}(undef, length(evRR));
        for cs=1:length(evR)
            evR[cs]=permute(evRR[cs],(1,2,3,),(4,))
        end
        evR_set[cc]=evR;

        #@assert (norm(abs.(euL)-abs.(euR))/norm(euL))<1e-8
    end
    return euL_set,evL_set,euR_set,evR_set,SPIN_set     
end
function HV_FL(vl,A,mpo)
    @tensor vl[:]:=vl[-1,1,3,5]*A'[1,-2,2]*mpo[3,4,-3,2]*A[5,-4,4];
    return vl
end
function HV_FR(vr,A,mpo)
    @tensor vr[:]:=A'[-1,1,2]*mpo[-2,4,3,2]*A[-3,5,4]*vr[1,3,5,-4];
    return vr
end
 
        
function GLR_eig(A,n,full_space,sector_set)
    spin_set=Vector{Any}(undef, length(sector_set));
    eul_set=Vector{Any}(undef, length(sector_set));
    evl_set=Vector{Any}(undef, length(sector_set));
    eur_set=Vector{Any}(undef, length(sector_set));
    evr_set=Vector{Any}(undef, length(sector_set));
    #println("GLR_eig: ")
    for cc=1:length(sector_set)
        sec=sector_set[cc];
        spin=(dim(sec)-1)/2;
        n_eff=Int(round(n/dim(sec)))+1
        #println("spin "*string(spin))
        
        GLR_eig_L(x)=HV_L_tensor(x,A,[]);
        vl_init=permute(TensorMap(randn, SU₂Space(spin=>1),space(A',1)*space(A,1)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        try
            eul,evl,info=eigsolve(GLR_eig_L, vl_init, minimum([n_eff,dim(full_space,sec)]),:LM,Arnoldi(krylovdim=minimum([n_eff,dim(full_space,sec)])*4));
            @assert info.converged >= minimum([n_eff,dim(full_space,sec)])
        catch e
            println("Number of eigenvalues obtained are not enough, use smaller tol")
            eul,evl,info=eigsolve(GLR_eig_L, vl_init, minimum([n_eff,dim(full_space,sec)]),:LM,Arnoldi(krylovdim=minimum([n_eff,dim(full_space,sec)])*4, tol=1e-14));
            if minimum(abs.(eul))/maximum(abs.(eul))<1e-7
                println("minimal singular value in this sector is quite small, skip checking number of converged values");flush(stdout);
            else
                if  info.converged >= minimum([n_eff,dim(full_space,sec)])
                    @warn "number of values converged is not enough"
                end
            end
        end
        @assert abs.(eul)==sort(abs.(eul), rev=true)
        
        eul=eul[1:minimum([length(eul),n_eff,dim(full_space,sec)])];
        evl=evl[1:minimum([length(evl),n_eff,dim(full_space,sec)])];
        eul_set[cc]=eul;
        evll=evl;
        evl=Vector{Any}(undef, length(evll));
        for cs=1:length(evl)
            evl[cs]=permute(evll[cs],(1,),(2,3,))
        end
        evl_set[cc]=evl;
        spin_set[cc]=ones(length(eul))*(dim(sec)-1)/2;

        GLR_eig_R(x)=HV_R_tensor(x,A,[]);
        vr_init=permute(TensorMap(randn, space(A',2)'*space(A,2)',SU₂Space(spin=>1)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        try
            eur,evr,info=eigsolve(GLR_eig_R, vr_init, minimum([n_eff,dim(full_space,sec)]),:LM,Arnoldi(krylovdim=minimum([n_eff,dim(full_space,sec)])*4));
            @assert info.converged >= minimum([n_eff,dim(full_space,sec)])
        catch e
            println("Number of eigenvalues obtained are not enough, use smaller tol")
            eur,evr,info=eigsolve(GLR_eig_R, vr_init, minimum([n_eff,dim(full_space,sec)]),:LM,Arnoldi(krylovdim=minimum([n_eff,dim(full_space,sec)])*8, tol=1e-14));
            if minimum(abs.(eur))/maximum(abs.(eur))<1e-7
                println("minimal singular value in this sector is quite small, skip checking number of converged values");flush(stdout);
            else
                if  info.converged >= minimum([n_eff,dim(full_space,sec)])
                    @warn "number of values converged is not enough"
                end
            end
        end
        @assert abs.(eur)==sort(abs.(eur), rev=true)
        eur=eur[1:minimum([length(eur),n_eff,dim(full_space,sec)])];
        evr=evr[1:minimum([length(evl),n_eff,dim(full_space,sec)])];
        eur_set[cc]=eur;
        evrr=evr;
        evr=Vector{Any}(undef, length(evrr));
        for cs=1:length(evr)
            evr[cs]=permute(evrr[cs],(1,2,),(3,))
        end
        evr_set[cc]=evr;

        #@assert (norm(abs.(eul)-abs.(eur))/norm(eul))<1e-8

    end
    return eul_set,evl_set,eur_set,evr_set,spin_set   
end
    
    

function FLR_svd(A,O,pow,n,full_space,sector_set)
    SPIN_set=Vector{Any}(undef, length(sector_set));
    S_set=Vector{Any}(undef, length(sector_set));
    U_set=Vector{Any}(undef, length(sector_set));
    Vh_set=Vector{Any}(undef, length(sector_set));
    #println("FLR_svd: ");flush(stdout);
    for cc=1:length(sector_set)
        sec=sector_set[cc];
        spin=(dim(sec)-1)/2;
        n_eff=Int(round(n/dim(sec)))+1
        #println("spin "*string(spin));flush(stdout);

        FLR_svd_R(x)=HV_FR_pow(x,A,O,pow);
        FLR_svd_R_conj(x)=HV_FR_conj_pow(x,A,O,pow);
        vl_init=permute(TensorMap(randn, SU₂Space(spin=>1),space(A',1)*space(O,1)*space(A,1)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        vr_init=permute(TensorMap(randn, space(A',2)'*space(O,3)'*space(A,2)',SU₂Space(spin=>1)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        try
            S,U,V,info=svdsolve((FLR_svd_R,FLR_svd_R_conj), vr_init, minimum([n_eff,dim(full_space,sec)]),:LR, krylovdim=minimum([n_eff,dim(full_space,sec)])*4);
            if minimum(S)/maximum(S)<1e-7
                println("minimal singular value in this sector is quite small, skip checking number of converged values");flush(stdout);
            else
                @assert info.converged >= minimum([n_eff,dim(full_space,sec)])
            end
        catch e
            println("Number of singular values obtained are not enough, use smaller tol")
            S,U,V,info=svdsolve((FLR_svd_R,FLR_svd_R_conj), vr_init, minimum([n_eff,dim(full_space,sec)]),:LR, krylovdim=minimum([n_eff,dim(full_space,sec)])*8, tol=(1e-14));
            if minimum(S)/maximum(S)<1e-7
                println("minimal singular value in this sector is quite small, skip checking number of converged values");flush(stdout);
            else
                if  info.converged >= minimum([n_eff,dim(full_space,sec)])
                    @warn "number of values converged is not enough"
                end
            end
        end
        @assert abs.(S)==sort(abs.(S), rev=true)
        S=S[1:minimum([length(S),n_eff,dim(full_space,sec)])];
        U=U[1:minimum([length(U),n_eff,dim(full_space,sec)])];
        V=V[1:minimum([length(V),n_eff,dim(full_space,sec)])];

        UU=U;
        U=Vector{Any}(undef, length(UU));
        for cs=1:length(U)
            U[cs]=permute(UU[cs],(1,2,3,),(4,))
        end

        Vh=Vector{Any}(undef, length(V));
        for cs=1:length(Vh)
            vvr=V[cs];
            Vh[cs]=permute(vvr',(4,),(1,2,3,));
        end
        S_set[cc]=S;
        U_set[cc]=U;
        Vh_set[cc]=Vh;
        SPIN_set[cc]=ones(length(S))*(dim(sec)-1)/2;
        
    end
    return S_set,U_set,Vh_set,SPIN_set     
end
function HV_FR_pow(vr,A,mpo,pow)
    for cc=1:pow
        vr=HV_FR(vr,A,mpo);
    end
    return vr
end
function HV_FR_conj(vr,A,mpo)
    @tensor vr[:]:=A[1,-1,2]*mpo'[3,4,-2,2]*A'[5,-3,4]*vr[1,3,5,-4];
    return vr
end
function HV_FR_conj_pow(vr,A,mpo,pow)
    for cc=1:pow
        vr=HV_FR_conj(vr,A,mpo);
    end
    return vr
end
    






function GLR_svd(A,pow,n,full_space,sector_set)
    spin_set=Vector{Any}(undef, length(sector_set));
    s_set=Vector{Any}(undef, length(sector_set));
    u_set=Vector{Any}(undef, length(sector_set));
    vh_set=Vector{Any}(undef, length(sector_set));
    A_noise=TensorMap(randn, codomain(A),domain(A));
    A_noise=A_noise/norm(A_noise)*norm(A)*1e-9;
    #println("GLR_svd: ")
    for cc=1:length(sector_set)
        sec=sector_set[cc];
        spin=(dim(sec)-1)/2;
        n_eff=Int(round(n/dim(sec)))+1
        #println("spin "*string(spin))


        vr_init=permute(TensorMap(randn, space(A',2)'*space(A,2)',SU₂Space(spin=>1)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        try
            GLR_svd_R(x)=HV_R_pow(x,A,pow,A_noise*0);
            GLR_svd_R_conj(x)=HV_R_conj_pow(x,A,pow,A_noise*0);
            s,u,v,info=svdsolve((GLR_svd_R,GLR_svd_R_conj), vr_init, minimum([n_eff,dim(full_space,sec)]),:LR, krylovdim=minimum([n_eff,dim(full_space,sec)])*4);
            @assert info.converged >= minimum([n_eff,dim(full_space,sec)])    
        catch e
            GLR_svd_R(x)=HV_R_pow(x,A,pow,A_noise);
            GLR_svd_R_conj(x)=HV_R_conj_pow(x,A,pow,A_noise);
            println("Number of singular values obtained are not enough, add noise, increase Krylov subspace and use smaller tol")
            s,u,v,info=svdsolve((GLR_svd_R,GLR_svd_R_conj), vr_init, minimum([n_eff,dim(full_space,sec)]),:LR, krylovdim=minimum([n_eff,dim(full_space,sec)])*8, tol=1e-14);
            if minimum(s)/maximum(s)<1e-7
                println("minimal singular value in this sector is quite small, skip checking number of converged values");flush(stdout);
            else
                if  info.converged >= minimum([n_eff,dim(full_space,sec)])
                    @warn "number of values converged is not enough"
                end
            end
        end
        @assert abs.(s)==sort(abs.(s), rev=true)
    
        s=s[1:minimum([length(s),n_eff,dim(full_space,sec)])];
        u=u[1:minimum([length(u),n_eff,dim(full_space,sec)])];
        v=v[1:minimum([length(v),n_eff,dim(full_space,sec)])];

        uu=u;
        u=Vector{Any}(undef, length(u));
        for cs=1:length(u)
            u[cs]=permute(uu[cs],(1,2,),(3,))
        end

        vh=Vector{Any}(undef, length(v));
        for cs=1:length(vh)
            vvr=v[cs];
            vh[cs]=permute(vvr',(3,),(1,2,));
        end
        s_set[cc]=s;
        u_set[cc]=u;
        vh_set[cc]=vh;
        spin_set[cc]=ones(length(s))*(dim(sec)-1)/2;
        
    end
    return s_set,u_set,vh_set,spin_set     
end
function HV_R_pow(vr,A,pow,A_noise)
    A_=A+A_noise;
    for cc=1:pow
        @tensor vr[:]:=A_'[-1,1,2]*A[-2,3,2]*vr[1,3,-3];
    end
    return vr
end
function HV_R_conj_pow(vr,A,pow,A_noise)
    A_=A+A_noise;
    for cc=1:pow
        @tensor vr[:]:=A_[1,-1,2]*A'[3,-2,2]*vr[1,3,-3];
    end
    return vr
end











function truncate_sectors(n,eu_set,ev_set,ev_set2,spin_set)
    eu_full=eu_set[1]
    dim_full=2*spin_set[1].+1
    for cc=2:length(eu_set)
        eu_full=vcat(eu_full, eu_set[cc])
        dim_full=vcat(dim_full, 2*spin_set[cc].+1)
    end
    dim_full=Int.(round.(dim_full))

    order=sortperm(abs.(eu_full),rev=true);
    eu_full_sorted=eu_full[order];
    dim_full_sorted=dim_full[order];

    total_size=dim_full_sorted[1];
    old_value=eu_full_sorted[1];
    for cc=2:length(eu_full_sorted)
        total_size=total_size+dim_full_sorted[cc];
        new_value=eu_full_sorted[cc];
        if total_size>n
            break;
        end
        old_value=new_value;
    end

    for cc=1:length(eu_set)
        spins=spin_set[cc];
        eu=eu_set[cc];
        ev=ev_set[cc];
        inds=findall(x->abs(x)>=abs(old_value), abs.(eu));
        eu=eu[inds];
        ev=ev[inds];
        spins=spins[inds];
        eu_set[cc]=eu;
        ev_set[cc]=ev;
        spin_set[cc]=spins;
        if ~(ev_set2==[])
            ev2=ev_set2[cc];
            ev2=ev2[inds];
            ev_set2[cc]=ev2;
        end
    end
    return eu_set,ev_set,ev_set2,spin_set

end




function group_singlespin_sector(group_size,E_set,vL_set,vR_set,spin_set,OO_transform,OO_unitary)
    #the input tensors are combined tensor, where vL and vR correspond to each other after correct transformation

    E_set_new=copy(E_set);
    vL_set_new=copy(vL_set);
    vR_set_new=copy(vR_set);
    spin_set_new=copy(spin_set);
    DTrun_set=copy(E_set);
    
    
    for cc=1:length(E_set)
        E=E_set_new[cc];
        vL=vL_set_new[cc];
        vR=vR_set_new[cc];
        spin=spin_set_new[cc];

        if vL==[]
            DTrun=0;
        else
            DTrun=Int(dim(codomain(vL))/(2*spin[1]+1));#sector size
        end
        
        DTrun_list=[];
        
        #determine grouped structure
        if (0<DTrun)&(DTrun<group_size)
            DTrun_list=[1:DTrun];
        elseif DTrun>=group_size

            for cc=1:Int((DTrun-DTrun%group_size)/group_size)
                DTrun_list=vcat(DTrun_list,[((cc-1)*group_size+1):cc*group_size])
            end
            if DTrun%group_size>0
                DTrun_list=vcat(DTrun_list,[(DTrun-(DTrun%group_size)+1):DTrun]);
            end
        elseif DTrun==0
            continue;
        end
        
        #group tensors according to the above part
        E_new=Vector{Any}(undef, length(DTrun_list));
        vL_new=Vector{Any}(undef, length(DTrun_list));
        vR_new=Vector{Any}(undef, length(DTrun_list));

        sec=Irrep[SU₂](spin[1]);
        
        vL_fullmatrix=convert(Dict,vL)[:data][string(sec)];
        vR_fullmatrix=convert(Dict,vR)[:data][string(sec)];
        E_fullmatrix=convert(Dict,E)[:data][string(sec)];
        dim_full=size(vL_fullmatrix,2);


        for cg=1:length(DTrun_list)

            DTrun_comp=DTrun_list[cg];
            
            space_single_spin=SU₂Space(spin[1]=>length(DTrun_comp));
            
            vL_comp=vL_fullmatrix[DTrun_comp,:];
            vR_comp=vR_fullmatrix[:,DTrun_comp];
            E_comp=E_fullmatrix[DTrun_comp,DTrun_comp];

            vL_comp_rand=TensorMap(randn,space_single_spin←domain(vL));
            vR_comp_rand=TensorMap(randn,domain(vL) ← space_single_spin);
            E_comp_rand=TensorMap(randn,space_single_spin ← space_single_spin);

            vL_comp_dict=convert(Dict,vL_comp_rand);
            vL_comp_dict[:data][string(sec)]=vL_comp;
            vL_comp=convert(TensorMap,vL_comp_dict);

            vR_comp_dict=convert(Dict,vR_comp_rand);
            vR_comp_dict[:data][string(sec)]=vR_comp;
            vR_comp=convert(TensorMap,vR_comp_dict);

            E_comp_dict=convert(Dict,E_comp_rand);
            E_comp_dict[:data][string(sec)]=E_comp;
            E_comp=convert(TensorMap,E_comp_dict);


            if OO_transform
                @tensor vL_comp[:]:=vL_comp[-1,-2,1,-5]*OO_unitary[1,-3,-4];
                vL_comp=permute(vL_comp,(1,),(2,3,4,5,));
                @tensor vR_comp[:]:=vR_comp[-1,1,-4,-5]*OO_unitary'[-2,-3,1];
                vR_comp=permute(vR_comp,(1,2,3,4,),(5,));
            end

            vL_new[cg]=vL_comp;
            vR_new[cg]=vR_comp;
            E_new[cg]=E_comp;


        end
        vL_set_new[cc]=vL_new;
        vR_set_new[cc]=vR_new;
        E_set_new[cc]=E_new;
        spin_set_new[cc]=spin[1];
        DTrun_set[cc]=DTrun_list;
    end
    return E_set_new, vL_set_new, vR_set_new, spin_set_new, DTrun_set
end


function combine_singlespin_sector(E_set,vL_set,vR_set,spin_set,is_eig)
    E_set_new=copy(E_set);
    vL_set_new=copy(vL_set);
    vR_set_new=copy(vR_set);
    spin_set_new=copy(spin_set);
    for cc=1:length(E_set)
        E=E_set[cc];
        vL=vL_set[cc];
        vR=vR_set[cc];
        spin=spin_set[cc];
        if length(vL)>0

            sec=Irrep[SU₂](spin[1]);
            space_single_spin=SU₂Space(spin[1]=>length(E));
            dim_full=size(convert(Dict,vL[1])[:data][string(sec)],2);
            vL_grouped=zeros(Complex{Float64},length(vL),dim_full);
            vR_grouped=zeros(Complex{Float64},dim_full,length(vL));
            E_grouped=zeros(Complex{Float64},length(vL),length(vL));

            for cs=1:length(E)
                vL_grouped[cs,:]=convert(Dict,vL[cs])[:data][string(sec)];
                vR_grouped[:,cs]=convert(Dict,vR[cs])[:data][string(sec)];
                E_grouped[cs,cs]=E[cs];
            end

            if is_eig
                Q=zeros(Complex{Float64},length(E),length(E))
                for ca=1:length(E)
                    for cb=1:length(E)
                        Q[ca,cb]=tr(vL[ca]*vR[cb])
                    end
                end
                #Q=vL_grouped*vR_grouped;#the normalization of this line is inccorect, due to the dimension of spin space
                vL_grouped=pinv(Q)*vL_grouped;
            end

            vL_grouped_rand=TensorMap(randn,space_single_spin←domain(vL[1]));
            vR_grouped_rand=TensorMap(randn,domain(vL[1]) ← space_single_spin);
            E_grouped_rand=TensorMap(randn,space_single_spin ← space_single_spin);

            vL_grouped_dict=convert(Dict,vL_grouped_rand);
            vL_grouped_dict[:data][string(sec)]=vL_grouped;
            vL_grouped=convert(TensorMap,vL_grouped_dict);

            vR_grouped_dict=convert(Dict,vR_grouped_rand);
            vR_grouped_dict[:data][string(sec)]=vR_grouped;
            vR_grouped=convert(TensorMap,vR_grouped_dict);

            E_grouped_dict=convert(Dict,E_grouped_rand);
            E_grouped_dict[:data][string(sec)]=E_grouped;
            E_grouped=convert(TensorMap,E_grouped_dict);

            vL_set_new[cc]=vL_grouped;
            vR_set_new[cc]=vR_grouped;
            E_set_new[cc]=E_grouped;
            spin_set_new[cc]=spin_set_new[cc][1]
            # println(Base.summarysize(vL_grouped));flush(stdout);
            # println(space(vL_grouped));flush(stdout);
            # println(Base.summarysize(vL));flush(stdout);

        end
    end
    return E_set_new,vL_set_new,vR_set_new,spin_set_new
end




function combine_singlespin_sector_unitary(E_set,vL_set,vR_set,spin_set,is_eig) #this should not be used for large bond dimension
    E_set_new=copy(E_set);
    vL_set_new=copy(vL_set);
    vR_set_new=copy(vR_set);
    spin_set_new=copy(spin_set);
    for cc=1:length(E_set)
        E=E_set[cc];
        vL=vL_set[cc];
        vR=vR_set[cc];
        spin=spin_set[cc];
        if length(vL)>0

            sec=Irrep[SU₂](spin[1]);
            space_single_spin=SU₂Space(spin[1]=>length(E));
            dim_full=dim(fuse(domain(vL[1])),sec);
            vL_grouped=zeros(Complex{Float64},length(vL),dim_full);
            vR_grouped=zeros(Complex{Float64},dim_full,length(vL));
            E_grouped=zeros(Complex{Float64},length(vL),length(vL));

            Unitary=unitary(domain(vL[1]),fuse(domain(vL[1])));#this step is not possible if the bond dimension is large.
            for cs=1:length(E)
                vL_grouped[cs,:]=convert(Dict,vL[cs]*Unitary)[:data][string(sec)];
                vR_grouped[:,cs]=convert(Dict,Unitary'*vR[cs])[:data][string(sec)];
                E_grouped[cs,cs]=E[cs];
            end

            if is_eig
                Q=zeros(Complex{Float64},length(E),length(E))
                for ca=1:length(E)
                    for cb=1:length(E)
                        Q[ca,cb]=tr(vL[ca]*vR[cb])
                    end
                end
                #Q=vL_grouped*vR_grouped;#the normalization of this line is inccorect, due to the dimension of spin space
                vL_grouped=pinv(Q)*vL_grouped;
            end

            vL_grouped_rand=TensorMap(randn,space_single_spin←fuse(domain(vL[1])));
            vR_grouped_rand=TensorMap(randn,fuse(domain(vL[1])) ← space_single_spin);
            E_grouped_rand=TensorMap(randn,space_single_spin ← space_single_spin);

            vL_grouped_dict=convert(Dict,vL_grouped_rand);
            vL_grouped_dict[:data][string(sec)]=vL_grouped;
            vL_grouped=convert(TensorMap,vL_grouped_dict);
            vL_grouped=vL_grouped*Unitary';

            vR_grouped_dict=convert(Dict,vR_grouped_rand);
            vR_grouped_dict[:data][string(sec)]=vR_grouped;
            vR_grouped=convert(TensorMap,vR_grouped_dict);
            vR_grouped=Unitary*vR_grouped;

            E_grouped_dict=convert(Dict,E_grouped_rand);
            E_grouped_dict[:data][string(sec)]=E_grouped;
            E_grouped=convert(TensorMap,E_grouped_dict);

            vL_set_new[cc]=vL_grouped;
            vR_set_new[cc]=vR_grouped;
            E_set_new[cc]=E_grouped;
            spin_set_new[cc]=spin_set_new[cc][1]
            # println(Base.summarysize(vL_grouped));flush(stdout);
            # println(space(vL_grouped));flush(stdout);
            # println(Base.summarysize(vL));flush(stdout);

        end
    end
    return E_set_new,vL_set_new,vR_set_new,spin_set_new
end




# #attention: when the dimension is slightly large, converting dense matrix to tensormap is extremely difficult. It's better to define tensormap from Dict
# using TensorKit
# sp=Rep[SU₂](0=>129, 1/2=>196, 1=>185, 3/2=>124, 2=>61, 5/2=>20, 3=>4); #out of memory error for this dimension
# sp=Rep[SU₂](0=>12, 1/2=>19, 1=>15, 3/2=>14, 2=>61, 5/2=>20, 3=>4); # it takes long time
# tt=TensorMap(randn,sp←sp);
# tt_dense=convert(Array,tt);
# tt_new=TensorMap(tt_dense,sp,sp);
function combine_singlespin_sector_bruteforce(E_set,vL_set,vR_set,spin_set)
    E_set_new=copy(euL_set);
    vL_set_new=copy(evL_set);
    vR_set_new=copy(evR_set);
    spin_set_new=copy(SPIN_eig_set);
    for cc=1:length(E_set)
        E=E_set[cc];
        vL=vL_set[cc];
        vR=vR_set[cc];
        spin=spin_set[cc];
        if length(vL)>0
            single_size=size(convert(Array,vL[1]));
            if length(single_size)==4
                vL_grouped=zeros(Complex{Float64},single_size[1]*length(vL),single_size[2]*single_size[3]*single_size[4]);
                vR_grouped=zeros(Complex{Float64},single_size[2]*single_size[3]*single_size[4],single_size[1]*length(vL));
                E_grouped=zeros(Complex{Float64},single_size[1]*length(vL),single_size[1]*length(vL));
                spin_dim=single_size[1];
                Unitary=unitary(domain(vL[1]),fuse(domain(vL[1])));
                for cs=1:length(E)
                    vL_grouped[spin_dim*(cs-1)+1:spin_dim*cs,:]=convert(Array,vL[cs]*Unitary);
                    vR_grouped[:,spin_dim*(cs-1)+1:spin_dim*cs]=convert(Array,Unitary'*vR[cs]);
                    E_grouped[spin_dim*(cs-1)+1:spin_dim*cs,spin_dim*(cs-1)+1:spin_dim*cs]=E[cs]*Matrix(I, spin_dim,spin_dim);
                end
                space_single_spin=SU₂Space(spin[1]=>length(E));

                vL_grouped=TensorMap(vL_grouped,space_single_spin←fuse(domain(vL[1])));
                vR_grouped=TensorMap(vR_grouped,fuse(domain(vL[1])) ← space_single_spin);
                E_grouped=TensorMap(E_grouped,space_single_spin ← space_single_spin);

                vL_grouped=vL_grouped*Unitary';
                vR_grouped=Unitary * vR_grouped;

                vL_set_new[cc]=vL_grouped;
                vR_set_new[cc]=vR_grouped;
                E_set_new[cc]=E_grouped;
                spin_set_new[cc]=spin_set[cc][1]
                # println(Base.summarysize(vL_grouped));flush(stdout);
                # println(space(vL_grouped));flush(stdout);
                # println(Base.summarysize(vL));flush(stdout);
            elseif length(single_size)==3
            end
        end
    end
    return E_set_new,vL_set_new,vR_set_new,spin_set_new
end
