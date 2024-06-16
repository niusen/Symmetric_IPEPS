function split_T1(T,Lx,Ly,pos_T,ind,bond_type)
    posx,posy=pos_T;
    if bond_type=="x"#left tensor
        if Rank(T)==3
            if posy==1
                u,s,v=my_tsvd(permute(T,(2,),(1,3,)));#Du,Dr,d,
                u=u;#Du,D_new
                v=s*v;#D_new,Dr,d,
                u=permute(u,(2,1,));#D_new,Du
                v=permute(v,(1,3,2,));#D_new,d,Dr
                T_=u;#D_new,Du
                T_keep=v;#D_new,d,Dr
            elseif posy==Ly
                u,s,v=my_tsvd(permute(T,(1,),(2,3,)));#Dd,Dr,d,
                u=u;#Dd,D_new
                v=s*v;#D_new,Dr,d,
                u=permute(u,(1,2,));#Dd,D_new
                v=permute(v,(1,3,2,));#D_new,d,Dr
                T_=u;#Dd,D_new
                T_keep=v;#D_new,d,Dr
            end
        elseif Rank(T)==4
            if posx==1
                #Dd,Dr,Du,d
                u,s,v=my_tsvd(permute(T,(1,3,),(2,4,)));#Dd,Du,Dr,d,
                u=u;#Dd,Du,D_new
                v=s*v;#D_new,Dr,d,
                u=permute(u,(1,3,2,));#Dd,D_new,Du
                v=permute(v,(1,3,2,));#D_new,d,Dr
                T_=u;#Dd,D_new,Du
                T_keep=v;#D_new,d,Dr
            elseif 1<posx<Lx
                if posy==1
                    #Dl,Dr,Du,d
                    u,s,v=my_tsvd(permute(T,(1,3,),(2,4,)));#Dl,Du,Dr,d,
                    u=u;#Dl,Du,D_new
                    v=s*v;#D_new,Dr,d,
                    u=permute(u,(1,3,2,));#Dl,D_new,Du
                    v=permute(v,(1,3,2,));#D_new,d,Dr
                    T_=u;#Dl,D_new,Du
                    T_keep=v;#D_new,d,Dr
                elseif posy==Ly
                    #Dl,Dd,Dr,d
                    u,s,v=my_tsvd(permute(T,(1,2,),(3,4,)));#Dl,Dd,Dr,d,
                    u=u;#Dl,Dd,D_new
                    v=s*v;#D_new,Dr,d,
                    u=permute(u,(1,2,3,));#Dl,Dd,D_new
                    v=permute(v,(1,3,2,));#D_new,d,Dr
                    T_=u;#Dl,Dd,D_new
                    T_keep=v;#D_new,d,Dr
                end
            elseif posx==Lx
                #nothing
            end
        elseif Rank(T)==5
            u,s,v=my_tsvd(permute(T,(1,2,4,),(3,5,)));#Dl,Dd,Du,Dr,d,
            u=u;#Dl,Dd,Du,D_new
            v=s*v;#D_new,Dr,d,
            u=permute(u,(1,2,4,3,));#Dl,Dd,D_new,Du
            v=permute(v,(1,3,2,));#D_new,d,Dr
            T_=u;#Dl,Dd,D_new,Du
            T_keep=v;#D_new,d,Dr

        end
    elseif bond_type=="y"#up tensor
        if Rank(T)==3
            if posx==1
                #Dd,Dr,d
                u,s,v=my_tsvd(permute(T,(2,),(1,3,)));#Dr,Dd,d,
                u=u;#Dr,D_new
                v=s*v;#D_new,Dd,d,
                u=permute(u,(2,1,));#D_new,Dr
                v=permute(v,(1,3,2,));#D_new,d,Dd
                T_=u;#D_new,Dr
                T_keep=v;#D_new,d,Dd
            elseif posx==Lx
                #Dl,Dd,d
                u,s,v=my_tsvd(permute(T,(1,),(2,3,)));#Dl,Dd,d,
                u=u;#Dl,D_new
                v=s*v;#D_new,Dd,d,
                u=permute(u,(1,2,));#Dl,D_new
                v=permute(v,(1,3,2,));#D_new,d,Dd
                T_=u;#Dl,D_new
                T_keep=v;#D_new,d,Dd
            end
        elseif Rank(T)==4
            if posy==1
                #nothing
            elseif 1<posy<Ly
                if posx==1
                    #Dd,Dr,Du,d
                    u,s,v=my_tsvd(permute(T,(2,3,),(1,4,)));#Dr,Du,Dd,d,
                    u=u;#Dr,Du,D_new
                    v=s*v;#D_new,Dd,d,
                    u=permute(u,(3,1,2,));#D_new,Dr,Du
                    v=permute(v,(1,3,2,));#D_new,d,Dd
                    T_=u;#D_new,Dr,Du
                    T_keep=v;#D_new,d,Dd
                elseif posx==Lx
                    #Dl,Dd,Du,d
                    u,s,v=my_tsvd(permute(T,(1,3,),(2,4,)));#Dl,Du,Dd,d,
                    u=u;#Dl,Du,D_new
                    v=s*v;#D_new,Dd,d,
                    u=permute(u,(1,3,2,));#Dl,D_new,Du
                    v=permute(v,(1,3,2,));#D_new,d,Dd
                    T_=u;#Dl,D_new,Du
                    T_keep=v;#D_new,d,Dd
                end
            elseif posy==Ly
                #Dl,Dd,Dr,d
                u,s,v=my_tsvd(permute(T,(1,3,),(2,4,)));#Dl,Dr,Dd,d,
                u=u;#Dl,Dr,D_new
                v=s*v;#D_new,Dd,d,
                u=permute(u,(1,3,2,));#Dl,D_new,Dr
                v=permute(v,(1,3,2,));#D_new,d,Dd
                T_=u;#Dl,D_new,Dr
                T_keep=v;#D_new,d,Dd
            end
        elseif Rank(T)==5
            #Dl,Dd,Dr,Du,d
            u,s,v=my_tsvd(permute(T,(1,3,4,),(2,5,)));#Dl,Dr,Du,Dd,d,
            u=u;#Dl,Dr,Du,D_new
            v=s*v;#D_new,Dd,d,
            u=permute(u,(1,4,2,3,));#Dl,D_new,Dr,Du
            v=permute(v,(1,3,2,));#D_new,d,Dd
            T_=u;#Dl,D_new,Dr,Du
            T_keep=v;#D_new,d,Dd
        end
    end
    return T_,T_keep
end

function split_T2(T,Lx,Ly,pos_T,ind,bond_type)
    posx,posy=pos_T;
    if bond_type=="x"#right tensor
        if Rank(T)==3
            if posy==1
                #Dl,Du,d
                u,s,v=my_tsvd(permute(T,(1,3,),(2,)));#Dl,d,Du
                u=u*s;#Dl,d,D_new
                v=v;#D_new,Du
                T_keep=permute(u,(1,3,2,));#Dl,D_new,d
                T_=v;#D_new,Du
            elseif posy==Ly
                #Dl,Dd,d
                u,s,v=my_tsvd(permute(T,(1,3,),(2,)));#Dl,d,Dd
                u=u*s;#Dl,d,D_new
                v=v;#D_new,Dd
                T_keep=permute(u,(1,3,2,));#Dl,D_new,d
                T_=v;#D_new,Dd
            end
        elseif Rank(T)==4
            if posx==1
                #nothing
            elseif 1<posx<Lx
                if posy==1
                    #Dl,Dr,Du,d
                    u,s,v=my_tsvd(permute(T,(1,4,),(2,3,)));#Dl,d,Dr,Du
                    u=u*s;#Dl,d,D_new
                    v=v;#D_new,Dr,Du
                    T_keep=permute(u,(1,3,2,));#Dl,D_new,d
                    T_=v;#D_new,Dr,Du
                elseif posy==Ly
                    #Dl,Dd,Dr,d
                    u,s,v=my_tsvd(permute(T,(1,4,),(2,3,)));#Dl,d,Dd,Dr
                    u=u*s;#Dl,d,D_new
                    v=v;#D_new,Dd,Dr
                    T_keep=permute(u,(1,3,2,));#Dl,D_new,d
                    T_=v;#D_new,Dd,Dr
                end
            elseif posx==Lx
                #Dl,Dd,Du,d
                u,s,v=my_tsvd(permute(T,(1,4,),(2,3,)));#Dl,d,Dd,Du
                u=u*s;#Dl,d,D_new
                v=v;#D_new,Dd,Du
                T_keep=permute(u,(1,3,2,));#Dl,D_new,d
                T_=v;#D_new,Dd,Du   
            end
        elseif Rank(T)==5
            u,s,v=my_tsvd(permute(T,(1,5,),(2,3,4,)));#Dl,d,Dd,Dr,Du
            u=u*s;#Dl,d,D_new
            v=v;#D_new,Dd,Dr,Du
            T_keep=permute(u,(1,3,2,));#Dl,D_new,d
            T_=v;#D_new,Dd,Dr,Du
        end
    elseif bond_type=="y"#down tensor
        if Rank(T)==3
            if posx==1
                #Dr,Du,d
                u,s,v=my_tsvd(permute(T,(2,3,),(1,)));#Du,d,Dr
                u=u*s;#Du,d,D_new
                v=v;#D_new,Dr
                T_keep=permute(u,(1,3,2,));#Du,D_new,d
                T_=permute(v,(2,1,));#Dr,D_new
            elseif posx==Lx
                #Dl,Du,d
                u,s,v=my_tsvd(permute(T,(2,3,),(1,)));#Du,d,Dl
                u=u*s;#Du,d,D_new
                v=v;#D_new,Dl
                T_keep=permute(u,(1,3,2,));#Du,D_new,d
                T_=permute(v,(2,1,));#Dl,D_new
            end
        elseif Rank(T)==4
            if posy==1
                #Dl,Dr,Du,d
                u,s,v=my_tsvd(permute(T,(3,4,),(1,2,)));#Du,d,Dl,Dr
                u=u*s;#Du,d,D_new
                v=v;#D_new,Dl,Dr
                T_keep=permute(u,(1,3,2,));#Du,D_new,d
                T_=permute(v,(2,3,1,));#Dl,Dr,D_new
            elseif 1<posy<Ly
                if posx==1
                    #Dd,Dr,Du,d
                    u,s,v=my_tsvd(permute(T,(3,4,),(1,2,)));#Du,d,Dd,Dr
                    u=u*s;#Du,d,D_new
                    v=v;#D_new,Dd,Dr
                    T_keep=permute(u,(1,3,2,));#Du,D_new,d
                    T_=permute(v,(2,3,1,));#Dd,Dr,D_new
                elseif posx==Lx
                    #Dl,Dd,Du,d
                    u,s,v=my_tsvd(permute(T,(3,4,),(1,2,)));#Du,d,Dl,Dd
                    u=u*s;#Du,d,D_new
                    v=v;#D_new,Dl,Dd
                    T_keep=permute(u,(1,3,2,));#Du,D_new,d
                    T_=permute(v,(2,3,1,));#Dl,Dd,D_new
                end
            elseif posy==Ly
                #nothing
            end
        elseif Rank(T)==5
            #Dl,Dd,Dr,Du,d
            u,s,v=my_tsvd(permute(T,(4,5,),(1,2,3,)));#Du,d,Dl,Dd,Dr
            u=u*s;#Du,d,D_new
            v=v;#D_new,Dl,Dd,Dr
            T_keep=permute(u,(1,3,2,));#Du,D_new,d
            T_=permute(v,(2,3,4,1,));#Dl,Dd,Dr,D_new
        end
    end
    return T_,T_keep
end



function recover_T1(T_,T_keep,Lx,Ly,pos_T,ind,bond_type)
    posx,posy=pos_T;
    if bond_type=="x"#left tensor
        if Rank(T_)==3-1
            if posy==1
                @tensor T[:]:=T_[1,-1]*T_keep[1,-2,-3]; #D_new,Du  D_new,d,Dr  => Du,d,Dr
                T=permute(T,(3,1,2,));
            elseif posy==Ly
                @tensor T[:]:=T_[-1,1]*T_keep[1,-2,-3]; #Dd,D_new  D_new,d,Dr => Dd,d,Dr
                T=permute(T,(1,3,2,));
            end
        elseif Rank(T_)==4-1
            if posx==1
                @tensor T[:]:=T_[-1,1,-2]*T_keep[1,-3,-4]; #Dd,D_new,Du   D_new,d,Dr => Dd,Du,d,Dr 
                T=permute(T,(1,4,2,3,));
            elseif 1<posx<Lx
                if posy==1
                    @tensor T[:]:=T_[-1,1,-2]*T_keep[1,-3,-4]; #Dl,D_new,Du  D_new,d,Dr => Dl,Du, d,Dr
                    T=permute(T,(1,4,2,3,));
                elseif posy==Ly
                    @tensor T[:]:=T_[-1,-2,1]*T_keep[1,-3,-4]; #Dl,Dd,D_new  D_new,d,Dr => Dl,Dd,d,Dr
                    T=permute(T,(1,2,4,3,));
                end
            elseif posx==Lx
                #nothing
            end
        elseif Rank(T_)==5-1
            @tensor T[:]:=T_[-1,-2,1,-3]*T_keep[1,-4,-5]; #Dl,Dd,D_new,Du,      Dnew,d,Dr  => Dl,Dd,Du,d,Dr
            T=permute(T,(1,2,5,3,4,));
        end
    elseif bond_type=="y"#up tensor
        if Rank(T_)==3-1
            if posx==1
                @tensor T[:]:=T_[1,-1]*T_keep[1,-2,-3]; #D_new,Dr   D_new,d,Dd => Dr,d,Dd
                T=permute(T,(3,1,2,));
            elseif posx==Lx
                @tensor T[:]:=T_[-1,1]*T_keep[1,-2,-3]; #Dl,D_new   D_new,d,Dd => Dl,d,Dd
                T=permute(T,(1,3,2,));
            end
        elseif Rank(T_)==4-1
            if posy==1
                #nothing
            elseif 1<posy<Ly
                if posx==1
                    @tensor T[:]:=T_[1,-1,-2]*T_keep[1,-3,-4]; #D_new,Dr,Du  D_new,d,Dd => Dr,Du,  d,Dd
                    T=permute(T,(4,1,2,3,));
                elseif posx==Lx
                    @tensor T[:]:=T_[-1,1,-2]*T_keep[1,-3,-4]; #Dl,D_new,Du  D_new,d,Dd => Dl,Du,d,Dd
                    T=permute(T,(1,4,2,3,));
                end
            elseif posy==Ly
                @tensor T[:]:=T_[-1,1,-2]*T_keep[1,-3,-4]; #Dl,D_new,Dr   D_new,d,Dd => Dl,Dr, d,Dd 
                T=permute(T,(1,4,2,3,));
            end
        elseif Rank(T_)==5-1
            @tensor T[:]:=T_[-1,1,-2,-3]*T_keep[1,-4,-5]; #Dl,D_new,Dr,Du   D_new,d,Dd => Dl,Dr,Du,d,Dd
            T=permute(T,(1,5,2,3,4,));
        end
    end
    return T
end

function recover_T2(T_,T_keep,Lx,Ly,pos_T,ind,bond_type)
    posx,posy=pos_T;
    if bond_type=="x"#right tensor
        if Rank(T_)==3-1
            if posy==1
                @tensor T[:]:=T_keep[-1,1,-2]*T_[1,-3]; #Dl,D_new,d   D_new,Du => Dl,d,Du
                T=permute(T,(1,3,2,));
            elseif posy==Ly
                @tensor T[:]:=T_keep[-1,1,-2]*T_[1,-3]; #Dl,D_new,d  D_new,Dd => Dl,d,Dd
                T=permute(T,(1,3,2,));
            end
        elseif Rank(T_)==4-1
            if posx==1
                #nothing
            elseif 1<posx<Lx
                if posy==1
                    @tensor T[:]:=T_keep[-1,1,-2]*T_[1,-3,-4]; #Dl,D_new,d  D_new,Dr,Du  => Dl,d, Dr,Du
                    T=permute(T,(1,3,4,2,));
                elseif posy==Ly
                    @tensor T[:]:=T_keep[-1,1,-2]*T_[1,-3,-4]; #Dl,d,Dd,Dr =>
                    T=permute(T,(1,3,4,2,));
                end
            elseif posx==Lx
                @tensor T[:]:=T_keep[-1,1,-2]*T_[1,-3,-4]; #Dl,D_new,d  D_new,Dd,Du   => Dl,d,Dd,Du
                T=permute(T,(1,3,4,2,));
            end
        elseif Rank(T_)==5-1
            @tensor T[:]:=T_keep[-1,1,-2]*T_[1,-3,-4,-5]; #Dl,Dnew,d,    D_new,Dd,Dr,Du  => Dl,d,Dd,Dr,Du
            T=permute(T,(1,3,4,5,2,));
        end
    elseif bond_type=="y"#down tensor
        if Rank(T_)==3-1
            if posx==1
                @tensor T[:]:=T_keep[-1,1,-2]*T_[-3,1]; #Du,D_new,d  Dr,D_new => Du,d,  Dr
                T=permute(T,(3,1,2,));
            elseif posx==Lx
                @tensor T[:]:=T_keep[-1,1,-2]*T_[-3,1]; #Du,D_new,d  Dl,D_new => Du,d,  Dl
                T=permute(T,(3,1,2,));
            end
        elseif Rank(T_)==4-1
            if posy==1
                @tensor T[:]:=T_keep[-1,1,-2]*T_[-3,-4,1]; #Du,D_new,d  Dl,Dr,D_new => Du,d,  Dl,Dr
                T=permute(T,(3,4,1,2,));
            elseif 1<posy<Ly
                if posx==1
                    @tensor T[:]:=T_keep[-1,1,-2]*T_[-3,-4,1]; #Du,D_new,d  Dd,Dr,D_new => Du,d,  Dd,Dr
                    T=permute(T,(3,4,1,2,));
                elseif posx==Lx
                    @tensor T[:]:=T_keep[-1,1,-2]*T_[-3,-4,1]; #Du,D_new,d  Dl,Dd,D_new => Du,d,  Dl,Dd
                    T=permute(T,(3,4,1,2,));
                end
            elseif posy==Ly
                #nothing
            end
        elseif Rank(T_)==5-1
            @tensor T[:]:=T_keep[-1,1,-2]*T_[-3,-4,-5,1]; #Du,D_new,d   Dl,Dd,Dr,D_new => Du,d,Dl,Dd,Dr
            T=permute(T,(3,4,5,1,2,));
        end
    end
    return T
end


function get_bond(psi,px,py,trun_bond_type,Noise)
    psi_left=deepcopy(psi);
    Lx,Ly=size(psi_left);
    @assert 1<=px<=Lx;
    @assert 1<=py<=Ly;
    if mod(px,1)==0.5
        bond_type="x";
    elseif mod(py,1)==0.5
        bond_type="y";
    else
        error("unknown bond");
    end
    if bond_type=="x"
        pos_T1=[px-0.5,py];
        pos_T2=[px+0.5,py];
        ind_T1=3;
        ind_T2=1;
    elseif bond_type=="y"
        pos_T1=[px,py+0.5];
        pos_T2=[px,py-0.5];
        ind_T1=2;
        ind_T2=4;
    end
    pos_T1=Int.(pos_T1);
    pos_T2=Int.(pos_T2);
    ind_T1=examine_ind(Lx,Ly,pos_T1,ind_T1);
    ind_T2=examine_ind(Lx,Ly,pos_T2,ind_T2);
    @assert space(psi_left[pos_T1[1],pos_T1[2]],ind_T1)==space(psi_left[pos_T2[1],pos_T2[2]],ind_T2)';

    if trun_bond_type=="full"
        #To Do
    elseif trun_bond_type=="dD"
        
        T1=psi_left[pos_T1[1],pos_T1[2]];
        T2=psi_left[pos_T2[1],pos_T2[2]];
    
        T1_,T1_keep=split_T1(T1,Lx,Ly,pos_T1,ind_T1,bond_type);
        T2_,T2_keep=split_T2(T2,Lx,Ly,pos_T2,ind_T2,bond_type);
    
        global D_connect
        D_connect=dim(space(T1_keep,3));#used for later svd truncation

        @tensor T_bond[:]:=T1_keep[-1,-2,1]*T2_keep[1,-3,-4];#D1,d1,D2,d2
        function add_noise(A,noise)
            A_noise=TensorMap(randn,codomain(A),domain(A))+im*TensorMap(randn,codomain(A),domain(A));
            A=A+A_noise*noise*norm(A)/norm(A_noise);
            return A
        end
        T_bond=add_noise(T_bond,Noise);
        
    end
    psi_left[pos_T1[1],pos_T1[2]]=T1_;
    psi_left[pos_T2[1],pos_T2[2]]=T2_;
    

    return T_bond,psi_left
end

function set_bond(psi_left,psi_double,px,py,T_bond)
    Lx,Ly=size(psi_left);
    @assert 1<=px<=Lx;
    @assert 1<=py<=Ly;
    if mod(px,1)==0.5
        bond_type="x";
    elseif mod(py,1)==0.5
        bond_type="y";
    else
        error("unknown bond");
    end
    if bond_type=="x"
        pos_T1=[px-0.5,py];
        pos_T2=[px+0.5,py];
        ind_T1=3;
        ind_T2=1;
    elseif bond_type=="y"
        pos_T1=[px,py+0.5];
        pos_T2=[px,py-0.5];
        ind_T1=2;
        ind_T2=4;
    end
    pos_T1=Int.(pos_T1);
    pos_T2=Int.(pos_T2);
    ind_T1=examine_ind(Lx,Ly,pos_T1,ind_T1);
    ind_T2=examine_ind(Lx,Ly,pos_T2,ind_T2);
    # println(space(psi_left[pos_T1[1],pos_T1[2]]))
    # println(space(psi_left[pos_T2[1],pos_T2[2]]))
    # println(space(T_bond))
    # println(space(psi_left[pos_T1[1],pos_T1[2]],ind_T1))
    # println(space(psi_left[pos_T2[1],pos_T2[2]],ind_T2))
    # @assert space(psi_left[pos_T1[1],pos_T1[2]],ind_T1)==space(psi_left[pos_T2[1],pos_T2[2]],ind_T2)';

    T_bond=permute(T_bond,(1,2,),(3,4,));#Dnew1,d1,Dnew2,d2
    # u,s,v=tsvd(T_bond; trunc=truncerr(1e-12));
    u,s,v=my_tsvd(T_bond; trunc=truncerr(1e-12));
    global chi
    @assert dim(space(s,1))<chi; #otherwise multiplets will be truncated

    T1_keep=u;#Dnew1,d1,D1 
    T2_keep=s*v;#D2,Dnew2,d2
    @assert norm(T1_keep*T2_keep-T_bond)/norm(T_bond)<1e-13;

    T1=recover_T1(psi_left[pos_T1[1],pos_T1[2]], T1_keep, Lx,Ly,pos_T1,ind_T1,bond_type);
    T2=recover_T2(psi_left[pos_T2[1],pos_T2[2]], T2_keep, Lx,Ly,pos_T2,ind_T2,bond_type);

    

    psi_left,psi_double=update_prepare_local(psi_left,psi_double,pos_T1[1],pos_T1[2],T1);
    psi_left,psi_double=update_prepare_local(psi_left,psi_double,pos_T2[1],pos_T2[2],T2);
    psi_new=psi_left;
    return psi_new,psi_double
end

function cost_fun_bond(x,psi_left,psi_double) #variational parameters are vector of TensorMap
    global chi, parameters,px,py

    E=energy_disk_new(x,psi_left,psi_double,px,py,"bond");
    E=real(E);
    global E_tem;
    E_tem=E;
    return E
end


################################

function set_bond_cut(psi_left,psi_double,px,py,T1_keep,T2_keep)
    Lx,Ly=size(psi_left);
    @assert 1<=px<=Lx;
    @assert 1<=py<=Ly;
    if mod(px,1)==0.5
        bond_type="x";
    elseif mod(py,1)==0.5
        bond_type="y";
    else
        error("unknown bond");
    end
    if bond_type=="x"
        pos_T1=[px-0.5,py];
        pos_T2=[px+0.5,py];
        ind_T1=3;
        ind_T2=1;
    elseif bond_type=="y"
        pos_T1=[px,py+0.5];
        pos_T2=[px,py-0.5];
        ind_T1=2;
        ind_T2=4;
    end
    pos_T1=Int.(pos_T1);
    pos_T2=Int.(pos_T2);
    ind_T1=examine_ind(Lx,Ly,pos_T1,ind_T1);
    ind_T2=examine_ind(Lx,Ly,pos_T2,ind_T2);




    T1=recover_T1(psi_left[pos_T1[1],pos_T1[2]], T1_keep, Lx,Ly,pos_T1,ind_T1,bond_type);
    T2=recover_T2(psi_left[pos_T2[1],pos_T2[2]], T2_keep, Lx,Ly,pos_T2,ind_T2,bond_type);

    

    psi_left,psi_double=update_prepare_local(psi_left,psi_double,pos_T1[1],pos_T1[2],T1);
    psi_left,psi_double=update_prepare_local(psi_left,psi_double,pos_T2[1],pos_T2[2],T2);
    psi_new=psi_left;
    return psi_new,psi_double
end





