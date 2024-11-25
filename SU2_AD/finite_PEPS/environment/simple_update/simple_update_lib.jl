function check_convergence(lambdaset,lambdaset_old,Lx,Ly)
    Lx,Ly=size(lambdaset);
    err_set=Matrix{Float64}(undef,Lx,Ly);
    for ca=1:Lx
        for cb=1:Ly
            if isassigned(lambdaset_old,ca,cb)
                es1=convert(Array,lambdaset[ca,cb]);
                es2=convert(Array,lambdaset_old[ca,cb]);
                if size(es1)==size(es2)
                    err_set[ca,cb]=norm(es1-es2);
                else
                    err_set[ca,cb]=100;
                end
            else
                err_set[ca,cb]=0;
            end
        end
    end

    return err_set
end

function prepare_gate_Heisenberg(dt)
    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");
    H=permute(H_Heisenberg,(1,2,),(3,4,));
    eu,ev=eigh(H);
    @assert norm(ev*eu*ev'-H)/norm(H)<1e-14 
    gate=ev*exp(-dt*eu)*ev';
    return gate
end

function initial_tensor(Lx,Ly,d,D)
    lambdax_set=Matrix{Any}(undef,Lx+1,Ly);#to the left of site (ca,cb) 
    lambday_set=Matrix{Any}(undef,Lx,Ly+1);#to the bot of site (ca,cb)
    T_set=Matrix{Any}(undef,Lx,Ly)

    
    T_set=Matrix{Any}(undef,Lx,Ly);
    for ca=1:Lx
        for cb=1:Ly
            if ca==1
                Dl=1;
                Dr=D;
                if cb==1
                    Dd=1;
                    Du=D;
                elseif 1<cb<Ly
                    Dd=D;
                    Du=D;
                elseif cb==Ly
                    Dd=D;
                    Du=1;
                end
            elseif 1<ca<Lx
                Dl=D;
                Dr=D;
                if cb==1
                    Dd=1;
                    Du=D;
                elseif 1<cb<Ly
                    Dd=D;
                    Du=D;
                elseif cb==Ly
                    Dd=D;
                    Du=1;
                end
            elseif ca==Lx
                Dl=D;
                Dr=1;
                if cb==1
                    Dd=1;
                    Du=D;
                elseif 1<cb<Ly
                    Dd=D;
                    Du=D;
                elseif cb==Ly
                    Dd=D;
                    Du=1;
                end
            end
            T=TensorMap(randn,(ℂ^Dl)*(ℂ^Dd)*(ℂ^Dr)'*(ℂ^Du)',(ℂ^d));
            T=permute(T,(1,2,3,4,5,));
            T_set[ca,cb]=T;
        end
    end

    for ca=1:Lx
        for cb=1:Ly
            vr=space(T_set[ca,cb],1);
            lambdax_set[ca,cb]=unitary(vr,vr);

            vd=space(T_set[ca,cb],2);
            lambday_set[ca,cb]=unitary(vd',vd');
        end
    end

    ca=Lx+1;
    for cb=1:Ly
        vr=space(T_set[ca-1,cb],3);
        lambdax_set[ca,cb]=unitary(vr',vr');
    end

    cb=Ly+1;
    for ca=1:Lx
        vd=space(T_set[ca,cb-1],4);
        lambday_set[ca,cb]=unitary(vd,vd);
    end

    return T_set,lambdax_set,lambday_set
end


function simple_update_Heisenberg(T_set,lambdax_set,lambday_set,tau,dt,Dmax)
    gate=prepare_gate_Heisenberg(dt);
    Lx,Ly=size(T_set);
    for ct=1:Int(round(tau/dt));
        for px=1.5:2:Lx-0.5
            for py=1:Ly
                # println([px,py])
                T_set,lambdax_set,lambday_set=tebd_xbond(T_set,lambdax_set,lambday_set, gate,px,py,Dmax);
            end
        end
        for px=2.5:2:Lx-0.5
            for py=1:Ly
                # println([px,py])
                T_set,lambdax_set,lambday_set=tebd_xbond(T_set,lambdax_set,lambday_set, gate,px,py,Dmax);
            end
        end

        for px=1:Lx
            for py=1.5:2:Ly-0.5
                # println([px,py])
                T_set,lambdax_set,lambday_set=tebd_ybond(T_set,lambdax_set,lambday_set, gate,px,py,Dmax);
            end
        end
        for px=1:Lx
            for py=2.5:2:Ly-0.5
                # println([px,py])
                T_set,lambdax_set,lambday_set=tebd_ybond(T_set,lambdax_set,lambday_set, gate,px,py,Dmax);
            end
        end

    end
    return T_set,lambdax_set,lambday_set
end


function tebd_xbond(Tset,lambdaxset,lambdayset, gate,px,py,Dmax);
    pos1=[Int(px-0.5),py];
    pos2=[Int(px+0.5),py];

    λ1=lambdaxset[pos1[1],pos1[2]];
    λ2=lambdayset[pos1[1],pos1[2]];
    λ3=lambdayset[pos1[1],pos1[2]+1];
    λ4=lambdayset[pos2[1],pos2[2]];
    λ5=lambdaxset[pos2[1]+1,pos2[2]];
    λ6=lambdayset[pos2[1],pos2[2]+1];

    T1=Tset[pos1[1],pos1[2]];
    T2=Tset[pos2[1],pos2[2]];

    @tensor T1[:]:=T1[1,2,-3,3,-5]*λ1[-1,1]*λ2[2,-2]*λ3[-4,3];
    @tensor T2[:]:=T2[-1,1,2,3,-5]*λ4[1,-2]*λ5[2,-3]*λ6[-4,3];

    u,s,v=tsvd(permute(T1,(1,2,4,),(3,5,)));#L1,D1,U1,newbond1,     newbond1,R1,d1
    T1_left=u;
    T1_keep=s*v;

    u,s,v=tsvd(permute(T2,(1,5,),(2,3,4,)));#L2,d2,newbond2,     newbond2,D2,R2,U2
    T2_left=v;
    T2_keep=u*s;

    @tensor Tbond[:]:=T1_keep[-1,1,2]*T2_keep[1,3,-3]*gate[-2,-4,2,3];#newbond1,R1,d1  L2,d2,newbond2 -> newbond1,d1  ,newbond2,d2
    u,s,v=tsvd(permute(Tbond,(1,2,),(3,4,)); trunc=truncdim(Dmax));#newbond1,d1  ,newbond2,d2 -> newbond1,d1,R1,    L2,newbond2,d2
    T1_keep=u*sqrt(s);
    T2_keep=sqrt(s)*v;

    @tensor T1[:]:=T1_left[-1,-2,-4,1]*T1_keep[1,-5,-3];#L1,D1,U1,newbond1    newbond1,d1,R1
    @tensor T2[:]:=T2_keep[-1,1,-5]*T2_left[1,-2,-3,-4];#L2,newbond2,d2,     newbond2,D2,R2,U2 
    
    λ1_inv=my_pinv(λ1);
    λ2_inv=my_pinv(λ2);
    λ3_inv=my_pinv(λ3);
    λ4_inv=my_pinv(λ4);
    λ5_inv=my_pinv(λ5);
    λ6_inv=my_pinv(λ6);
    @tensor T1[:]:=T1[1,2,-3,3,-5]*λ1_inv[-1,1]*λ2_inv[2,-2]*λ3_inv[-4,3];
    @tensor T2[:]:=T2[-1,1,2,3,-5]*λ4_inv[1,-2]*λ5_inv[2,-3]*λ6_inv[-4,3];

    T1=T1/norm(T1);
    T2=T2/norm(T2);
    s=s/norm(s);
    lambdaxset[pos2[1],pos2[2]]=sqrt(s);
    Tset[pos1[1],pos1[2]]=T1;
    Tset[pos2[1],pos2[2]]=T2;
    return Tset,lambdaxset,lambdayset
end

function tebd_ybond(Tset,lambdaxset,lambdayset, gate,px,py,Dmax);
    pos1=[px,Int(py+0.5)];
    pos2=[px,Int(py-0.5)];

    λ1=lambdaxset[pos1[1],pos1[2]];
    λ2=lambdaxset[pos1[1]+1,pos1[2]];
    λ3=lambdayset[pos1[1],pos1[2]+1];
    λ4=lambdaxset[pos2[1],pos2[2]];
    λ5=lambdayset[pos2[1],pos2[2]];
    λ6=lambdaxset[pos2[1]+1,pos2[2]];

    T1=Tset[pos1[1],pos1[2]];
    T2=Tset[pos2[1],pos2[2]];

    @tensor T1[:]:=T1[1,-2,2,3,-5]*λ1[-1,1]*λ2[2,-3]*λ3[-4,3];
    @tensor T2[:]:=T2[1,2,3,-4,-5]*λ4[-1,1]*λ5[2,-2]*λ6[3,-3];

    u,s,v=tsvd(permute(T1,(1,3,4,),(2,5,)));#L1,R1,U1,newbond1,     newbond1,D1,d1
    T1_left=u;
    T1_keep=s*v;

    u,s,v=tsvd(permute(T2,(4,5,),(1,2,3,)));#U2,d2,newbond2,     newbond2,L2,D2,R2
    T2_keep=u*s;
    T2_left=v;

    @tensor Tbond[:]:=T1_keep[-1,1,2]*T2_keep[1,3,-3]*gate[-2,-4,2,3];#newbond1,D1,d1  U2,d2,newbond2 -> newbond1,d1  newbond2,d2
    u,s,v=tsvd(permute(Tbond,(1,2,),(3,4,)); trunc=truncdim(Dmax));#newbond1,d1  ,newbond2,d2 -> newbond1,d1,D1,    U2,newbond2,d2
    T1_keep=u*sqrt(s);
    T2_keep=sqrt(s)*v;

    @tensor T1[:]:=T1_left[-1,-3,-4,1]*T1_keep[1,-5,-2];#L1,R1,U1,newbond1,    newbond1,d1,D1
    @tensor T2[:]:=T2_keep[-4,1,-5]*T2_left[1,-1,-2,-3];#U2,newbond2,d2     newbond2,L2,D2,R2
    
    λ1_inv=my_pinv(λ1);
    λ2_inv=my_pinv(λ2);
    λ3_inv=my_pinv(λ3);
    λ4_inv=my_pinv(λ4);
    λ5_inv=my_pinv(λ5);
    λ6_inv=my_pinv(λ6);
    @tensor T1[:]:=T1[1,-2,2,3,-5]*λ1_inv[-1,1]*λ2_inv[2,-3]*λ3_inv[-4,3];
    @tensor T2[:]:=T2[1,2,3,-4,-5]*λ4_inv[-1,1]*λ5_inv[2,-2]*λ6_inv[3,-3];

    T1=T1/norm(T1);
    T2=T2/norm(T2);
    s=s/norm(s);
    lambdayset[pos1[1],pos1[2]]=sqrt(s);
    Tset[pos1[1],pos1[2]]=T1;
    Tset[pos2[1],pos2[2]]=T2;


    return Tset,lambdaxset,lambdayset
end






function SU_triangle_trun(T_set,lambdax_set,lambday_set,plaquatte,sites,op,Dmax)
    x_range=[plaquatte[1],plaquatte[1]+1];
    y_range=[plaquatte[2],plaquatte[2]+1];
    # T_plaquatte=psi[x_range,y_range];

    if sites=="123"
        pos1=[x_range[1],y_range[2]];
        pos2=[x_range[2],y_range[2]];
        pos3=[x_range[2],y_range[1]];
        T1=T_set[pos1[1],pos1[2]];
        T2=T_set[pos2[1],pos2[2]];
        T3=T_set[pos3[1],pos3[2]];
        λ1=lambdax_set[pos1[1],pos1[2]];
        λ2=lambday_set[pos1[1],pos1[2]];
        λ3=lambday_set[pos1[1],pos1[2]+1];
        λ4=lambdax_set[pos2[1]+1,pos2[2]];
        λ5=lambday_set[pos2[1],pos2[2]+1];
        λ6=lambdax_set[pos3[1],pos3[2]];
        λ7=lambday_set[pos3[1],pos3[2]];
        λ8=lambdax_set[pos3[1]+1,pos3[2]];

        @tensor T1[:]:=T1[1,2,-3,4,-5]*λ1[-1,1]*λ2[2,-2]*λ3[-4,4];
        @tensor T2[:]:=T2[-1,-2,3,4,-5]*λ4[3,-3]*λ5[-4,4];
        @tensor T3[:]:=T3[1,2,3,-4,-5]*λ6[-1,1]*λ7[2,-2]*λ8[3,-3];

        u1,s1,v1=tsvd(permute(T1,(1,2,4,),(3,5,)));#L1,D1,U1,   R1,d1
        T1_left=u1;#L1,D1,U1,newbond1
        T1_keep=s1*v1;#newbond1,R1,d1

        u3,s3,v3=tsvd(permute(T3,(4,5,),(1,2,3,)));#U3,d3    L3,D3,R3,   
        T3_keep=u3*s3;#U3,d3,newbond3
        T3_left=v3;#newbond3,L3,D3,R3,  

        @tensor T2_new[:]:=T1_keep[-1,1,-5]*T2[1,2,-3,-4,-6]*T3_keep[2,-7,-2];#newbond1,R1,d1   L2,D2,R2,U2,d2   U3,d3,newbond3  
        @tensor T2_new[:]:=T2_new[-1,-2,-3,-4,1,2,3]*op[-5,-6,-7,1,2,3];#newbond1,newbond3,R2,U2,d1,d2,d3, 

        u1,s1,v1=tsvd(permute(T2_new,(1,5,),(2,3,4,6,7,)); trunc=truncdim(Dmax));#newbond1,d1,   newbond3,R2,U2,d2,d3, 
        T1_keep=u1*sqrt(s1);#newbond1,d1,R1,
        @tensor T1_new[:]:=T1_left[-1,-2,-4,1]*T1_keep[1,-5,-3];#L1,D1,U1,newbond1      newbond1,d1,R1,

        T2T3=s1*v1;#L2,newbond3,R2,U2,d2,d3,
        u3,s3,v3=tsvd(permute(T2T3,(1,3,4,5,),(2,6,)); trunc=truncdim(Dmax));#L2,R2,U2,d2,      newbond3,d3,   
        T2_new=permute(u3*sqrt(s3),(1,5,2,3,4,));#L2,R2,U2,d2,D2
        T3_keep=sqrt(s3)*v3;#U3,newbond3,d3,
        @tensor T3_new[:]:=T3_keep[-4,1,-5]*T3_left[1,-1,-2,-3];#U3,newbond3,d3,  newbond3,L3,D3,R3,  

        s1_inv_sqrt=sqrt(my_pinv(s1));
        @tensor T2_new[:]:=T2_new[1,-2,-3,-4,-5]*s1_inv_sqrt[-1,1];

        λ1_inv=my_pinv(λ1);
        λ2_inv=my_pinv(λ2);
        λ3_inv=my_pinv(λ3);
        λ4_inv=my_pinv(λ4);
        λ5_inv=my_pinv(λ5);
        λ6_inv=my_pinv(λ6);
        λ7_inv=my_pinv(λ7);
        λ8_inv=my_pinv(λ8);
        @tensor T1_new[:]:=T1_new[1,2,-3,4,-5]*λ1_inv[-1,1]*λ2_inv[2,-2]*λ3_inv[-4,4];
        @tensor T2_new[:]:=T2_new[-1,-2,3,4,-5]*λ4_inv[3,-3]*λ5_inv[-4,4];
        @tensor T3_new[:]:=T3_new[1,2,3,-4,-5]*λ6_inv[-1,1]*λ7_inv[2,-2]*λ8_inv[3,-3];

        s1=s1/norm(s1);
        s3=s3/norm(s3);
        lambdax_set[pos2[1],pos2[2]]=sqrt(s1);
        lambday_set[pos2[1],pos2[2]]=sqrt(s3);

        T1_new=T1_new/norm(T1_new);
        T2_new=T2_new/norm(T2_new);
        T3_new=T3_new/norm(T3_new);
        T_set[pos1[1],pos1[2]]=T1_new;
        T_set[pos2[1],pos2[2]]=T2_new;
        T_set[pos3[1],pos3[2]]=T3_new;


    elseif sites=="234"
        pos1=[x_range[2],y_range[2]];
        pos2=[x_range[2],y_range[1]];
        pos3=[x_range[1],y_range[1]];
        T1=T_set[pos1[1],pos1[2]];
        T2=T_set[pos2[1],pos2[2]];
        T3=T_set[pos3[1],pos3[2]];
        λ1=lambdax_set[pos1[1],pos1[2]];
        λ2=lambdax_set[pos1[1]+1,pos1[2]];
        λ3=lambday_set[pos1[1],pos1[2]+1];
        λ4=lambday_set[pos2[1],pos2[2]];
        λ5=lambdax_set[pos2[1]+1,pos2[2]];
        λ6=lambdax_set[pos3[1],pos3[2]];
        λ7=lambday_set[pos3[1],pos3[2]];
        λ8=lambday_set[pos3[1],pos3[2]+1];

        @tensor T1[:]:=T1[1,-2,3,4,-5]*λ1[-1,1]*λ2[3,-3]*λ3[-4,4];
        @tensor T2[:]:=T2[-1,2,3,-4,-5]*λ4[2,-2]*λ5[3,-3];
        @tensor T3[:]:=T3[1,2,-3,4,-5]*λ6[-1,1]*λ7[2,-2]*λ8[-4,4];

        u1,s1,v1=tsvd(permute(T1,(1,3,4,),(2,5,)));#L1,R1,U1,newbond1,  newbond1,D1,d1   
        T1_left=u1;#L1,R1,U1,newbond1  
        T1_keep=s1*v1;#newbond1,D1,d1  

        u3,s3,v3=tsvd(permute(T3,(3,5,),(1,2,4,)));#R3,d3,newbond3,  newbond3,L3,D3,U3 
        T3_keep=u3*s3;#R3,d3,newbond3 
        T3_left=v3;#newbond3,L3,D3,U3 

        @tensor T2_new[:]:=T1_keep[-4,1,-5]*T2[2,-2,-3,1,-6]*T3_keep[2,-7,-1];#newbond1,D1,d1,   L2,D2,R2,U2,d2,   R3,d3,newbond3   
        @tensor T2_new[:]:=T2_new[-1,-2,-3,-4,1,2,3]*op[-5,-6,-7,1,2,3];#newbond3,D2,R2,newbond1,d1,d2,d3

        u1,s1,v1=tsvd(permute(T2_new,(4,5,),(1,2,3,6,7,)); trunc=truncdim(Dmax));#newbond1,d1,    newbond3,D2,R2,d2,d3
        T1_keep=u1*sqrt(s1);#newbond1,d1,D1,
        @tensor T1_new[:]:=T1_left[-1,-3,-4,1]*T1_keep[1,-5,-2];#L1,R1,U1,newbond1,   newbond1,d1,D1

        T2T3=s1*v1;#U2,newbond3,D2,R2,d2,d3
        u3,s3,v3=tsvd(permute(T2T3,(1,3,4,5,),(2,6,)); trunc=truncdim(Dmax));#U2,D2,R2,d2,    newbond3,d3
        T2_new=permute(u3*sqrt(s3),(5,2,3,1,4,));
        T3_keep=sqrt(s3)*v3;#R3,newbond3,d3
        @tensor T3_new[:]:=T3_keep[-3,1,-5]*T3_left[1,-1,-2,-4];#R3,newbond3,d3    newbond3,L3,D3,U3 

        s1_inv_sqrt=sqrt(my_pinv(s1));
        @tensor T2_new[:]:=T2_new[-1,-2,-3,1,-5]*s1_inv_sqrt[-4,1];

        λ1_inv=my_pinv(λ1);
        λ2_inv=my_pinv(λ2);
        λ3_inv=my_pinv(λ3);
        λ4_inv=my_pinv(λ4);
        λ5_inv=my_pinv(λ5);
        λ6_inv=my_pinv(λ6);
        λ7_inv=my_pinv(λ7);
        λ8_inv=my_pinv(λ8);
        @tensor T1_new[:]:=T1_new[1,-2,3,4,-5]*λ1_inv[-1,1]*λ2_inv[3,-3]*λ3_inv[-4,4];
        @tensor T2_new[:]:=T2_new[-1,2,3,-4,-5]*λ4_inv[2,-2]*λ5_inv[3,-3];
        @tensor T3_new[:]:=T3_new[1,2,-3,4,-5]*λ6_inv[-1,1]*λ7_inv[2,-2]*λ8_inv[-4,4];

        s1=s1/norm(s1);
        s3=s3/norm(s3);
        lambday_set[pos1[1],pos1[2]]=sqrt(s1);
        lambdax_set[pos2[1],pos2[2]]=permute(sqrt(s3),(2,),(1,));

        T1_new=T1_new/norm(T1_new);
        T2_new=T2_new/norm(T2_new);
        T3_new=T3_new/norm(T3_new);
        T_set[pos1[1],pos1[2]]=T1_new;
        T_set[pos2[1],pos2[2]]=T2_new;
        T_set[pos3[1],pos3[2]]=T3_new;

    elseif sites=="341"
        pos1=[x_range[2],y_range[1]];
        pos2=[x_range[1],y_range[1]];
        pos3=[x_range[1],y_range[2]];
        T1=T_set[pos1[1],pos1[2]];
        T2=T_set[pos2[1],pos2[2]];
        T3=T_set[pos3[1],pos3[2]];
        λ1=lambday_set[pos1[1],pos1[2]];
        λ2=lambdax_set[pos1[1]+1,pos1[2]];
        λ3=lambday_set[pos1[1],pos1[2]+1];
        λ4=lambdax_set[pos2[1],pos2[2]];
        λ5=lambday_set[pos2[1],pos2[2]];
        λ6=lambdax_set[pos3[1],pos3[2]];
        λ7=lambdax_set[pos3[1]+1,pos3[2]];
        λ8=lambday_set[pos3[1],pos3[2]+1];
        @tensor T1[:]:=T1[-1,2,3,4,-5]*λ1[2,-2]*λ2[3,-3]*λ3[-4,4];
        @tensor T2[:]:=T2[1,2,-3,-4,-5]*λ4[-1,1]*λ5[2,-2];
        @tensor T3[:]:=T3[1,-2,3,4,-5]*λ6[-1,1]*λ7[3,-3]*λ8[-4,4];

        u1,s1,v1=tsvd(permute(T1,(1,5,),(2,3,4,)));#L1,d1,newbond1,    newbond1,D1,R1,U1   
        T1_left=v1;#newbond1,D1,R1,U1
        T1_keep=u1*s1;#L1,d1,newbond1,

        u3,s3,v3=tsvd(permute(T3,(1,3,4,),(2,5,)));#L3,R3,U3,newbond3,     newbond3, D3,d3,  
        T3_keep=s3*v3;#newbond3, D3,d3,  
        T3_left=u3;#L3,R3,U3,newbond3,

        @tensor T2_new[:]:=T3_keep[-4,1,-7]*T2[-1,-2,2,1,-6]*T1_keep[2,-5,-3];#newbond3, D3,d3,   L2,D2,R2,U2,d2,   L1,d1,newbond1,  
        @tensor T2_new[:]:=T2_new[-1,-2,-3,-4,1,2,3]*op[-5,-6,-7,1,2,3];#L2,D2,newbond1,newbond3,d1,d2,d3

        u1,s1,v1=tsvd(permute(T2_new,(1,2,4,6,7,),(3,5,)); trunc=truncdim(Dmax));#L2,D2,newbond3,d2,d3    newbond1,d1,
        T1_keep=sqrt(s1)*v1;#L1,newbond1,d1,
        @tensor T1_new[:]:=T1_left[1,-2,-3,-4]*T1_keep[-1,1,-5];#newbond1,D1,R1,U1     L1,newbond1,d1,

        T2T3=u1*s1;#L2,D2,newbond3,d2,d3,R2
        u3,s3,v3=tsvd(permute(T2T3,(3,5,),(1,2,4,6,)); trunc=truncdim(Dmax));#newbond3,d3,    L2,D2,d2,R2
        T2_new=permute(sqrt(s3)*v3,(2,3,5,1,4,));#U2,L2,D2,d2,R2
        T3_keep=u3*sqrt(s3);#newbond3,d3,D3
        @tensor T3_new[:]:=T3_left[-1,-3,-4,1]*T3_keep[1,-5,-2];#L3,R3,U3,newbond3,    newbond3,d3,D3
        
        s1_inv_sqrt=sqrt(my_pinv(s1));
        @tensor T2_new[:]:=T2_new[-1,-2,1,-4,-5]*s1_inv_sqrt[1,-3];

        λ1_inv=my_pinv(λ1);
        λ2_inv=my_pinv(λ2);
        λ3_inv=my_pinv(λ3);
        λ4_inv=my_pinv(λ4);
        λ5_inv=my_pinv(λ5);
        λ6_inv=my_pinv(λ6);
        λ7_inv=my_pinv(λ7);
        λ8_inv=my_pinv(λ8);
        @tensor T1_new[:]:=T1_new[-1,2,3,4,-5]*λ1_inv[2,-2]*λ2_inv[3,-3]*λ3_inv[-4,4];
        @tensor T2_new[:]:=T2_new[1,2,-3,-4,-5]*λ4_inv[-1,1]*λ5_inv[2,-2];
        @tensor T3_new[:]:=T3_new[1,-2,3,4,-5]*λ6_inv[-1,1]*λ7_inv[3,-3]*λ8_inv[-4,4];

        s1=s1/norm(s1);
        s3=s3/norm(s3);
        lambdax_set[pos1[1],pos1[2]]=sqrt(s1);
        lambday_set[pos3[1],pos3[2]]=sqrt(s3);

        T1_new=T1_new/norm(T1_new);
        T2_new=T2_new/norm(T2_new);
        T3_new=T3_new/norm(T3_new);
        T_set[pos1[1],pos1[2]]=T1_new;
        T_set[pos2[1],pos2[2]]=T2_new;
        T_set[pos3[1],pos3[2]]=T3_new;


    elseif sites=="412"
        pos1=[x_range[1],y_range[1]];
        pos2=[x_range[1],y_range[2]];
        pos3=[x_range[2],y_range[2]];
        T1=T_set[pos1[1],pos1[2]];
        T2=T_set[pos2[1],pos2[2]];
        T3=T_set[pos3[1],pos3[2]];
        λ1=lambdax_set[pos1[1],pos1[2]];
        λ2=lambday_set[pos1[1],pos1[2]];
        λ3=lambdax_set[pos1[1]+1,pos1[2]];
        λ4=lambdax_set[pos2[1],pos2[2]];
        λ5=lambday_set[pos2[1],pos2[2]+1];
        λ6=lambday_set[pos3[1],pos3[2]];
        λ7=lambdax_set[pos3[1]+1,pos3[2]];
        λ8=lambday_set[pos3[1],pos3[2]+1];

        @tensor T1[:]:=T1[1,2,3,-4,-5]*λ1[-1,1]*λ2[2,-2]*λ3[3,-3];
        @tensor T2[:]:=T2[1,-2,-3,4,-5]*λ4[-1,1]*λ5[-4,4];
        @tensor T3[:]:=T3[-1,2,3,4,-5]*λ6[2,-2]*λ7[3,-3]*λ8[-4,4];

        u1,s1,v1=tsvd(permute(T1,(1,2,3,),(4,5,)));#L1,D1,R1,newbond1,  newbond1,U1,d1   
        T1_left=u1;#L1,D1,R1,newbond1
        T1_keep=s1*v1;#newbond1,U1,d1  

        u3,s3,v3=tsvd(permute(T3,(1,5,),(2,3,4,)));#L3,d3,newbond3,  newbond3,D3,R3,U3 
        T3_keep=u3*s3;#L3,d3,newbond3, 
        T3_left=v3;#newbond3,D3,R3,U3 

        @tensor T2_new[:]:=T1_keep[-2,1,-5]*T2[-1,1,2,-4,-6]*T3_keep[2,-7,-3];#newbond1,U1,d1     L2,D2,R2,U2,d2,   L3,d3,newbond3, 
        @tensor T2_new[:]:=T2_new[-1,-2,-3,-4,1,2,3]*op[-5,-6,-7,1,2,3];#L2,newbond1,newbond3,U2,d1,d2,d3

        u1,s1,v1=tsvd(permute(T2_new,(2,5,),(1,3,4,6,7,)); trunc=truncdim(Dmax));#newbond1,d1,    L2,newbond3,U2,d2,d3
        T1_keep=u1*sqrt(s1);#newbond1,d1,U1
        @tensor T1_new[:]:=T1_left[-1,-2,-3,1]*T1_keep[1,-5,-4];#L1,D1,R1,newbond1,   newbond1,d1,U1 

        T2T3=s1*v1;#D2,L2,newbond3,U2,d2,d3
        u3,s3,v3=tsvd(permute(T2T3,(1,2,4,5,),(3,6,)); trunc=truncdim(Dmax));#D2,L2,U2,d2,    newbond3,d3,
        T2_new=permute(u3*sqrt(s3),(2,1,5,3,4,));#D2,L2,U2,d2,R2
        T3_keep=sqrt(s3)*v3;#L3,newbond3,d3,
        @tensor T3_new[:]:=T3_keep[-1,1,-5]*T3_left[1,-2,-3,-4];#L3,newbond3,d3,    newbond3,D3,R3,U3 

        s1_inv_sqrt=sqrt(my_pinv(s1));
        @tensor T2_new[:]:=T2_new[-1,1,-3,-4,-5]*s1_inv_sqrt[-2,1];

        λ1_inv=my_pinv(λ1);
        λ2_inv=my_pinv(λ2);
        λ3_inv=my_pinv(λ3);
        λ4_inv=my_pinv(λ4);
        λ5_inv=my_pinv(λ5);
        λ6_inv=my_pinv(λ6);
        λ7_inv=my_pinv(λ7);
        λ8_inv=my_pinv(λ8);
        @tensor T1_new[:]:=T1_new[1,2,3,-4,-5]*λ1_inv[-1,1]*λ2_inv[2,-2]*λ3_inv[3,-3];
        @tensor T2_new[:]:=T2_new[1,-2,-3,4,-5]*λ4_inv[-1,1]*λ5_inv[-4,4];
        @tensor T3_new[:]:=T3_new[-1,2,3,4,-5]*λ6_inv[2,-2]*λ7_inv[3,-3]*λ8_inv[-4,4];

        s1=s1/norm(s1);
        s3=s3/norm(s3);
        lambday_set[pos2[1],pos2[2]]=permute(sqrt(s1),(2,),(1,));
        lambdax_set[pos3[1],pos3[2]]=sqrt(s3);

        T1_new=T1_new/norm(T1_new);
        T2_new=T2_new/norm(T2_new);
        T3_new=T3_new/norm(T3_new);
        T_set[pos1[1],pos1[2]]=T1_new;
        T_set[pos2[1],pos2[2]]=T2_new;
        T_set[pos3[1],pos3[2]]=T3_new;
    end


    return T_set,lambdax_set,lambday_set
end

function simple_update_triangle(parameters,T_set,lambdax_set,lambday_set,tau,dt,Dmax)
    Lx,Ly=size(T_set);
    triangle_set_set=get_triangles(parameters,Lx,Ly,dt);
    
    for ct=1:Int(round(tau/dt));
        for c1=1:length(triangle_set_set)
            triangle_set=triangle_set_set[c1];
            
            for c2=1:length(triangle_set)
                triangle=triangle_set[c2];
                T_set,lambdax_set,lambday_set=SU_triangle_trun(T_set,lambdax_set,lambday_set,triangle["plaquatte"], triangle["sites"], triangle["gate"], Dmax);
                # println(triangle["plaquatte"])
                # println(triangle["sites"])
            end
        end
    end
    return T_set,lambdax_set,lambday_set
end