function contract_1D_with_plaquatte(mps_set1,mps_set2,p1a,p1b,p2a,p2b,h_plaquatte)
    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy);
    U_ss_ss=unitary(fuse(space(U_ss,1)*space(U_ss,1)), space(U_ss,1)*space(U_ss,1));
    U_ss_ss=permute(U_ss_ss',(3,1,2,));

    Lx=length(mps_set1);
    if (length(vcat(p1a,p1b))==0)&(length(vcat(p2a,p2b))==0)

        cx=1
        @tensor env[:]:=mps_set1[cx][-1,1]*mps_set2[cx][-2,1];
        for cx=2:Lx-1
            @tensor env[:]:=env[1,3]*mps_set1[cx][1,-1,2]*mps_set2[cx][3,-2,2];
        end
        cx=Lx;
        Norm=@tensor env[1,2]*mps_set1[cx][1,3]*mps_set2[cx][2,3];

    elseif (length(vcat(p1a,p1b))==2)&(length(vcat(p2a,p2b))==0)
        if p1a==1
            cx=1;
            @tensor env[:]:=mps_set1[cx][-2,1,-1]*mps_set2[cx][-3,1];
            cx=2;
            @tensor env[:]:=env[-1,1,3]*mps_set1[cx][1,-3,2,-2]*mps_set2[cx][3,-4,2];

            #@tensor env[:]:=env[1,2,-1,-2]*U_ss_ss[1,6,3]*U_ss_ss[2,5,4]*h_plaquatte[3,4,5,6];#pay attention to order of physical index

            for cx=3:Lx-1
                @tensor env[:]:=env[-1,-2,1,3]*mps_set1[cx][1,-3,2]*mps_set2[cx][3,-4,2];
            end
            cx=Lx;
            @tensor Norm[:]:= env[-1,-2,1,2]*mps_set1[cx][1,3]*mps_set2[cx][2,3];
        elseif (1<p1a)&(p1b<Lx)
            cx=1;
            @tensor env[:]:=mps_set1[cx][-1,1]*mps_set2[cx][-2,1];
            for cx=2:p1a-1
                @tensor env[:]:=env[1,3]*mps_set1[cx][1,-1,2]*mps_set2[cx][3,-2,2];
            end
            cx=p1a;
            @tensor env[:]:=env[1,3]*mps_set1[cx][1,-2,2,-1]*mps_set2[cx][3,-3,2];
            cx=p1a+1;
            @tensor env[:]:=env[-1,1,3]*mps_set1[cx][1,-3,2,-2]*mps_set2[cx][3,-4,2];

            for cx=p1b+1:Lx-1
                @tensor env[:]:=env[-1,-2,1,3]*mps_set1[cx][1,-3,2]*mps_set2[cx][3,-4,2];
            end

            cx=Lx;
            @tensor Norm[:]:= env[-1,-2,1,2]*mps_set1[cx][1,3]*mps_set2[cx][2,3];
        elseif p1b==Lx
            cx=1;
            @tensor env[:]:=mps_set1[cx][-1,1]*mps_set2[cx][-2,1];
            for cx=2:Lx-2
                @tensor env[:]:=env[1,3]*mps_set1[cx][1,-1,2]*mps_set2[cx][3,-2,2];
            end
            cx=Lx-1;
            @tensor env[:]:=env[1,3]*mps_set1[cx][1,-2,2,-1]*mps_set2[cx][3,-3,2];
            cx=Lx;
            @tensor Norm[:]:=env[-1,1,3]*mps_set1[cx][1,2,-2]*mps_set2[cx][3,2];
        end

    elseif (length(vcat(p1a,p1b))==0)&(length(vcat(p2a,p2b))==2)
        if p2a==1
            cx=1;
            @tensor env[:]:=mps_set1[cx][-2,1]*mps_set2[cx][-3,1,-1];
            cx=2;
            @tensor env[:]:=env[-1,1,3]*mps_set1[cx][1,-3,2]*mps_set2[cx][3,-4,2,-2];

            #@tensor env[:]:=env[1,2,-1,-2]*U_ss_ss[1,3,6]*U_ss_ss[2,4,5]*h_plaquatte[3,4,5,6];#pay attention to order of physical index

            for cx=3:Lx-1
                #@tensor env[:]:=env[1,3]*mps_set1[cx][1,-1,2]*mps_set2[cx][3,-2,2];
                @tensor env[:]:=env[-1,-2,1,3]*mps_set1[cx][1,-3,2]*mps_set2[cx][3,-4,2];
            end
            cx=Lx;
            #Norm=@tensor env[1,2]*mps_set1[cx][1,3]*mps_set2[cx][2,3];
            @tensor Norm[:]:= env[-1,-2,1,2]*mps_set1[cx][1,3]*mps_set2[cx][2,3];
        elseif (1<p2a)&(p2b<Lx)
            cx=1;
            @tensor env[:]:=mps_set1[cx][-1,1]*mps_set2[cx][-2,1];
            for cx=2:p2a-1
                @tensor env[:]:=env[1,3]*mps_set1[cx][1,-1,2]*mps_set2[cx][3,-2,2];
            end
            cx=p2a;
            @tensor env[:]:=env[1,3]*mps_set1[cx][1,-2,2]*mps_set2[cx][3,-3,2,-1];
            cx=p2a+1;
            @tensor env[:]:=env[-1,1,3]*mps_set1[cx][1,-3,2]*mps_set2[cx][3,-4,2,-2];

            for cx=p2b+1:Lx-1
                @tensor env[:]:=env[-1,-2,1,3]*mps_set1[cx][1,-3,2]*mps_set2[cx][3,-4,2];
            end

            cx=Lx;
            @tensor Norm[:]:= env[-1,-2,1,2]*mps_set1[cx][1,3]*mps_set2[cx][2,3];
        elseif p2b==Lx
            cx=1;
            @tensor env[:]:=mps_set1[cx][-1,1]*mps_set2[cx][-2,1];
            for cx=2:Lx-2
                @tensor env[:]:=env[1,3]*mps_set1[cx][1,-1,2]*mps_set2[cx][3,-2,2];
            end
            cx=Lx-1;
            @tensor env[:]:=env[1,3]*mps_set1[cx][1,-2,2]*mps_set2[cx][3,-3,2,-1];
            cx=Lx;
            @tensor Norm[:]:=env[-1,1,3]*mps_set1[cx][1,2]*mps_set2[cx][3,2,-2];
        end
    elseif (length(vcat(p1a,p1b))==2)&(length(vcat(p2a,p2b))==2)
    end


    return Norm
end