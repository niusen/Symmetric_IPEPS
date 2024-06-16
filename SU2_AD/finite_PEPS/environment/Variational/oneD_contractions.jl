function norm_1D_no_conjugate(mps_set1,mps_set2)
    Lx=length(mps_set1);

    cx=1
    @tensor env[:]:=mps_set1[cx][-1,1]*mps_set2[cx][-2,1];
    for cx=2:Lx-1
        @tensor env[:]:=env[1,3]*mps_set1[cx][1,-1,2]*mps_set2[cx][3,-2,2];
    end
    cx=Lx;
    Norm=@tensor env[1,2]*mps_set1[cx][1,3]*mps_set2[cx][2,3];
    return Norm
end

function norm_1D_no_conjugate(mps_set1,mpo,mps_set2)
    Lx=length(mps_set1);

    cx=1
    @tensor env[:]:=mps_set1[cx][-1,1]*mpo[cx][2,-2,1]*mps_set2[cx][-3,2];
    for cx=2:Lx-1
        @tensor env[:]:=env[1,3,5]*mps_set1[cx][1,-1,2]*mpo[cx][3,4,-2,2]*mps_set2[cx][5,-3,4];
    end
    cx=Lx;
    Norm=@tensor env[1,3,5]*mps_set1[cx][1,2]*mpo[cx][3,4,2]*mps_set2[cx][5,4];
    return Norm
end

function derivative_3row_top(mps_up,mpo,mps_down,px)
    Lx=length(mps_up);
    global U_phy
    I_phy=unitary(U_phy,U_phy);
    U_s_s=@ignore_derivatives unitary(fuse(U_phy'*U_phy), U_phy' ⊗ U_phy);

    # if px==1
    #     @tensor envR[:]:=mps_up[Lx][-1,1]*mps_down[Lx][-2,1];
    #     for cc=Lx-1:-1:2
    #         @tensor envR[:]:=mps_up[cc][-1,1,2]*mps_down[cc][-2,3,2]*envR[1,3];
    #     end
    #     @tensor Norm[:]:=mps_down[1][1,-2]*envR[-1,1]*I_phy[7,8]*U_s_s'[7,8,-3];

    # elseif 1<px<Lx
    #     @tensor envL[:]:=mps_up[1][-1,1]*mps_down[1][-2,1];
    #     @tensor envR[:]:=mps_up[Lx][-1,1]*mps_down[Lx][-2,1];
    #     for cc=2:px-1
    #         @tensor envL[:]:=mps_up[cc][1,-1,2]*mps_down[cc][3,-2,2]*envL[1,3];
    #     end
    #     for cc=Lx-1:-1:px+1
    #         @tensor envR[:]:=mps_up[cc][-1,1,2]*mps_down[cc][-2,3,2]*envR[1,3];
    #     end
    #     @tensor Norm[:]:=mps_down[px][1,2,-3]*envL[-1,1]*envR[-2,2]*I_phy[7,8]*U_s_s'[7,8,-4];
    # elseif px==Lx
    #     @tensor envL[:]:=mps_up[1][-1,1]*mps_down[1][-2,1];
    #     for cc=2:Lx-1
    #         @tensor envL[:]:=mps_up[cc][1,-1,2]*mps_down[cc][3,-2,2]*envL[1,3];
    #     end
    #     @tensor Norm[:]:=mps_down[Lx][1,-2]*envL[-1,1]*I_phy[7,8]*U_s_s'[7,8,-3];
    # end

    if px==1
        @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
        for cc=Lx-1:-1:2
            @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
        end
        @tensor Norm[:]:=mpo[1][1,3,-2]*mps_down[1][2,1]*envR[-1,3,2]*I_phy[7,8]*U_s_s'[7,8,-3];

    elseif 1<px<Lx
        @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
        @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
        for cc=2:px-1
            @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
        end
        for cc=Lx-1:-1:px+1
            @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
        end
        @tensor Norm[:]:=mpo[px][2,3,4,-3]*mps_down[px][1,5,3]*envL[-1,2,1]*envR[-2,4,5]*I_phy[7,8]*U_s_s'[7,8,-4];
    elseif px==Lx
        @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
        for cc=2:Lx-1
            @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
        end
        @tensor Norm[:]:=mpo[Lx][2,1,-2]*mps_down[Lx][3,1]*envL[-1,2,3]*I_phy[7,8]*U_s_s'[7,8,-3];
    end

    return Norm
end

function derivative_3row_bot(mps_up,mpo,mps_down,px)
    Lx=length(mps_up);
    global U_phy
    I_phy=unitary(U_phy,U_phy);
    U_s_s=@ignore_derivatives unitary(fuse(U_phy'*U_phy), U_phy' ⊗ U_phy);

    # if px==1
    #     @tensor envR[:]:=mps_up[Lx][-1,1]*mps_down[Lx][-2,1];
    #     for cc=Lx-1:-1:2
    #         @tensor envR[:]:=mps_up[cc][-1,1,2]*mps_down[cc][-2,3,2]*envR[1,3];
    #     end
    #     @tensor Norm[:]:=mps_up[1][1,-2]*envR[1,-1]*I_phy[7,8]*U_s_s'[7,8,-3];

    # elseif 1<px<Lx
    #     @tensor envL[:]:=mps_up[1][-1,1]*mps_down[1][-2,1];
    #     @tensor envR[:]:=mps_up[Lx][-1,1]*mps_down[Lx][-2,1];
    #     for cc=2:px-1
    #         @tensor envL[:]:=mps_up[cc][1,-1,2]*mps_down[cc][3,-2,2]*envL[1,3];
    #     end
    #     for cc=Lx-1:-1:px+1
    #         @tensor envR[:]:=mps_up[cc][-1,1,2]*mps_down[cc][-2,3,2]*envR[1,3];
    #     end
    #     @tensor Norm[:]:=mps_up[px][1,2,-3]*envL[1,-1]*envR[2,-2]*I_phy[7,8]*U_s_s'[7,8,-4];
    # elseif px==Lx
    #     @tensor envL[:]:=mps_up[1][-1,1]*mps_down[1][-2,1];
    #     for cc=2:Lx-1
    #         @tensor envL[:]:=mps_up[cc][1,-1,2]*mps_down[cc][3,-2,2]*envL[1,3];
    #     end
    #     @tensor Norm[:]:=mps_up[Lx][1,-2]*envL[1,-1]*I_phy[7,8]*U_s_s'[7,8,-3];
    # end

    if px==1
        @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
        for cc=Lx-1:-1:2
            @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
        end
        @tensor Norm[:]:=mps_up[1][2,1]*mpo[1][-2,3,1]*envR[2,3,-1]*I_phy[7,8]*U_s_s'[7,8,-3];

    elseif 1<px<Lx
        @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
        @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
        for cc=2:px-1
            @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
        end
        for cc=Lx-1:-1:px+1
            @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
        end
        @tensor Norm[:]:=mps_up[px][1,4,3]*mpo[px][2,-3,5,3]*envL[1,2,-1]*envR[4,5,-2]*I_phy[7,8]*U_s_s'[7,8,-4];
    elseif px==Lx
        @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
        for cc=2:Lx-1
            @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
        end
        @tensor Norm[:]:=mps_up[Lx][2,1]*mpo[Lx][3,-2,1]*envL[2,3,-1]*I_phy[7,8]*U_s_s'[7,8,-3];
    end

    return Norm
end

function derivative_3row_middle(mps_up,mpo,mps_down,px)
    Lx=length(mps_up);
    global U_phy
    I_phy=unitary(U_phy,U_phy);
    U_s_s=@ignore_derivatives unitary(fuse(U_phy'*U_phy), U_phy' ⊗ U_phy);

    if px==1
        @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
        for cc=Lx-1:-1:2
            @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
        end
        @tensor Norm[:]:=mps_up[1][2,-3]*mps_down[1][1,-1]*envR[2,-2,1]*I_phy[7,8]*U_s_s'[7,8,-4];

    elseif 1<px<Lx
        @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
        @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
        for cc=2:px-1
            @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
        end
        for cc=Lx-1:-1:px+1
            @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
        end
        @tensor Norm[:]:=mps_up[px][1,3,-4]*mps_down[px][4,2,-2]*envL[1,-1,4]*envR[3,-3,2]*I_phy[7,8]*U_s_s'[7,8,-5];
    elseif px==Lx
        @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
        for cc=2:Lx-1
            @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
        end
        @tensor Norm[:]:=mps_up[Lx][2,-3]*mps_down[Lx][1,-2]*envL[2,-1,1]*I_phy[7,8]*U_s_s'[7,8,-4];
    end

    return Norm
end

function contract_3Row_plaquatte_mps_site(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap) #first row has plaquatte, second row has empty site
    L=length(mps_up);

    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy);
    U_ss_ss=unitary(fuse(space(U_ss,1)*space(U_ss,1)), space(U_ss,1)*space(U_ss,1));
    U_ss_ss=permute(U_ss_ss',(3,1,2,));

    T1=mps_up[x_range[1]];
    if Rank(T1)==2+1
        @tensor T1_closed[:]:=T1[-1,-2,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T1[:]:=T1[-1,-2,1]*U_ss_ss[1,-4,-3];
    elseif Rank(T1)==3+1
        @tensor T1_closed[:]:=T1[-1,-2,-3,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T1[:]:=T1[-1,-2,-3,1]*U_ss_ss[1,-5,-4];
    end

    T2=mps_up[x_range[2]];
    if Rank(T2)==2+1
        @tensor T2_closed[:]:=T2[-1,-2,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T2[:]:=T2[-1,-2,1]*U_ss_ss[1,-4,-3];
    elseif Rank(T2)==3+1
        @tensor T2_closed[:]:=T2[-1,-2,-3,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T2[:]:=T2[-1,-2,-3,1]*U_ss_ss[1,-5,-4];
    end

    u,s,v=tsvd(h_plaquatte,(1,4,),(2,3,));
    v=s*v;
    U_tem=unitary(fuse(space(v,1)*space(T2,1)), space(v,1)*space(T2,1));

    if Rank(T1)==2+2
        @tensor T1[:]:=T1[3,-2,1,2]*u[1,2,4]*U_tem'[4,3,-1];
    elseif Rank(T1)==3+2
        @tensor T1[:]:=T1[-1,3,-3,1,2]*u[1,2,4]*U_tem'[4,3,-2];
    end

    if Rank(T2)==2+2
        @tensor T2[:]:=T2[4,-2,1,2]*v[3,1,2]*U_tem[-1,3,4];
    elseif Rank(T2)==3+2
        @tensor T2[:]:=T2[4,-2,-3,1,2]*v[3,1,2]*U_tem[-1,3,4];
    end


    ########################################################
    # #verification energy
    # mps_up1=deepcopy(mps_up);
    # mps_up2=deepcopy(mps_up);
    # mps_up1[x_range[1]]=T1;
    # mps_up1[x_range[2]]=T2;
    # mps_up2[x_range[1]]=T1_closed;
    # mps_up2[x_range[2]]=T2_closed;
    # ee=norm_1D_no_conjugate(mps_up1,mps_down);
    # Norm=norm_1D_no_conjugate(mps_up2,mps_down);
    # println(ee/Norm)
    #######################################################

    mps_up[x_range[1]]=T1;
    mps_up[x_range[2]]=T2;
    term=derivative_3row_bot(mps_up,mpo,mps_down,px);
    return term
end



function contract_3Row_site_mps_plaquatte(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap) #first row has empty site, second row has plaquatte
    L=length(mps_up);

    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy);
    U_ss_ss=unitary(fuse(space(U_ss,1)*space(U_ss,1)), space(U_ss,1)*space(U_ss,1));
    U_ss_ss=permute(U_ss_ss',(3,1,2,));

    T1=mps_down[x_range[1]];

    if Rank(T1)==2+1
        @tensor T1_closed[:]:=T1[-1,-2,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T1[:]:=T1[-1,-2,1]*U_ss_ss[1,-3,-4];
    elseif Rank(T1)==3+1
        @tensor T1_closed[:]:=T1[-1,-2,-3,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T1[:]:=T1[-1,-2,-3,1]*U_ss_ss[1,-4,-5];
    end

    T2=mps_down[x_range[2]];
    if Rank(T2)==2+1
        @tensor T2_closed[:]:=T2[-1,-2,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T2[:]:=T2[-1,-2,1]*U_ss_ss[1,-3,-4];
    elseif Rank(T2)==3+1
        @tensor T2_closed[:]:=T2[-1,-2,-3,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T2[:]:=T2[-1,-2,-3,1]*U_ss_ss[1,-4,-5];
    end

    u,s,v=tsvd(h_plaquatte,(1,4,),(2,3,));
    v=s*v;
    U_tem=unitary(fuse(space(v,1)*space(T2,1)), space(v,1)*space(T2,1));

    if Rank(T1)==2+2
        @tensor T1[:]:=T1[3,-2,1,2]*u[1,2,4]*U_tem'[4,3,-1];
    elseif Rank(T1)==3+2
        @tensor T1[:]:=T1[-1,3,-3,1,2]*u[1,2,4]*U_tem'[4,3,-2];
    end

    if Rank(T2)==2+2
        @tensor T2[:]:=T2[4,-2,1,2]*v[3,1,2]*U_tem[-1,3,4];
    elseif Rank(T2)==3+2
        @tensor T2[:]:=T2[4,-2,-3,1,2]*v[3,1,2]*U_tem[-1,3,4];
    end


    ########################################################
    # #verification energy
    # mps_down1=deepcopy(mps_down);
    # mps_down2=deepcopy(mps_down);
    # mps_down1[x_range[1]]=T1;
    # mps_down1[x_range[2]]=T2;
    # mps_down2[x_range[1]]=T1_closed;
    # mps_down2[x_range[2]]=T2_closed;
    # ee=norm_1D_no_conjugate(mps_up,mps_down1);
    # Norm=norm_1D_no_conjugate(mps_up,mps_down2);
    # println(ee/Norm)
    #######################################################

    mps_down[x_range[1]]=T1;
    mps_down[x_range[2]]=T2;
    term=derivative_3row_top(mps_up,mpo,mps_down,px);

    return term
end

function contract_3Row_site_plaquatte_plaquatte(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap) #first row has empty site, second row and third row has plaquatte
    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy)';

    u,s,v=tsvd(h_plaquatte,(1,2,),(4,3,));#1,2,virtual1
    v=s*v;
    v=permute(v,(2,3,1,));#4,3,virtual1

    u1,s1,v1=tsvd(u,(1,),(2,3,));
    v1=s1*v1;
    u2,s2,v2=tsvd(v,(1,),(2,3,));
    v2=s2*v2;

    T1up=mpo[x_range[1]];
    T2up=mpo[x_range[1]+1];
    T1down=mps_down[x_range[1]];
    T2down=mps_down[x_range[1]+1];

    if x_range[1]==1
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,1)), space(u2,2)*space(T1down,1));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[3,-2,1]*u2[1,2]*U2[-1,2,3];
        @tensor T2up[:]:=T2up[3,5,-3,-4,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-2,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];

    elseif x_range[1]+1==Lx
        U1=unitary(fuse(space(u1,2)*space(T1up,3)), space(u1,2)*space(T1up,3));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,-2,3,-4,1]*u1[1,2]*U1[-3,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,5,-3,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-2,4,5];
        @tensor T2down[:]:=T2down[3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
    elseif 1<x_range[1]<Lx-1
        U1=unitary(fuse(space(u1,2)*space(T1up,3)), space(u1,2)*space(T1up,3));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,-2,3,-4,1]*u1[1,2]*U1[-3,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,5,-3,-4,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-2,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
        
    end

    #################################
    ##verification energy
    # mpo1=deepcopy(mpo);
    # mpo2=deepcopy(mpo);
    # mps_down1=deepcopy(mps_down);
    # mps_down2=deepcopy(mps_down);
    # mpo1[x_range[1]]=T1up;
    # mpo1[x_range[2]]=T2up;
    # mpo2[x_range[1]]=T1up_closed;
    # mpo2[x_range[2]]=T2up_closed;
    # mps_down1[x_range[1]]=T1down;
    # mps_down1[x_range[2]]=T2down;
    # mps_down2[x_range[1]]=T1down_closed;
    # mps_down2[x_range[2]]=T2down_closed;
    # ee=norm_1D_no_conjugate(mps_up,mpo1,mps_down1);
    # Norm=norm_1D_no_conjugate(mps_up,mpo2,mps_down2);
    # println(ee/Norm)
    #################################

    mpo[x_range[1]]=T1up;
    mpo[x_range[1]+1]=T2up;
    mps_down[x_range[1]]=T1down;
    mps_down[x_range[1]+1]=T2down;

    term=derivative_3row_top(mps_up,mpo,mps_down,px);

    return term
end

function contract_3Row_plaquatte_plaquatte_site(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap) #first row has empty site, second row and third row has plaquatte
    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy)';

    u,s,v=tsvd(h_plaquatte,(1,2,),(4,3,));#1,2,virtual1
    v=s*v;
    v=permute(v,(2,3,1,));#4,3,virtual1

    u1,s1,v1=tsvd(u,(1,),(2,3,));
    v1=s1*v1;
    u2,s2,v2=tsvd(v,(1,),(2,3,));
    v2=s2*v2;

    T1up=mps_up[x_range[1]];
    T2up=mps_up[x_range[1]+1];
    T1down=mpo[x_range[1]];
    T2down=mpo[x_range[1]+1];

    if x_range[1]==1
        U1=unitary(fuse(space(u1,2)*space(T1up,1)), space(u1,2)*space(T1up,1));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,3)), space(v1,3)*space(T2up,3));
        @tensor T1up_closed[:]:=T1up[-1,-2,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,-4,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[3,-2,1]*u1[1,2]*U1[-1,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,-2,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,-3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-4];

    elseif x_range[1]+1==Lx
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,3)), space(u2,2)*space(T1down,3));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[-1,-2,3,-4,1]*u2[1,2]*U2[-3,2,3];
        @tensor T2up[:]:=T2up[3,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
    elseif 1<x_range[1]<Lx-1
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,3)), space(u2,2)*space(T1down,3));
        U12=unitary(fuse(space(v1,3)*space(T2up,3)), space(v1,3)*space(T2up,3));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,-4,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[-1,-2,3,-4,1]*u2[1,2]*U2[-3,2,3];
        @tensor T2up[:]:=T2up[3,-2,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,-3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-4];

        
    end

    #################################
    # #verification energy
    # mps_up1=deepcopy(mps_up);
    # mps_up2=deepcopy(mps_up);
    # mpo1=deepcopy(mpo);
    # mpo2=deepcopy(mpo);
    # mps_up1[x_range[1]]=T1up;
    # mps_up1[x_range[2]]=T2up;
    # mps_up2[x_range[1]]=T1up_closed;
    # mps_up2[x_range[2]]=T2up_closed;
    # mpo1[x_range[1]]=T1down;
    # mpo1[x_range[2]]=T2down;
    # mpo2[x_range[1]]=T1down_closed;
    # mpo2[x_range[2]]=T2down_closed;
    # ee=norm_1D_no_conjugate(mps_up1,mpo1,mps_down);
    # Norm=norm_1D_no_conjugate(mps_up2,mpo2,mps_down);
    # println(ee/Norm)
    #################################

    mps_up[x_range[1]]=T1up;
    mps_up[x_range[1]+1]=T2up;
    mpo[x_range[1]]=T1down;
    mpo[x_range[1]+1]=T2down;

    term=derivative_3row_bot(mps_up,mpo,mps_down,px);

    return term
end


function contract_3Row_plaquatte_site_mps(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap) #first row has plaquatte, second row has empty site, third row normal mps
    L=length(mps_up);

    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy);
    U_ss_ss=unitary(fuse(space(U_ss,1)*space(U_ss,1)), space(U_ss,1)*space(U_ss,1));
    U_ss_ss=permute(U_ss_ss',(3,1,2,));

    T1=mps_up[x_range[1]];
    if Rank(T1)==2+1
        @tensor T1_closed[:]:=T1[-1,-2,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T1[:]:=T1[-1,-2,1]*U_ss_ss[1,-4,-3];
    elseif Rank(T1)==3+1
        @tensor T1_closed[:]:=T1[-1,-2,-3,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T1[:]:=T1[-1,-2,-3,1]*U_ss_ss[1,-5,-4];
    end

    T2=mps_up[x_range[2]];
    if Rank(T2)==2+1
        @tensor T2_closed[:]:=T2[-1,-2,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T2[:]:=T2[-1,-2,1]*U_ss_ss[1,-4,-3];
    elseif Rank(T2)==3+1
        @tensor T2_closed[:]:=T2[-1,-2,-3,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T2[:]:=T2[-1,-2,-3,1]*U_ss_ss[1,-5,-4];
    end

    u,s,v=tsvd(h_plaquatte,(1,4,),(2,3,));
    v=s*v;
    U_tem=unitary(fuse(space(v,1)*space(T2,1)), space(v,1)*space(T2,1));

    if Rank(T1)==2+2
        @tensor T1[:]:=T1[3,-2,1,2]*u[1,2,4]*U_tem'[4,3,-1];
    elseif Rank(T1)==3+2
        @tensor T1[:]:=T1[-1,3,-3,1,2]*u[1,2,4]*U_tem'[4,3,-2];
    end

    if Rank(T2)==2+2
        @tensor T2[:]:=T2[4,-2,1,2]*v[3,1,2]*U_tem[-1,3,4];
    elseif Rank(T2)==3+2
        @tensor T2[:]:=T2[4,-2,-3,1,2]*v[3,1,2]*U_tem[-1,3,4];
    end


    ########################################################
    # #verification energy
    # mps_up1=deepcopy(mps_up);
    # mps_up2=deepcopy(mps_up);
    # mps_up1[x_range[1]]=T1;
    # mps_up1[x_range[2]]=T2;
    # mps_up2[x_range[1]]=T1_closed;
    # mps_up2[x_range[2]]=T2_closed;
    # ee=norm_1D_no_conjugate(mps_up1,mpo,mps_down);
    # Norm=norm_1D_no_conjugate(mps_up2,mpo,mps_down);
    # println(ee/Norm)
    #######################################################

    mps_up[x_range[1]]=T1;
    mps_up[x_range[2]]=T2;
    term=derivative_3row_middle(mps_up,mpo,mps_down,px);
    return term
end

function contract_3Row_mps_site_plaquatte(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap) #first row normal mps, second row empty site, third row has plaquatte
    L=length(mps_up);

    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy);
    U_ss_ss=unitary(fuse(space(U_ss,1)*space(U_ss,1)), space(U_ss,1)*space(U_ss,1));
    U_ss_ss=permute(U_ss_ss',(3,1,2,));

    T1=mps_down[x_range[1]];
    if Rank(T1)==2+1
        @tensor T1_closed[:]:=T1[-1,-2,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T1[:]:=T1[-1,-2,1]*U_ss_ss[1,-3,-4];
    elseif Rank(T1)==3+1
        @tensor T1_closed[:]:=T1[-1,-2,-3,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T1[:]:=T1[-1,-2,-3,1]*U_ss_ss[1,-4,-5];
    end

    T2=mps_down[x_range[2]];
    if Rank(T2)==2+1
        @tensor T2_closed[:]:=T2[-1,-2,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T2[:]:=T2[-1,-2,1]*U_ss_ss[1,-3,-4];
    elseif Rank(T2)==3+1
        @tensor T2_closed[:]:=T2[-1,-2,-3,1]*U_ss_ss[1,2,3]*U_ss'[4,4,2]*U_ss'[5,5,3];
        @tensor T2[:]:=T2[-1,-2,-3,1]*U_ss_ss[1,-4,-5];
    end

    u,s,v=tsvd(h_plaquatte,(1,4,),(2,3,));
    v=s*v;
    U_tem=unitary(fuse(space(v,1)*space(T2,1)), space(v,1)*space(T2,1));

    if Rank(T1)==2+2
        @tensor T1[:]:=T1[3,-2,1,2]*u[1,2,4]*U_tem'[4,3,-1];
    elseif Rank(T1)==3+2
        @tensor T1[:]:=T1[-1,3,-3,1,2]*u[1,2,4]*U_tem'[4,3,-2];
    end

    if Rank(T2)==2+2
        @tensor T2[:]:=T2[4,-2,1,2]*v[3,1,2]*U_tem[-1,3,4];
    elseif Rank(T2)==3+2
        @tensor T2[:]:=T2[4,-2,-3,1,2]*v[3,1,2]*U_tem[-1,3,4];
    end


    ########################################################
    # #verification energy
    # mps_down1=deepcopy(mps_down);
    # mps_down2=deepcopy(mps_down);
    # mps_down1[x_range[1]]=T1;
    # mps_down1[x_range[2]]=T2;
    # mps_down2[x_range[1]]=T1_closed;
    # mps_down2[x_range[2]]=T2_closed;
    # ee=norm_1D_no_conjugate(mps_up,mpo,mps_down1);
    # Norm=norm_1D_no_conjugate(mps_up,mpo,mps_down2);
    # println(ee/Norm)
    #######################################################

    mps_down[x_range[1]]=T1;
    mps_down[x_range[2]]=T2;
    term=derivative_3row_middle(mps_up,mpo,mps_down,px);
    return term
end


function contract_2row_LR_plaquatte(mps_up,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap,site_pos)
    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy)';

    u,s,v=tsvd(h_plaquatte,(1,2,),(4,3,));#1,2,virtual1
    v=s*v;
    v=permute(v,(2,3,1,));#4,3,virtual1

    u1,s1,v1=tsvd(u,(1,),(2,3,));
    v1=s1*v1;
    u2,s2,v2=tsvd(v,(1,),(2,3,));
    v2=s2*v2;

    T1up=mps_up[x_range[1]];
    T2up=mps_up[x_range[1]+1];
    T1down=mps_down[x_range[1]];
    T2down=mps_down[x_range[1]+1];

    if x_range[1]==1
        U1=unitary(fuse(space(u1,2)*space(T1up,1)), space(u1,2)*space(T1up,1));
        U2=unitary(fuse(space(u2,2)*space(T1down,1)), space(u2,2)*space(T1down,1));
        U12=unitary(fuse(space(v1,3)*space(T2up,3)), space(v1,3)*space(T2up,3));
        @tensor T1up_closed[:]:=T1up[-1,-2,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[3,-2,1]*u1[1,2]*U1[-1,2,3];
        @tensor T1down[:]:=T1down[3,-2,1]*u2[1,2]*U2[-1,2,3];
        @tensor T2up[:]:=T2up[3,-2,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];

    elseif x_range[1]+1==Lx
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
    elseif 1<x_range[1]<Lx-1
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,3)), space(v1,3)*space(T2up,3));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,-2,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];

        
    end

    #################################
    # #verification energy
    # mps_up1=deepcopy(mps_up);
    # mps_up2=deepcopy(mps_up);
    # mps_down1=deepcopy(mps_down);
    # mps_down2=deepcopy(mps_down);
    # mps_up1[x_range[1]]=T1up;
    # mps_up1[x_range[2]]=T2up;
    # mps_up2[x_range[1]]=T1up_closed;
    # mps_up2[x_range[2]]=T2up_closed;
    # mps_down1[x_range[1]]=T1down;
    # mps_down1[x_range[2]]=T2down;
    # mps_down2[x_range[1]]=T1down_closed;
    # mps_down2[x_range[2]]=T2down_closed;
    # ee=norm_1D_no_conjugate(mps_up1,mps_down1);
    # Norm=norm_1D_no_conjugate(mps_up2,mps_down2);
    # println(ee/Norm)
    #################################

    mps_up[x_range[1]]=T1up;
    mps_up[x_range[1]+1]=T2up;
    mps_down[x_range[1]]=T1down;
    mps_down[x_range[1]+1]=T2down;

    if site_pos=="up"
        term=derivative_2row_top(mps_up,mps_down,px);
    elseif site_pos=="down"
        term=derivative_2row_bot(mps_up,mps_down,px);
    end
    return term
end


function contract_3row_LR_plaquatte_12(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap)#mps_up thick
    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy)';

    u,s,v=tsvd(h_plaquatte,(1,2,),(4,3,));#1,2,virtual1
    v=s*v;
    v=permute(v,(2,3,1,));#4,3,virtual1

    u1,s1,v1=tsvd(u,(1,),(2,3,));
    v1=s1*v1;
    u2,s2,v2=tsvd(v,(1,),(2,3,));
    v2=s2*v2;

    T1up=mps_up[x_range[1]];
    T2up=mps_up[x_range[1]+1];
    T1down=mpo[x_range[1]];
    T2down=mpo[x_range[1]+1];

    if x_range[1]==1
        U1=unitary(fuse(space(u1,2)*space(T1up,1)), space(u1,2)*space(T1up,1));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,3)), space(v1,3)*space(T2up,3));
        @tensor T1up_closed[:]:=T1up[-1,-2,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,-4,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[3,-2,1]*u1[1,2]*U1[-1,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,-2,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,-3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-4];

    elseif x_range[1]+1==Lx
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,3)), space(u2,2)*space(T1down,3));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[-1,-2,3,-4,1]*u2[1,2]*U2[-3,2,3];
        @tensor T2up[:]:=T2up[3,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
    elseif 1<x_range[1]<Lx-1
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,3)), space(u2,2)*space(T1down,3));
        U12=unitary(fuse(space(v1,3)*space(T2up,3)), space(v1,3)*space(T2up,3));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,-4,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[-1,-2,3,-4,1]*u2[1,2]*U2[-3,2,3];
        @tensor T2up[:]:=T2up[3,-2,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,-3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-4];

        
    end

    #################################
    # #verification energy
    # mps_up1=deepcopy(mps_up);
    # mps_up2=deepcopy(mps_up);
    # mpo1=deepcopy(mpo);
    # mpo2=deepcopy(mpo);
    # mps_up1[x_range[1]]=T1up;
    # mps_up1[x_range[2]]=T2up;
    # mps_up2[x_range[1]]=T1up_closed;
    # mps_up2[x_range[2]]=T2up_closed;
    # mpo1[x_range[1]]=T1down;
    # mpo1[x_range[2]]=T2down;
    # mpo2[x_range[1]]=T1down_closed;
    # mpo2[x_range[2]]=T2down_closed;
    # ee=norm_1D_no_conjugate(mps_up1,mpo1,mps_down);
    # Norm=norm_1D_no_conjugate(mps_up2,mpo2,mps_down);
    # println(ee/Norm)
    #################################

    mps_up[x_range[1]]=T1up;
    mps_up[x_range[1]+1]=T2up;
    mpo[x_range[1]]=T1down;
    mpo[x_range[1]+1]=T2down;

    term=derivative_3row_middle(mps_up,mpo,mps_down,px);

    return term
end

function contract_3row_LR_plaquatte_23(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap)#mps_down thick
    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy)';

    u,s,v=tsvd(h_plaquatte,(1,2,),(4,3,));#1,2,virtual1
    v=s*v;
    v=permute(v,(2,3,1,));#4,3,virtual1

    u1,s1,v1=tsvd(u,(1,),(2,3,));
    v1=s1*v1;
    u2,s2,v2=tsvd(v,(1,),(2,3,));
    v2=s2*v2;

    T1up=mpo[x_range[1]];
    T2up=mpo[x_range[1]+1];
    T1down=mps_down[x_range[1]];
    T2down=mps_down[x_range[1]+1];

    if x_range[1]==1
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,1)), space(u2,2)*space(T1down,1));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[3,-2,1]*u2[1,2]*U2[-1,2,3];
        @tensor T2up[:]:=T2up[3,5,-3,-4,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-2,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];

    elseif x_range[1]+1==Lx
        U1=unitary(fuse(space(u1,2)*space(T1up,3)), space(u1,2)*space(T1up,3));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,-2,3,-4,1]*u1[1,2]*U1[-3,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,5,-3,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-2,4,5];
        @tensor T2down[:]:=T2down[3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
    elseif 1<x_range[1]<Lx-1
        U1=unitary(fuse(space(u1,2)*space(T1up,3)), space(u1,2)*space(T1up,3));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,-2,3,-4,1]*u1[1,2]*U1[-3,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,5,-3,-4,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-2,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
        
    end

    #################################
    # #verification energy
    # mpo1=deepcopy(mpo);
    # mpo2=deepcopy(mpo);
    # mps_down1=deepcopy(mps_down);
    # mps_down2=deepcopy(mps_down);
    # mpo1[x_range[1]]=T1up;
    # mpo1[x_range[2]]=T2up;
    # mpo2[x_range[1]]=T1up_closed;
    # mpo2[x_range[2]]=T2up_closed;
    # mps_down1[x_range[1]]=T1down;
    # mps_down1[x_range[2]]=T2down;
    # mps_down2[x_range[1]]=T1down_closed;
    # mps_down2[x_range[2]]=T2down_closed;
    # ee=norm_1D_no_conjugate(mps_up,mpo1,mps_down1);
    # Norm=norm_1D_no_conjugate(mps_up,mpo2,mps_down2);
    # println(ee/Norm)
    #################################

    mpo[x_range[1]]=T1up;
    mpo[x_range[1]+1]=T2up;
    mps_down[x_range[1]]=T1down;
    mps_down[x_range[1]+1]=T2down;

    term=derivative_3row_middle(mps_up,mpo,mps_down,px);

    return term
end


function contract_3row_LR_plaquatte_23_site_bot(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap)#mps_down thick
    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy)';

    u,s,v=tsvd(h_plaquatte,(1,2,),(4,3,));#1,2,virtual1
    v=s*v;
    v=permute(v,(2,3,1,));#4,3,virtual1

    u1,s1,v1=tsvd(u,(1,),(2,3,));
    v1=s1*v1;
    u2,s2,v2=tsvd(v,(1,),(2,3,));
    v2=s2*v2;

    T1up=mpo[x_range[1]];
    T2up=mpo[x_range[1]+1];
    T1down=mps_down[x_range[1]];
    T2down=mps_down[x_range[1]+1];

    if x_range[1]==1
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,1)), space(u2,2)*space(T1down,1));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[3,-2,1]*u2[1,2]*U2[-1,2,3];
        @tensor T2up[:]:=T2up[3,5,-3,-4,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-2,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];

    elseif x_range[1]+1==Lx
        U1=unitary(fuse(space(u1,2)*space(T1up,3)), space(u1,2)*space(T1up,3));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,-2,3,-4,1]*u1[1,2]*U1[-3,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,5,-3,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-2,4,5];
        @tensor T2down[:]:=T2down[3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
    elseif 1<x_range[1]<Lx-1
        U1=unitary(fuse(space(u1,2)*space(T1up,3)), space(u1,2)*space(T1up,3));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,-2,3,-4,1]*u1[1,2]*U1[-3,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,5,-3,-4,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-2,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
        
    end

    #################################
    # #verification energy
    # mpo1=deepcopy(mpo);
    # mpo2=deepcopy(mpo);
    # mps_down1=deepcopy(mps_down);
    # mps_down2=deepcopy(mps_down);
    # mpo1[x_range[1]]=T1up;
    # mpo1[x_range[2]]=T2up;
    # mpo2[x_range[1]]=T1up_closed;
    # mpo2[x_range[2]]=T2up_closed;
    # mps_down1[x_range[1]]=T1down;
    # mps_down1[x_range[2]]=T2down;
    # mps_down2[x_range[1]]=T1down_closed;
    # mps_down2[x_range[2]]=T2down_closed;
    # ee=norm_1D_no_conjugate(mps_up,mpo1,mps_down1);
    # Norm=norm_1D_no_conjugate(mps_up,mpo2,mps_down2);
    # println(ee/Norm)
    #################################

    mpo[x_range[1]]=T1up;
    mpo[x_range[1]+1]=T2up;
    mps_down[x_range[1]]=T1down;
    mps_down[x_range[1]+1]=T2down;

    term=derivative_3row_bot(mps_up,mpo,mps_down,px);

    return term
end

function contract_3row_LR_plaquatte_12_site_top(mps_up,mpo,mps_down,x_range::Vector,px::Number,h_plaquatte::TensorMap)#mps_up thick
    global U_phy
    U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy)';

    u,s,v=tsvd(h_plaquatte,(1,2,),(4,3,));#1,2,virtual1
    v=s*v;
    v=permute(v,(2,3,1,));#4,3,virtual1

    u1,s1,v1=tsvd(u,(1,),(2,3,));
    v1=s1*v1;
    u2,s2,v2=tsvd(v,(1,),(2,3,));
    v2=s2*v2;

    T1up=mps_up[x_range[1]];
    T2up=mps_up[x_range[1]+1];
    T1down=mpo[x_range[1]];
    T2down=mpo[x_range[1]+1];

    if x_range[1]==1
        U1=unitary(fuse(space(u1,2)*space(T1up,1)), space(u1,2)*space(T1up,1));
        U2=unitary(fuse(space(u2,2)*space(T1down,2)), space(u2,2)*space(T1down,2));
        U12=unitary(fuse(space(v1,3)*space(T2up,3)), space(v1,3)*space(T2up,3));
        @tensor T1up_closed[:]:=T1up[-1,-2,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,-4,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[3,-2,1]*u1[1,2]*U1[-1,2,3];
        @tensor T1down[:]:=T1down[-1,3,-3,1]*u2[1,2]*U2[-2,2,3];
        @tensor T2up[:]:=T2up[3,-2,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,-3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-4];

    elseif x_range[1]+1==Lx
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,3)), space(u2,2)*space(T1down,3));
        U12=unitary(fuse(space(v1,3)*space(T2up,2)), space(v1,3)*space(T2up,2));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[-1,-2,3,-4,1]*u2[1,2]*U2[-3,2,3];
        @tensor T2up[:]:=T2up[3,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-3];
    elseif 1<x_range[1]<Lx-1
        U1=unitary(fuse(space(u1,2)*space(T1up,2)), space(u1,2)*space(T1up,2));
        U2=unitary(fuse(space(u2,2)*space(T1down,3)), space(u2,2)*space(T1down,3));
        U12=unitary(fuse(space(v1,3)*space(T2up,3)), space(v1,3)*space(T2up,3));
        @tensor T1up_closed[:]:=T1up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T1down_closed[:]:=T1down[-1,-2,-3,-4,1]*U_ss[2,2,1];
        @tensor T2up_closed[:]:=T2up[-1,-2,-3,1]*U_ss[2,2,1];
        @tensor T2down_closed[:]:=T2down[-1,-2,-3,-4,1]*U_ss[2,2,1];

        @tensor T1up[:]:=T1up[-1,3,-3,1]*u1[1,2]*U1[-2,2,3];
        @tensor T1down[:]:=T1down[-1,-2,3,-4,1]*u2[1,2]*U2[-3,2,3];
        @tensor T2up[:]:=T2up[3,-2,5,1]*v1[2,1,4]*U1'[2,3,-1]*U12[-3,4,5];
        @tensor T2down[:]:=T2down[3,-2,-3,5,1]*v2[2,1,4]*U2'[2,3,-1]*U12'[4,5,-4];

        
    end

    #################################
    # #verification energy
    # mps_up1=deepcopy(mps_up);
    # mps_up2=deepcopy(mps_up);
    # mpo1=deepcopy(mpo);
    # mpo2=deepcopy(mpo);
    # mps_up1[x_range[1]]=T1up;
    # mps_up1[x_range[2]]=T2up;
    # mps_up2[x_range[1]]=T1up_closed;
    # mps_up2[x_range[2]]=T2up_closed;
    # mpo1[x_range[1]]=T1down;
    # mpo1[x_range[2]]=T2down;
    # mpo2[x_range[1]]=T1down_closed;
    # mpo2[x_range[2]]=T2down_closed;
    # ee=norm_1D_no_conjugate(mps_up1,mpo1,mps_down);
    # Norm=norm_1D_no_conjugate(mps_up2,mpo2,mps_down);
    # println(ee/Norm)
    #################################

    mps_up[x_range[1]]=T1up;
    mps_up[x_range[1]+1]=T2up;
    mpo[x_range[1]]=T1down;
    mpo[x_range[1]+1]=T2down;

    term=derivative_3row_top(mps_up,mpo,mps_down,px);

    return term
end

function envL_2row(mps_up,mps_down,pos)
    cx=1
    @tensor env[:]:=mps_up[cx][-1,1]*mps_down[cx][-2,1];
    for cx=2:pos
        @tensor env[:]:=env[1,3]*mps_up[cx][1,-1,2]*mps_down[cx][3,-2,2];
    end
    return env
end

function envR_2row(mps_up,mps_down,pos)
    Lx=length(mps_up);
    cx=Lx;
    @tensor env[:]:=mps_up[Lx][-1,1]*mps_down[Lx][-2,1];
    for cc=Lx-1:-1:pos
        @tensor env[:]:=mps_up[cc][-1,1,2]*mps_down[cc][-2,3,2]*env[1,3];
    end
    return env
end

function envL_3row(mps_up,mpo,mps_down,pos)
    cx=1
    @tensor env[:]:=mps_up[cx][-1,1]*mpo[cx][2,-2,1]*mps_down[cx][-3,2];
    for cx=2:pos
        @tensor env[:]:=env[1,3,5]*mps_up[cx][1,-1,2]*mpo[cx][3,4,-2,2]*mps_down[cx][5,-3,4];
    end
    return env
end

function envR_3row(mps_up,mpo,mps_down,pos)
    Lx=length(mps_up);
    cx=Lx;
    @tensor env[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
    for cc=Lx-1:-1:pos
        @tensor env[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*env[1,3,5];
    end
    return env
end