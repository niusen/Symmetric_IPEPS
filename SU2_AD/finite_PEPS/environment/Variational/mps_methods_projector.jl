

function reconstruct_boundary_mps_down_move(psi_double,norm_coe,unitarys_R_set,unitarys_L_set, UR_set,UL_set,projectors_R_set,projectors_L_set,cy_end=3, truncate_special_bond=true)
    Lx=size(psi_double,1);
    Ly=size(psi_double,2);
    mps_set=Matrix{TensorMap}(undef,Lx,Ly);

    cy=Ly;
    mps_trun=pi_rotate_mps_down_move(psi_double[:,cy]);
    mps_set[:,cy]=pi_inverse_rotate_mps_down_move(mps_trun);

    for cy=Ly-1:-1:cy_end
        mpo=pi_rotate_mpo_down_move(psi_double[:,cy]);
        if cy>cy_end
            mps_trun=mpo_mps_projector(mpo,mps_trun,unitarys_R_set[cy],unitarys_L_set[cy], UR_set[cy],UL_set[cy],projectors_R_set[cy],projectors_L_set[cy],true);
        elseif cy==cy_end
            mps_trun=mpo_mps_projector(mpo,mps_trun,unitarys_R_set[cy],unitarys_L_set[cy], UR_set[cy],UL_set[cy],projectors_R_set[cy],projectors_L_set[cy],truncate_special_bond);
        end
        mps_trun[end]=mps_trun[end]/norm_coe[cy];
        mps_set[:,cy]=pi_inverse_rotate_mps_down_move(mps_trun);
    end
    return mps_set
end


function reconstruct_boundary_mps_up_move(psi_double,norm_coe,unitarys_R_set,unitarys_L_set, UR_set,UL_set,projectors_R_set,projectors_L_set,cy_end=100, truncate_special_bond=true)
    Lx=size(psi_double,1);
    Ly=size(psi_double,2);
    mps_set=Matrix{TensorMap}(undef,Lx,Ly);

    cy=1;
    mps_trun=psi_double[:,cy];
    mps_set[:,cy]=mps_trun;

    for cy=2:min(Ly-2,cy_end)
        if cy<min(Ly-2,cy_end)
            mps_trun=mpo_mps_projector(psi_double[:,cy],mps_trun,unitarys_R_set[cy],unitarys_L_set[cy], UR_set[cy],UL_set[cy],projectors_R_set[cy],projectors_L_set[cy],true);
        elseif cy==min(Ly-2,cy_end)
            mps_trun=mpo_mps_projector(psi_double[:,cy],mps_trun,unitarys_R_set[cy],unitarys_L_set[cy], UR_set[cy],UL_set[cy],projectors_R_set[cy],projectors_L_set[cy],truncate_special_bond);
        end
        mps_trun[end]=mps_trun[end]/norm_coe[cy];
        mps_set[:,cy]=mps_trun;
    end
    return mps_set
end



function mpo_mps_no_projector(mpo,mps_old, UR,UL)
    L=length(mpo);
    mps_new=Vector{TensorMap}(undef,L);

    mpo_rank=Rank.(mpo);
    mpo_rank[1]+=1;
    mpo_rank[end]+=1;

    mps_rank=Rank.(mps_old);
    mps_rank[1]+=1;
    mps_rank[end]+=1;

    pos1=findall(x->x==4,mps_rank);
    pos2=findall(x->x==5,mpo_rank);
    pos_op=sort(unique(vcat(pos1,pos2)));
    

    function apply_left(M,A,ur)
        if (Rank(A)==2)&(Rank(M)==3)#without open physical leg
            @tensor A_trun[:]:= M[4,3,-2]*A[2,4]*ur[-1,3,2];
        elseif (Rank(A)==2+1)&(Rank(M)==3)
            v1=space(M,2);
            v2=space(A,1);
            ur=unitary(fuse(v1*v2),v1*v2);
            @tensor A_trun[:]:= M[4,3,-2]*A[2,4,-3]*ur[-1,3,2];
        elseif (Rank(A)==2+2)&(Rank(M)==3)
            v1=space(M,2);
            v2=space(A,1);
            ur=unitary(fuse(v1*v2),v1*v2);
            @tensor A_trun[:]:= M[4,3,-2]*A[2,4,-3,-4]*ur[-1,3,2];
        elseif (Rank(A)==2)&(Rank(M)==3+1)
            v1=space(M,2);
            v2=space(A,1);
            ur=unitary(fuse(v1*v2),v1*v2);
            @tensor A_trun[:]:= M[4,3,-2,-3]*A[2,4]*ur[-1,3,2];
        elseif (Rank(A)==2+1)&(Rank(M)==3+1)
            v1=space(M,2);
            v2=space(A,1);
            ur=unitary(fuse(v1*v2),v1*v2);
            @tensor A_trun[:]:= M[4,3,-2,-3]*A[2,4,-4]*ur[-1,3,2];

            #combine two physical legs
            U_tem=unitary(fuse(space(A_trun,3)*space(A_trun,4)), space(A_trun,3)*space(A_trun,4));
            @tensor A_trun[:]:= A_trun[-1,-2,1,2]*U_tem[-3,1,2];
        end
        return A_trun
    end

    function apply_middle(M,A,ur,ul,pos)
        if (Rank(A)==3)&(Rank(M)==4)#without open physical leg
            @tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5]*ul[4,3,-1]*ur[-2,6,7];
        elseif (Rank(A)==3+1)&(Rank(M)==4)
            if pos==pos_op[1]
                #ur=UR;
                v1=space(M,3);
                v2=space(A,2);
                ur=unitary(fuse(v1*v2),v1*v2);
                @tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5,-4]*ul[4,3,-1]*ur[-2,6,7];
            elseif pos==pos_op[2]
                v1=space(M,1);
                v2=space(A,1);
                ul=unitary(fuse(v1*v2),v1'*v2')';
                @tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5,-4]*ul[4,3,-1]*ur[-2,6,7];
            end
        elseif (Rank(A)==3+2)&(Rank(M)==4)
            @tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5,-4,-5]*ul[4,3,-1]*ur[-2,6,7];
        elseif (Rank(A)==3)&(Rank(M)==4+1)
            if pos==pos_op[1]
                #ur=UR;
                v1=space(M,3);
                v2=space(A,2);
                ur=unitary(fuse(v1*v2),v1*v2);
                @tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5]*ul[4,3,-1]*ur[-2,6,7];
            elseif pos==pos_op[2]
                v1=space(M,1);
                v2=space(A,1);
                ul=unitary(fuse(v1*v2),v1'*v2')';
                @tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5]*ul[4,3,-1]*ur[-2,6,7];
            end
        elseif (Rank(A)==3+1)&(Rank(M)==4+1)
            if pos==pos_op[1]
                v1=space(M,3);
                v2=space(A,2);
                ur=unitary(fuse(v1*v2),v1*v2);
                @tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5,-5]*ul[4,3,-1]*ur[-2,6,7];
            elseif pos==pos_op[2]
                v1=space(M,1);
                v2=space(A,1);
                ul=unitary(fuse(v1*v2),v1'*v2')';
                @tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5,-5]*ul[4,3,-1]*ur[-2,6,7];
            end

            #combine two physical legs
            U_tem=unitary(fuse(space(A_trun,4)*space(A_trun,5)), space(A_trun,4)*space(A_trun,5));
            @tensor A_trun[:]:= A_trun[-1,-2,-3,1,2]*U_tem[-4,1,2];
        end
        return A_trun
    end

    function apply_right(M,A,ul)
        if (Rank(A)==2)&(Rank(M)==3)#without open physical leg
            @tensor A_trun[:]:= M[3,4,-2]*A[2,4]*ul[3,2,-1];
        elseif (Rank(A)==2+1)&(Rank(M)==3)
            v1=space(M,1);
            v2=space(A,1);
            ul=unitary(fuse(v1*v2),v1'*v2')';
            @tensor A_trun[:]:= M[3,4,-2]*A[2,4,-3]*ul[3,2,-1];
        elseif (Rank(A)==2+2)&(Rank(M)==3)
            v1=space(M,1);
            v2=space(A,1);
            ul=unitary(fuse(v1*v2),v1'*v2')';
            @tensor A_trun[:]:= M[3,4,-2]*A[2,4,-3,-4]*ul[3,2,-1];
        elseif (Rank(A)==2)&(Rank(M)==3+1)
            v1=space(M,1);
            v2=space(A,1);
            ul=unitary(fuse(v1*v2),v1'*v2')';
            @tensor A_trun[:]:= M[3,4,-2,-3]*A[2,4]*ul[3,2,-1];
        elseif (Rank(A)==2+1)&(Rank(M)==3+1)
            v1=space(M,1);
            v2=space(A,1);
            ul=unitary(fuse(v1*v2),v1'*v2')';
            @tensor A_trun[:]:= M[3,4,-2,-3]*A[2,4,-4]*ul[3,2,-1];

            #combine two physical legs
            U_tem=unitary(fuse(space(A_trun,3)*space(A_trun,4)), space(A_trun,3)*space(A_trun,4));
            @tensor A_trun[:]:= A_trun[-1,-2,1,2]*U_tem[-3,1,2];
        end
        return A_trun
    end

    cx=1;
    mps_new[cx]=apply_left(mpo[cx],mps_old[cx],UR[cx]);

    for cx=2:L-1
        mps_new[cx]=apply_middle(mpo[cx],mps_old[cx], UR[cx], UL[cx],cx);
    end

    cx=L;
    mps_new[cx]=apply_right(mpo[cx],mps_old[cx],UL[cx]);

    return mps_new 
end

function mpo_mps_projector(mpo,mps_old,unitarys_R,unitarys_L, UR,UL,projectors_R,projectors_L,truncate_special_bond=true)
    L=length(mpo);
    mps_new=Vector{TensorMap}(undef,L);

    mpo_rank=Rank.(mpo);
    mpo_rank[1]+=1;
    mpo_rank[end]+=1;

    mps_rank=Rank.(mps_old);
    mps_rank[1]+=1;
    mps_rank[end]+=1;

    pos1=findall(x->x==4,mps_rank);
    pos2=findall(x->x==5,mpo_rank);
    pos_op=sort(unique(vcat(pos1,pos2)));
    

    function truncate_left(M,A,unitarys_R,UR,projector_r,pos)
        if (Rank(A)==2)&(Rank(M)==3)#without open physical leg
            @tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
            @tensor A_trun[:]:= M[4,3,-2]*A[2,4]*ur[1,3,2]*projector_r[1,-1];
            #@tensor A_trun[:]:= M[4,3,-2]*A[2,4]*ur[-1,3,2];
        elseif (Rank(A)==2+1)&(Rank(M)==3)
            #@tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
            #ur=UR;
            v1=space(M,2);
            v2=space(A,1);
            ur=unitary(fuse(v1*v2),v1*v2);
            #@tensor A_trun[:]:= M[4,3,-2]*A[2,4,-3]*ur[1,3,2]*projector_r[1,-1]; #incorrect! projector should be determined later!
            @tensor A_trun[:]:= M[4,3,-2]*A[2,4,-3]*ur[-1,3,2];
        elseif (Rank(A)==2+2)&(Rank(M)==3)
            #@tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
            #ur=UR;
            v1=space(M,2);
            v2=space(A,1);
            ur=unitary(fuse(v1*v2),v1*v2);
            @tensor A_trun[:]:= M[4,3,-2]*A[2,4,-3,-4]*ur[1,3,2]*projector_r[1,-1];
        elseif (Rank(A)==2)&(Rank(M)==3+1)
            #@tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
            #ur=UR;
            v1=space(M,2);
            v2=space(A,1);
            ur=unitary(fuse(v1*v2),v1*v2);
            #@tensor A_trun[:]:= M[4,3,-2,-3]*A[2,4]*ur[1,3,2]*projector_r[1,-1];#incorrect! projector should be determined later!
            @tensor A_trun[:]:= M[4,3,-2,-3]*A[2,4]*ur[-1,3,2];
        elseif (Rank(A)==2+1)&(Rank(M)==3+1)
            #@tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
            #ur=UR;
            v1=space(M,2);
            v2=space(A,1);
            ur=unitary(fuse(v1*v2),v1*v2);
            #@tensor A_trun[:]:= M[4,3,-2,-3]*A[2,4,-4]*ur[1,3,2]*projector_r[1,-1];#incorrect! projector should be determined later!
            @tensor A_trun[:]:= M[4,3,-2,-3]*A[2,4,-4]*ur[-1,3,2];

            #combine two physical legs
            U_tem=unitary(fuse(space(A_trun,3)*space(A_trun,4)), space(A_trun,3)*space(A_trun,4));
            @tensor A_trun[:]:= A_trun[-1,-2,1,2]*U_tem[-3,1,2];
        end
        return A_trun
    end

    function truncate_middle(M,A,unitarys_R,UR,projector_r,unitarys_L,UL,projector_l,pos)
        if (Rank(A)==3)&(Rank(M)==4)#without open physical leg
            @tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
            @tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
            @tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5]*ul[4,3,1]*projector_l[-1,1]*ur[2,6,7]*projector_r[2,-2];
            #@tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5]*ul[4,3,-1]*ur[-2,6,7];
        elseif (Rank(A)==3+1)&(Rank(M)==4)

            #@tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5,-4]*ul[4,3,1]*projector_l[-1,1]*ur[2,6,7]*projector_r[2,-2];
            #@tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5,-4]*ul[4,3,-1]*ur[-2,6,7];
            if pos==pos_op[1]
                #ur=UR;
                v1=space(M,3);
                v2=space(A,2);
                ur=unitary(fuse(v1*v2),v1*v2);
                @tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
                @tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5,-4]*ul[4,3,1]*projector_l[-1,1]*ur[-2,6,7];
            elseif pos==pos_op[2]
                @tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
                #ul=UL;
                v1=space(M,1);
                v2=space(A,1);
                ul=unitary(fuse(v1*v2),v1'*v2')';
                @tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5,-4]*ul[4,3,-1]*ur[2,6,7]*projector_r[2,-2];
            end
        elseif (Rank(A)==3+2)&(Rank(M)==4)
            @tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
            @tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
            @tensor A_trun[:]:= M[4,5,6,-3]*A[3,7,5,-4,-5]*ul[4,3,1]*projector_l[-1,1]*ur[2,6,7]*projector_r[2,-2];
        elseif (Rank(A)==3)&(Rank(M)==4+1)
            #@tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5]*ul[4,3,1]*projector_l[-1,1]*ur[2,6,7]*projector_r[2,-2];
            if pos==pos_op[1]
                #ur=UR;
                v1=space(M,3);
                v2=space(A,2);
                ur=unitary(fuse(v1*v2),v1*v2);
                @tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
                @tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5]*ul[4,3,1]*projector_l[-1,1]*ur[-2,6,7];
            elseif pos==pos_op[2]
                @tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
                #ul=UL;
                v1=space(M,1);
                v2=space(A,1);
                ul=unitary(fuse(v1*v2),v1'*v2')';
                @tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5]*ul[4,3,-1]*ur[2,6,7]*projector_r[2,-2];
            end
        elseif (Rank(A)==3+1)&(Rank(M)==4+1)
            #@tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5,-5]*ul[4,3,1]*projector_l[-1,1]*ur[2,6,7]*projector_r[2,-2];
            if pos==pos_op[1]
                #ur=UR;
                v1=space(M,3);
                v2=space(A,2);
                ur=unitary(fuse(v1*v2),v1*v2);
                @tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
                @tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5,-5]*ul[4,3,1]*projector_l[-1,1]*ur[-2,6,7];
            elseif pos==pos_op[2]
                @tensor ur[:]:=unitarys_R[1,-1]*UR[1,-2,-3];
                #ul=UL;
                v1=space(M,1);
                v2=space(A,1);
                ul=unitary(fuse(v1*v2),v1'*v2')';
                @tensor A_trun[:]:= M[4,5,6,-3,-4]*A[3,7,5,-5]*ul[4,3,-1]*ur[2,6,7]*projector_r[2,-2];
            end

            #combine two physical legs
            U_tem=unitary(fuse(space(A_trun,4)*space(A_trun,5)), space(A_trun,4)*space(A_trun,5));
            @tensor A_trun[:]:= A_trun[-1,-2,-3,1,2]*U_tem[-4,1,2];
        end
        return A_trun
    end

    function truncate_right(M,A,unitarys_L,UL,projector_l,pos)
        if (Rank(A)==2)&(Rank(M)==3)#without open physical leg
            @tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
            @tensor A_trun[:]:= M[3,4,-2]*A[2,4]*ul[3,2,1]*projector_l[-1,1];
            #@tensor A_trun[:]:= M[3,4,-2]*A[2,4]*ul[3,2,-1];
        elseif (Rank(A)==2+1)&(Rank(M)==3)
            #@tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
            #ul=UL;
            v1=space(M,1);
            v2=space(A,1);
            ul=unitary(fuse(v1*v2),v1'*v2')';
            #@tensor A_trun[:]:= M[3,4,-2]*A[2,4,-3]*ul[3,2,1]*projector_l[-1,1];#incorrect! projector should be determined later!
            @tensor A_trun[:]:= M[3,4,-2]*A[2,4,-3]*ul[3,2,-1];
        elseif (Rank(A)==2+2)&(Rank(M)==3)
            #@tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
            #ul=UL;
            v1=space(M,1);
            v2=space(A,1);
            ul=unitary(fuse(v1*v2),v1'*v2')';
            @tensor A_trun[:]:= M[3,4,-2]*A[2,4,-3,-4]*ul[3,2,1]*projector_l[-1,1];
        elseif (Rank(A)==2)&(Rank(M)==3+1)
            #@tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
            #ul=UL;
            v1=space(M,1);
            v2=space(A,1);
            ul=unitary(fuse(v1*v2),v1'*v2')';
            #@tensor A_trun[:]:= M[3,4,-2,-3]*A[2,4]*ul[3,2,1]*projector_l[-1,1];#incorrect! projector should be determined later!
            @tensor A_trun[:]:= M[3,4,-2,-3]*A[2,4]*ul[3,2,-1];
        elseif (Rank(A)==2+1)&(Rank(M)==3+1)
            #@tensor ul[:]:=UL[-1,-2,1]*unitarys_L[-3,1];
            #ul=UL;
            v1=space(M,1);
            v2=space(A,1);
            ul=unitary(fuse(v1*v2),v1'*v2')';
            #@tensor A_trun[:]:= M[3,4,-2,-3]*A[2,4,-4]*ul[3,2,1]*projector_l[-1,1];#incorrect! projector should be determined later!
            @tensor A_trun[:]:= M[3,4,-2,-3]*A[2,4,-4]*ul[3,2,-1];

            #combine two physical legs
            U_tem=unitary(fuse(space(A_trun,3)*space(A_trun,4)), space(A_trun,3)*space(A_trun,4));
            @tensor A_trun[:]:= A_trun[-1,-2,1,2]*U_tem[-3,1,2];
        end
        return A_trun
    end

    cx=1;
    mps_new[cx]=truncate_left(mpo[cx],mps_old[cx],unitarys_R[cx],UR[cx],projectors_R[cx],cx);
    #mps_new[cx]=truncate_left(mpo[cx],mps_old[cx],ur,[]);

    for cx=2:L-1
        mps_new[cx]=truncate_middle(mpo[cx],mps_old[cx],unitarys_R[cx],UR[cx], projectors_R[cx], unitarys_L[cx], UL[cx], projectors_L[cx],cx);
        #mps_new[cx]=truncate_middle(mpo[cx],mps_old[cx], ur,[], ul,[]);
    end

    cx=L;
    mps_new[cx]=truncate_right(mpo[cx],mps_old[cx],unitarys_L[cx], UL[cx],projectors_L[cx],cx);
    #mps_new[cx]=truncate_right(mpo[cx],mps_old[cx],ul,[]);


    # for c1=1:L
    #     println(norm(mps_new[c1]))
    # end 

    function truncate_with_op(T1,T2)
        global chi,multiplet_tol;
        if pos_op[1]==1
            u1,s1,v1=tsvd(T1,(2,3,),(1,));
            u2,s2,v2=tsvd(T2,(1,),(2,3,4,));
            M=s1*v1*u2*s2;
            u3,s3,v3=my_tsvd(M;trunc=truncdim(chi+20));
            # s3=truncate_multiplet(s3,chi,multiplet_tol,1e-8);
            # u3,s3,v3=delet_zero_block(u3,s3,v3);
            T1_new=u1*u3*s3;
            T2_new=v3*v2;
            T1_new=permute(T1_new,(3,1,2,));
            T2_new=permute(T2_new,(1,2,3,4,));
        elseif pos_op[2]==L
            u1,s1,v1=tsvd(T1,(1,3,4,),(2,));
            u2,s2,v2=tsvd(T2,(1,),(2,3,));
            M=s1*v1*u2*s2;
            u3,s3,v3=my_tsvd(M;trunc=truncdim(chi+20));
            # s3=truncate_multiplet(s3,chi,multiplet_tol,1e-8);
            # u3,s3,v3=delet_zero_block(u3,s3,v3);
            T1_new=u1*u3*s3;
            T2_new=v3*v2;
            T1_new=permute(T1_new,(1,4,2,3,));
            T2_new=permute(T2_new,(1,2,3,));
        else
            u1,s1,v1=tsvd(T1,(1,3,4,),(2,));
            u2,s2,v2=tsvd(T2,(1,),(2,3,4,));
            M=s1*v1*u2*s2;
            u3,s3,v3=my_tsvd(M;trunc=truncdim(chi+20));
            # s3=truncate_multiplet(s3,chi,multiplet_tol,1e-8);
            # u3,s3,v3=delet_zero_block(u3,s3,v3);
            T1_new=u1*u3*s3;
            T2_new=v3*v2;
            T1_new=permute(T1_new,(1,4,2,3,));
            T2_new=permute(T2_new,(1,2,3,4,));
        end
        return [T1_new,T2_new]
    end


    if truncate_special_bond
        if length(pos_op)>1
            mps_new[pos_op[1:2]]=truncate_with_op(mps_new[pos_op[1]],mps_new[pos_op[2]]);
        end
    end
    

    return mps_new 
end



function construct_double_layer_open_plaquatte(psi,x_range,y_range)
    Lx=size(psi,1);
    Ly=size(psi,2);
    psi_double_=construct_double_layer(psi,psi);

    for ca =1:length(x_range)
        for cb=1:length(y_range)
            psi_double_[x_range[ca],y_range[cb]]=build_double_layer_open_position(psi[x_range[ca],y_range[cb]],x_range[ca],y_range[cb],Lx,Ly)
        end
    end

    return psi_double_
end


function pi_rotate_mpo_down_move(mpo_set)
    Lx=length(mpo_set);
    mpo_set_new=Vector{TensorMap}(undef,Lx);

    cx=1;
    T=mpo_set[cx];
    if Rank(T)==3
        T=permute(T,(2,3,1,));
    elseif Rank(T)==4
        T=permute(T,(2,3,1,4,));
    end
    mpo_set_new[cx]=T;

    for cx=2:Lx-1
        T=mpo_set[cx];
        if Rank(T)==4
            T=permute(T,(3,4,1,2,));
        elseif Rank(T)==5
            T=permute(T,(3,4,1,2,5,));
        end
        mpo_set_new[cx]=T;
    end

    cx=Lx;
    T=mpo_set[cx];
    if Rank(T)==3
        T=permute(T,(3,1,2,));
    elseif Rank(T)==4
        T=permute(T,(3,1,2,4,));
    end
    mpo_set_new[cx]=T;

    return mpo_set_new[end:-1:1]
end

function pi_rotate_mps_down_move(mps_set)
    Lx=length(mps_set);
    mps_set_new=Vector{TensorMap}(undef,Lx);

    cx=1;
    T=mps_set[cx];
    if Rank(T)==2
        T=permute(T,(2,1,));
    elseif Rank(T)==3
        T=permute(T,(2,1,3,));
    end
    mps_set_new[cx]=T;

    for cx=2:Lx-1
        T=mps_set[cx];
        if Rank(T)==3
            T=permute(T,(3,1,2,));
        elseif Rank(T)==4
            T=permute(T,(3,1,2,4,));
        end
        mps_set_new[cx]=T;
    end

    cx=Lx;
    T=mps_set[cx];
    if Rank(T)==2
        T=permute(T,(1,2,));
    elseif Rank(T)==3
        T=permute(T,(1,2,3,));
    end
    mps_set_new[cx]=T;

    return mps_set_new[end:-1:1]
end


function pi_inverse_rotate_mps_down_move(mps_set)
    #index order: left virtual, right virtual, physical
    Lx=length(mps_set);
    mps_set_new=Vector{TensorMap}(undef,Lx);

    cx=1;
    T=mps_set[cx];
    if Rank(T)==2
        T=permute(T,(1,2,));
    elseif Rank(T)==3
        T=permute(T,(1,2,3,));
    elseif Rank(T)==4
        T=permute(T,(1,2,3,4,));
    end
    mps_set_new[cx]=T;

    for cx=2:Lx-1
        T=mps_set[cx];
        if Rank(T)==3
            T=permute(T,(2,1,3,));
        elseif Rank(T)==4
            T=permute(T,(2,1,3,4,));
        elseif Rank(T)==5
            T=permute(T,(2,1,3,4,5,));
        end
        mps_set_new[cx]=T;
    end

    cx=Lx;
    T=mps_set[cx];
    if Rank(T)==2
        T=permute(T,(1,2,));
    elseif Rank(T)==3
        T=permute(T,(1,2,3,));
    elseif Rank(T)==4
        T=permute(T,(1,2,3,4,));
    end
    mps_set_new[cx]=T;

    return mps_set_new[end:-1:1]
end


function get_projector_up_move(psi_double,L_max=100)
    global multiplet_tol, chi;
    Lx=size(psi_double,1);
    Ly=size(psi_double,2);

    UR_set=Vector{Vector}(undef,Ly);
    UL_set=Vector{Vector}(undef,Ly);
    unitarys_R_set=Vector{Vector}(undef,Ly);
    unitarys_L_set=Vector{Vector}(undef,Ly);
    projectors_R_set=Vector{Vector}(undef,Ly);
    projectors_L_set=Vector{Vector}(undef,Ly);
    mps_set_up_move=Matrix{TensorMap}(undef,Lx,Ly);
    trun_errs=[];

    norm_coe=Vector{Number}(undef,Ly);

    cy=1;
    mps_trun=psi_double[:,cy];
    mps_set_up_move[:,cy]=mps_trun;
    for cy=2:min(L_max,Ly-1-1)
        if cy<Ly-1
            mps_set_big,UR,UL=apply_mpo(psi_double[:,cy], mps_trun);
            mps_trun,trun_err, unitarys_R,unitarys_L, projectors_R,projectors_L=left_truncate_simple(mps_set_big, chi, multiplet_tol);
            trun_errs=vcat(trun_errs,trun_err);
            mps_trun=collect(mps_trun);

            coe=norm(mps_trun[end]);
            mps_trun[end]=mps_trun[end]/coe;
            norm_coe[cy]=coe;


            mps_set_up_move[:,cy]=mps_trun;#tuple to Array
            UR_set[cy]=UR;
            UL_set[cy]=UL;
            unitarys_R_set[cy]=unitarys_R;
            unitarys_L_set[cy]=unitarys_L;
            projectors_R_set[cy]=projectors_R;
            projectors_L_set[cy]=projectors_L;
        # elseif cy==Ly-1
        #     mps_set_big,UR,UL=apply_mpo(psi_double[:,cy], mps_trun);
        #     mps_trun=collect(mps_set_big);

        #     coe=1;
        #     mps_trun[end]=mps_trun[end]/coe;
        #     norm_coe[cy]=coe;


        #     mps_set_up_move[:,cy]=mps_trun;#tuple to Array
        #     UR_set[cy]=UR;
        #     UL_set[cy]=UL;
        #     unitarys_R_set[cy]=[];
        #     unitarys_L_set[cy]=[];
        #     projectors_R_set[cy]=[];
        #     projectors_L_set[cy]=[];
        end
    end
    return mps_set_up_move,trun_errs, norm_coe, UR_set, UL_set, unitarys_R_set, unitarys_L_set, projectors_R_set, projectors_L_set
end



function get_projector_down_move(psi_double,L_min=2+1)

    global multiplet_tol, chi;
    Lx=size(psi_double,1);
    Ly=size(psi_double,2);

    UR_set=Vector{Vector}(undef,Ly);
    UL_set=Vector{Vector}(undef,Ly);
    unitarys_R_set=Vector{Vector}(undef,Ly);
    unitarys_L_set=Vector{Vector}(undef,Ly);
    projectors_R_set=Vector{Vector}(undef,Ly);
    projectors_L_set=Vector{Vector}(undef,Ly);
    mps_set_down_move=Matrix{TensorMap}(undef,Lx,Ly);
    trun_errs=[];
    norm_coe=Vector{Number}(undef,Ly);

    cy=Ly;
    mps_trun=pi_rotate_mps_down_move(psi_double[:,cy]);
    mps_set_down_move[:,cy]=pi_inverse_rotate_mps_down_move(mps_trun);
    for cy=Ly-1:-1:max(L_min,2+1)
        if cy>2
            mpo_set=pi_rotate_mpo_down_move((psi_double[:,cy]...,));
            mps_set_big,UR,UL=apply_mpo(mpo_set, mps_trun);
            mps_trun,trun_errs, unitarys_R,unitarys_L, projectors_R,projectors_L=left_truncate_simple(mps_set_big, chi, multiplet_tol);
            mps_trun=collect(mps_trun);
            coe=norm(mps_trun[end]);
            mps_trun[end]=mps_trun[end]/coe;
            norm_coe[cy]=coe;

            mps_set_down_move[:,cy]=pi_inverse_rotate_mps_down_move(mps_trun);
            UR_set[cy]=UR;
            UL_set[cy]=UL;
            unitarys_R_set[cy]=unitarys_R;
            unitarys_L_set[cy]=unitarys_L;
            projectors_R_set[cy]=projectors_R;
            projectors_L_set[cy]=projectors_L;
        # elseif cy==2
        #     mpo_set=pi_rotate_mpo_down_move((psi_double[:,cy]...,));
        #     mps_set_big,UR,UL=apply_mpo(mpo_set, mps_trun);
        #     mps_trun=collect(mps_set_big);
        #     coe=1;
        #     mps_trun[end]=mps_trun[end]/coe;
        #     norm_coe[cy]=coe;

        #     mps_set_down_move[:,cy]=pi_inverse_rotate_mps_down_move(mps_trun);
        #     UR_set[cy]=UR;
        #     UL_set[cy]=UL;
        #     unitarys_R_set[cy]=[];
        #     unitarys_L_set[cy]=[];
        #     projectors_R_set[cy]=[];
        #     projectors_L_set[cy]=[];
        end
    end
    return mps_set_down_move,trun_errs, norm_coe, UR_set, UL_set, unitarys_R_set, unitarys_L_set, projectors_R_set, projectors_L_set
end