


function solve_ITEBD_excitation_TrunTransOp_iterative(Ag,O1,O2,OO,n_E,N,kset,ES_sectors,pow,U,S,Vt,SPIN_group,DTrun_group,mpo_type,multi_threads)
    
    Eset=Matrix{Any}(undef, length(kset),length(ES_sectors)); 

    for kk=1:length(kset)

        ck=kset[kk]
        k=2*pi*ck/N

        norm_eff=excitation_TrunTransOp_iterative_norm_eff(Ag,pow,N,k) # put it on cpu because this matrix maybe large
        norm_eff=permute(norm_eff,(1,2,3,),(4,5,6,))
        uu,ss,vvt=tsvd(norm_eff, trunc=truncerr(0.0000001));
        norm_eff=[]#clear this big matrix
        input_transform=vvt';
        output_transform=uu';
        output_transform=output_transform;
        output_transform=pinv(ss)*output_transform;

        for sector_ind=1:length(ES_sectors)
            SPIN=ES_sectors[sector_ind];
            sectr=Irrep[SU₂](SPIN);
            display("sector "*"k="*string(ck)*", spin="*string(SPIN)*":");

            if dim(fuse(domain(input_transform,1)),Irrep[SU₂](SPIN))==0
                display("MPS decomposition does not have this sector, skip it")
                Eset[kk,sector_ind]=[];
                continue;
            end

            v_init=TensorMap(randn,domain(input_transform), SU₂Space(SPIN=>1));
            
            excitation_iterative(x)=excitation_TrunTransOp_iterative_H_eff(x,input_transform,output_transform,O1,O2,OO,Ag,pow,U,S,Vt,SPIN_group,N,k,DTrun_group,mpo_type,multi_threads)

            @time Es,_=eigsolve(excitation_iterative, v_init, n_E,:LM,Arnoldi(krylovdim=minimum([10,n_E*3])));
            Eset[kk,sector_ind]=Es
            display(Es)
        end

    end


    return kset, Eset
end

        
function excitation_TrunTransOp_iterative_H_eff(x,input_transform,output_transform,O1,O2,OO,Ag,pow,VR,EU,VL,SPIN,N,k,DTrun_list,mpo_type,multi_threads)
    @assert pow==round((N-2)/2)
    x=input_transform*x;# do this calculation on cpu because the matrix 'input_transform' maybe large
    x=permute(x,(1,2,3,4,),());
    H_eff_x=x*0;
    cm=1;

    H_eff_x_set=Vector{Any}(undef, N);
    if multi_threads
        Threads.@threads for cn=1:N
            coe=exp(-im*k*(cm-cn))
            H_eff_x0=H_eff_x*0;
            for c_sector=1:length(DTrun_list)
                for  c_comp=1:length(DTrun_list[c_sector])
                    #display(DTrun_list[c_sector][c_comp])
                    
                    H_eff_x0=H_eff_x0+coe*contract_H_eff_trun(x,O1,O2,OO,Ag,VR[c_sector][c_comp],EU[c_sector][c_comp],VL[c_sector][c_comp],SPIN[c_sector],cm,cn,pow,N,DTrun_list[c_sector][c_comp],mpo_type);
                end
            end
            H_eff_x_set[cn]=H_eff_x0
        end
    else
        for cn=1:N
            coe=exp(-im*k*(cm-cn))
            H_eff_x0=H_eff_x*0;
            for c_sector=1:length(DTrun_list)
                for  c_comp=1:length(DTrun_list[c_sector])
                    #display(DTrun_list[c_sector][c_comp])
                    
                    H_eff_x0=H_eff_x0+coe*contract_H_eff_trun(x,O1,O2,OO,Ag,VR[c_sector][c_comp],EU[c_sector][c_comp],VL[c_sector][c_comp],SPIN[c_sector],cm,cn,pow,N,DTrun_list[c_sector][c_comp],mpo_type);
                end
            end
            H_eff_x_set[cn]=H_eff_x0
        end
    end

    for cn=1:N
        H_eff_x=H_eff_x+H_eff_x_set[cn];
    end
    x=permute(H_eff_x,(1,2,3,),(4,));
    x_output=output_transform*x;
    return x_output
end

function excitation_TrunTransOp_iterative_norm_eff(Ag,pow,N,k)
    @assert pow==round((N-2)/2)
    norm_eff=permute(TensorMap(randn,codomain(Ag)←codomain(Ag))*0,(1,2,3,4,5,6,),());
    cm=1;
    for cn=1:N
            coe=exp(-im*k*(cm-cn))
            norm_eff=norm_eff+coe*contract_norm_eff(Ag,cm,cn,N)
    end
    return norm_eff
end



function create_puMPS(A,N)
    mps=Vector{Any}(undef, N);
    for c1=1:N
        mps[c1]=A
    end
    return mps
end


function contiguous_TM(mps1,mps2,positions)
    if length(positions)==1
        @tensor E[:]:=mps1[positions[1]]'[-1,-3,3]*mps2[positions[1]][-2,-4,3];
    elseif length(positions)>1
        @tensor E[:]:=mps1[positions[1]]'[-1,-3,3]*mps2[positions[1]][-2,-4,3];
        for ci=positions[2]:positions[end]
            @tensor E[:]:=E[-1,-2,1,2]*mps1[ci]'[1,-3,3]*mps2[ci][2,-4,3];
        end
    end
    return E
end


function contract_norm_eff(AL,cm,cn,N)
    mps=create_puMPS(AL,N);
    mps1=mps;
    mps2=mps;
    if cm==cn # cm=cn=1
        E=contiguous_TM(mps1,mps2,2:N)
        Id=id(space(Ag,3));
        @tensor norm_eff[:]:=E[-2,-5,-1,-4]*Id[-3,-6];
    elseif cn==2
        E=contiguous_TM(mps1,mps2,3:N);
        @tensor norm_eff[:]:=mps2[1][1,-4,-3]*mps1[2]'[-2,2,-6]*E[2,-5,-1,1];
    elseif cn==N
        E=contiguous_TM(mps1,mps2,2:N-1);
        @tensor norm_eff[:]:=mps2[1][-5,1,-3]*E[-2,1,2,-4]*mps1[N]'[2,-1,-6]
    else
        T1=mps2[1];
        T2=contiguous_TM(mps1,mps2,2:(cn-1))
        T3=mps1[cn]';
        T4=contiguous_TM(mps1,mps2,(cn+1):N)
        @tensor norm_eff[:]:=T1[6,1,-3]*T2[-2,1,3,-4]*T3[3,4,-6]*T4[4,-5,-1,6];
    end
    return norm_eff
end


function contract_H_eff_trun(x,O1,O2,OO,Ag,VR_comp,EU_comp,VL_comp,spin_comp,cm,cn,pow,N,D_set,mpo_type)
    H_eff_x=x*0;
    L_Ea=cn-cm-1
    L_Eb=N-cn
    if L_Ea>=L_Eb
        L=L_Ea
        decomp_posit="Ea"
    elseif L_Ea<L_Eb
        L=L_Eb
        decomp_posit="Eb"
    end
    
    VL_comp=v_Transop(VL_comp,Ag,O1,O2,OO,L-pow,mpo_type);
    VR_comp=VR_comp*EU_comp;

    if mpo_type=="O_O"
        if cm==cn # cm=cn=0
            @tensor ttt[:]:=VL_comp[9,-1,5,3,1]*O2[5,4,6,-3]*O1[3,2,7,4]*x[1,8,2,-4]*VR_comp[-2,6,7,8,9];
            H_eff_x=H_eff_x+ttt;
        elseif cn==2
            @tensor T1[:]:=VL_comp[-1,-2,5,3,1]*O2[5,4,-3,-6]*O1[3,2,-4,4]*Ag[1,-5,2];
            @tensor T2[:]:=Ag'[-1,1,2]*O2[-2,4,3,2]*O1[-3,-5,5,4]*VR_comp[1,3,5,-4,-6];
            @tensor T1T2x[:]:=T1[6,-1,3,4,5,-3]*T2[-2,3,4,2,1,6]*x[5,2,1,-4];
            H_eff_x=H_eff_x+T1T2x;
        elseif cn==N
            @tensor T1[:]:=VL_comp[-1,1,3,5,-2]*Ag'[1,-3,2]*O2[3,4,-4,2]*O1[5,-6,-5,4];
            @tensor T2[:]:=O2[-1,4,5,-5]*O1[-2,2,3,4]*Ag[-3,1,2]*VR_comp[-4,5,3,1,-6];
            @tensor T1xT2[:]:=T1[6,1,-1,3,4,2]*x[1,5,2,-4]*T2[3,4,5,-2,-3,6];
            H_eff_x=H_eff_x+T1xT2;
        else
            if  decomp_posit=="Ea"
                @tensor T1[:]:=VL_comp[-1,1,3,5,7]*Ag'[1,-2,2]*O2[3,4,-3,2]*O1[5,6,-4,4]*x[7,-5,6,-6];
                @tensor T2[:]:=O2[-1,4,5,-5]*O1[-2,2,3,4]*Ag[-3,1,2]*VR_comp[-4,5,3,1,-6];
                @assert L_Eb>=1
                for cr=1:L_Eb
                    @tensor T1[:]:=T1[-1,1,3,5,7,-6]*Ag'[1,-2,2]*O2[3,4,-3,2]*O1[5,6,-4,4]*Ag[7,-5,6];
                end
                @tensor T1T2[:]:=T1[4,-1,1,2,3,-4]*T2[1,2,3,-2,-3,4];
                H_eff_x=H_eff_x+T1T2;
            elseif decomp_posit=="Eb"
                @tensor T1[:]:=VL_comp[-1,-2,5,3,1]*O2[5,4,-3,-6]*O1[3,2,-4,4]*Ag[1,-5,2];
                @tensor T2[:]:=Ag'[-1,1,2]*O2[-2,4,3,2]*O1[-3,6,5,4]*x[-4,7,6,-6]*VR_comp[1,3,5,7,-5];
                @assert L_Ea>=1
                for cr=1:L_Ea
                    @tensor T2[:]:=Ag'[-1,1,2]*O2[-2,4,3,2]*O1[-3,6,5,4]*Ag[-4,7,6]*T2[1,3,5,7,-5,-6];
                end
                @tensor T1T2[:]:=T1[4,-1,1,2,3,-3]*T2[-2,1,2,3,4,-4];
                H_eff_x=H_eff_x+T1T2;
            end
        end
    elseif mpo_type=="OO"
        if cm==cn # cm=cn=0
            @tensor ttt[:]:=VL_comp[6,-1,2,1]*OO[2,3,4,-3]*x[1,5,3,-4]*VR_comp[-2,4,5,6];
            H_eff_x=H_eff_x+ttt;
        elseif cn==2
            @tensor T1[:]:=VL_comp[-1,-2,3,1]*OO[3,2,-3,-5]*Ag[1,-4,2];
            @tensor T2[:]:=Ag'[-1,1,2]*OO[-2,-4,3,2]*VR_comp[1,3,-3,-5];
            @tensor T1T2x[:]:=T1[5,-1,4,3,-3]*T2[-2,4,2,1,5]*x[3,2,1,-4];
            H_eff_x=H_eff_x+T1T2x;
        elseif cn==N
            @tensor T1[:]:=VL_comp[-1,1,3,-2]*Ag'[1,-3,2]*OO[3,-5,-4,2];
            @tensor T2[:]:=OO[-1,2,3,-4]*Ag[-2,1,2]*VR_comp[-3,3,1,-5];
            @tensor T1xT2[:]:=T1[5,1,-1,3,2]*x[1,4,2,-4]*T2[3,4,-2,-3,5];
            H_eff_x=H_eff_x+T1xT2;
        else
            if  decomp_posit=="Ea"
                @tensor T1[:]:=VL_comp[-1,1,3,5]*Ag'[1,-2,2]*OO[3,4,-3,2]*x[5,-4,4,-5];
                @tensor T2[:]:=OO[-1,2,3,-4]*Ag[-2,1,2]*VR_comp[-3,3,1,-5];
                @assert L_Eb>=1
                for cr=1:L_Eb
                    @tensor T1[:]:=T1[-1,1,3,5,-5]*Ag'[1,-2,2]*OO[3,4,-3,2]*Ag[5,-4,4];
                end
                @tensor T1T2[:]:=T1[4,-1,1,2,-4]*T2[1,2,-2,-3,4];
                H_eff_x=H_eff_x+T1T2;
            elseif decomp_posit=="Eb"
                @tensor T1[:]:=VL_comp[-1,-2,3,1]*OO[3,2,-3,-5]*Ag[1,-4,2];
                @tensor T2[:]:=Ag'[-1,1,2]*OO[-2,4,3,2]*x[-3,5,4,-5]*VR_comp[1,3,5,-4];
                @assert L_Ea>=1
                for cr=1:L_Ea
                    @tensor T2[:]:=Ag'[-1,1,2]*OO[-2,4,3,2]*Ag[-3,5,4]*T2[1,3,5,-4,-5];
                end
                @tensor T1T2[:]:=T1[3,-1,1,2,-3]*T2[-2,1,2,3,-4];
                H_eff_x=H_eff_x+T1T2;
            end
        end
    end
    H_eff_x=H_eff_x*(2*spin_comp+1);
    return H_eff_x
end


    
function v_Transop(VL_comp,Ag,O1,O2,OO,L,mpo_type)
    VL_comp=deepcopy(VL_comp)
    if mpo_type=="O_O"
        for cl=1:L
            @tensor VL_comp[:]:=VL_comp[-1,1,3,5,7]*Ag'[1,-2,2]*O2[3,4,-3,2]*O1[5,6,-4,4]*Ag[7,-5,6];
        end
    elseif mpo_type=="OO"
        for cl=1:L
            @tensor VL_comp[:]:=VL_comp[-1,1,3,5]*Ag'[1,-2,2]*OO[3,4,-3,2]*Ag[5,-4,4];
        end
    end
    return VL_comp
end

