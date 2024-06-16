function add_global_noise(psi,init_noise)
    psi=deepcopy(psi);
    Lx=size(psi,1);
    Ly=size(psi,2);
    for cx=1:Lx
        for cy=1:Ly
            A=psi[cx,cy];
            A_noise=TensorMap(randn,codomain(A),domain(A))+im*TensorMap(randn,codomain(A),domain(A));
            A=A+A_noise*init_noise*norm(A)/norm(A_noise);
            psi[cx,cy]=A;
        end
    end
    return psi
end
function initial_SU2_state(init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    if init_statenm=="nothing" 
    else
        println("load state: "*init_statenm);flush(stdout);
        data=load(init_statenm);

        psi=data["psi"];
        psi=add_global_noise(psi,init_noise);
        return psi;
    end
end

function create_isometry(V1,V2)
    tt=TensorMap(randn,V1,V2);
    for cc=1:length(tt.data.values)
        mm=tt.data.values[cc];
        tt.data.values[cc]=Matrix(I, size(mm,1), size(mm,2));
    end
    return tt
end

function examine_ind(Lx,Ly,pos,ind)
    px=pos[1];
    py=pos[2];
    if px==1
        if py==1
            return ind-2
        elseif 1<py<Ly
            return ind-1
        elseif py==Ly
            return ind-1
        end
    elseif 1<px<Lx
        if py==1
            if ind==1
                return ind
            elseif (ind==3)|(ind==4)
                return ind-1
            end
        elseif 1<py<Ly
            return ind
        elseif py==Ly
            return ind
        end
    elseif px==Lx
        if py==1
            if ind==1
                return ind
            elseif (ind==4)
                return ind-2
            end
        elseif 1<py<Ly
            if (ind==1)|(ind==2)
                return ind
            elseif (ind==4)
                return ind-1
            end
        elseif py==Ly
            return ind
        end
    end
end

function extend_single_bond(virtual_spin, psi, bond_pos, Noise)
    psi=deepcopy(psi);
    Lx,Ly=size(psi);
    bx=bond_pos[1];
    by=bond_pos[2];
    @assert 1<=bx<=Lx;
    @assert 1<=by<=Ly;
    if mod(bx,1)==0.5
        bond_type="x";
    elseif mod(by,1)==0.5
        bond_type="y";
    else
        error("unknown bond");
    end

    if bond_type=="x"
        pos_T1=[bx-0.5,by];
        pos_T2=[bx+0.5,by];
        ind_T1=3;
        ind_T2=1;
    elseif bond_type=="y"
        pos_T1=[bx,by+0.5];
        pos_T2=[bx,by-0.5];
        ind_T1=2;
        ind_T2=4;
    end
    pos_T1=Int.(pos_T1);
    pos_T2=Int.(pos_T2);



    ind_T1=examine_ind(Lx,Ly,pos_T1,ind_T1);
    ind_T2=examine_ind(Lx,Ly,pos_T2,ind_T2);
    @assert space(psi[pos_T1[1],pos_T1[2]],ind_T1)==space(psi[pos_T2[1],pos_T2[2]],ind_T2)';
    V0=space(psi[pos_T1[1],pos_T1[2]],ind_T1);
    if V0.dual
        V_new=V0⊕Rep[SU₂](virtual_spin=>1)';
    else
        V_new=V0⊕Rep[SU₂](virtual_spin=>1);
    end

    

    function apply_U(T,U,ind)
        if Rank(T)==3
            if ind==1
                @tensor T_new[:]:=T[1,-2,-3]*U[-1,1];
            elseif ind==2
                @tensor T_new[:]:=T[-1,2,-3]*U[-2,2];
            end
        elseif Rank(T)==4
            if ind==1
                @tensor T_new[:]:=T[1,-2,-3,-4]*U[-1,1];
            elseif ind==2
                @tensor T_new[:]:=T[-1,2,-3,-4]*U[-2,2];
            elseif ind==3
                @tensor T_new[:]:=T[-1,-2,3,-4]*U[-3,3];
            end
        elseif Rank(T)==5
            if ind==1
                @tensor T_new[:]:=T[1,-2,-3,-4,-5]*U[-1,1];
            elseif ind==2
                @tensor T_new[:]:=T[-1,2,-3,-4,-5]*U[-2,2];
            elseif ind==3
                @tensor T_new[:]:=T[-1,-2,3,-4,-5]*U[-3,3];
            elseif ind==4
                @tensor T_new[:]:=T[-1,-2,-3,4,-5]*U[-4,4];
            end
        end
    end
    
    U=create_isometry(V_new,V0);
    T1=psi[pos_T1[1],pos_T1[2]];
    T2=psi[pos_T2[1],pos_T2[2]];

    T1=apply_U(T1,U,ind_T1);
    T2=apply_U(T2,permute(U',(2,),(1,)),ind_T2);

    function add_noise(A,init_noise)
        A_noise=TensorMap(randn,codomain(A),domain(A))+im*TensorMap(randn,codomain(A),domain(A));
        A=A+A_noise*init_noise*norm(A)/norm(A_noise);
    end

    T1=add_noise(T1,Noise);
    T2=add_noise(T2,Noise);
    
    psi[pos_T1[1],pos_T1[2]]=T1;
    psi[pos_T2[1],pos_T2[2]]=T2;
    return psi
end




function add_virtual_particle_global(virtual_spin, psi, Noise)
    Lx,Ly=size(psi);
    psi=deepcopy(psi);

    for ba=1.5:1:Lx-0.5
        for bb=1:Ly
            bond_pos=[ba,bb];
            psi=extend_single_bond(virtual_spin, psi, bond_pos, Noise);
        end
    end
    
    for ba=1:Lx
        for bb=1.5:1:Ly-0.5
            bond_pos=[ba,bb];
            psi=extend_single_bond(virtual_spin, psi, bond_pos, Noise);
        end
    end
    return psi
end

function add_virtual_particle_boundary(virtual_spin, psi, Noise)
    Lx,Ly=size(psi);
    psi=deepcopy(psi);

    ba=1;
    for bb=1.5:1:Ly-0.5
        bond_pos=[ba,bb];
        psi=extend_single_bond(virtual_spin, psi, bond_pos, Noise);
    end

    ba=Lx;
    for bb=1.5:1:Ly-0.5
        bond_pos=[ba,bb];
        psi=extend_single_bond(virtual_spin, psi, bond_pos, Noise);
    end
    
    for ba=1.5:1:Lx-0.5
        bb=1;
        bond_pos=[ba,bb];
        psi=extend_single_bond(virtual_spin, psi, bond_pos, Noise);
    end

    for ba=1.5:1:Lx-0.5
        bb=Ly;
        bond_pos=[ba,bb];
        psi=extend_single_bond(virtual_spin, psi, bond_pos, Noise);
    end
    return psi
end



function normalize_PEPS(psi0)
    #normalize PEPS ansatz 
    psi00=deepcopy(psi0);
    psi=deepcopy(psi0);

    Lx,Ly=size(psi);


    psi_double_open, U_s_s=construct_double_layer_open(psi);

    # AA1,_=build_double_layer_bulk_open(psi[2,2], psi[2,2], false);
    # AA2,_=build_double_layer_bulk_open(psi[2,2],false);




    global U_s_s
    pos_bra=[1,1];
    pos_ket=[1,1];
    psi_double=contract_physical_all(psi_double_open, U_s_s);

    psi=disk_to_torus(psi);
    # psi=remove_trivial_boundary_leg(psi);
    A_bra=psi[pos_bra[1],pos_bra[2]];
    A_ket=psi[pos_ket[1],pos_ket[2]];

    real_imag="real";#"real" or "imag"
    N_operator_sites=1;

    N0=cost_fun_bra_ket(N_operator_sites,A_bra,A_ket,pos_bra,pos_ket,psi,psi,psi_double_open,psi_double,U_s_s,"norm",real_imag);
    N0=N0/(Lx-1)/(Ly-1);

    coe=(N0)^(1/2/Lx/Ly);
    for c1=1:Lx
        for c2=1:Ly
            psi00[c1,c2]=psi00[c1,c2]/coe;
        end
    end
    return psi00
end