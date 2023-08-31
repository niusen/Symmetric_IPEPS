using TensorKit
using KrylovKit
cd(@__DIR__)

T=TensorMap(randn,Rep[SU₂](0=>8, 1/2=>10, 1=>4)' ⊗ Rep[SU₂](0=>2, 1/2=>2, 1=>1), Rep[SU₂](0=>8, 1/2=>10, 1=>4)');
T=permute(T,(1,3,2,),())

function HR(T,x)
    @tensor xp[:]:=T'[-1,1,2]*T[-2,3,2]*x[1,3,-3];
    return xp
end

function HR_conj(T,x)
    @tensor xp[:]:=T[1,-1,2]*T'[3,-2,2]*x[1,3,-3];
    return xp
end

spins=Vector(undef,0);
S_set=Vector(undef,0);
U_set=Vector(undef,0);
V_set=Vector(undef,0);

Vspace=fuse(space(T,2)*space(T,2));
for cs=1:length(Vspace.dims.keys)
    spin=Vspace.dims.keys[cs].j;
    
    vr_init=permute(TensorMap(randn, space(T',2)'*space(T,2)',SU₂Space(spin=>1)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
    GLR_svd_R(x)=HR(T,x);
    GLR_svd_R_conj(x)=HR_conj(T,x);
    n_keep=Vspace.dims.values[cs];
    S,U,V,info=svdsolve((GLR_svd_R,GLR_svd_R_conj), vr_init, n_keep,:LR, krylovdim=n_keep*3);
    S_set=vcat(S_set,S);U_set=vcat(U_set,U);V_set=vcat(V_set,V);
    spins=vcat(spins,spin*ones(length(S)));
    #@assert info.converged >= minimum([n_eff,dim(full_space,sec)])
end

for cc=1:length(S_set)
    U_set[cc]=permute(U_set[cc],(1,2,),(3,));
    V_set[cc]=permute(V_set[cc],(1,2,),(3,));
end


@tensor TT[:]:=T'[-1,-3,1]*T[-2,-4,1];
u,s,v=tsvd(TT,(1,2,),(3,4,))


TT_new=TT*0;
for cs=1:length(S_set)
    S=S_set[cs];
    U=U_set[cs];
    V=V_set[cs];
    spin=spins[cs];

    @tensor TT_comp[:]:=U[-1,-2,1]*V'[1,-3,-4]
    TT_new=TT_new+TT_comp*S*(2*spin+1);



end

@assert norm(TT-TT_new)/norm(TT)<1e-12



function group_svd_components(U_set,S_set,V_set,spins)
    allspin=sort(unique(spins));
    spin_dim=deepcopy(allspin);
    spin_range=deepcopy(allspin);
    for cs=1:length(allspin)
        spin_range[cs]=findall(abs.(spins.-allspin[cs]).<1e-6)
        spin_dim[cs]=length(spin_range[cs])
    end


    Vtotal=Rep[SU₂](allspin[1]=>spin_dim[1]);
    for cs=2:length(allspin)
        Vtotal=Vtotal⊕ Rep[SU₂](allspin[cs]=>spin_dim[cs]);
    end
    vtem=space(V_set[1]',1);
    if vtem.dual
        Vtotal=Vtotal';
    end
    Um=TensorMap(randn,space(U_set[1],1)*space(U_set[1],2),Vtotal)*(0*im);
    Vm=Um*0;
    Sm=TensorMap(randn,Vtotal,Vtotal)*(0*im);



    for cs=1:length(allspin)
        Range=spin_range[cs];
        U_block=Um.data.values[cs];
        V_block=Vm.data.values[cs];
        S_block=Sm.data.values[cs];
        
        for ccc=1:spin_dim[cs]
            U=U_set[Range[ccc]];
            U_block[:,ccc]=U.data.values[1];

            V=V_set[Range[ccc]];
            V_block[:,ccc]=V.data.values[1];

            S=S_set[Range[ccc]];
            S_block[ccc,ccc]=S;
        end
        Um.data.values[cs]=U_block*sqrt((2*allspin[cs]+1));
        Vm.data.values[cs]=V_block*sqrt((2*allspin[cs]+1));
        Sm.data.values[cs]=S_block;
    end


    @tensor TT_[:]:=Um[-1,-2,1]*Sm[1,2]*Vm'[2,-3,-4];
    return Um,Sm,Vm
end

Um,Sm,Vm=group_svd_components(U_set,S_set,V_set,spins);
@assert norm(TT-TT_)/norm(TT)<1e-12