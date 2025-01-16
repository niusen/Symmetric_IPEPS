using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
using LineSearches,OptimKit
cd(@__DIR__)



include("../../../../environment/MC/sampling.jl")

filenm="Heisenberg_SU_4x4_D2.jld2";


J1=1;
J2=0;
Jchi=0;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
global parameters



"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""





data=load(filenm);
psi=data["psi"];










#########################################
@tensor mps_top[:]:=psi[1,4][-5,1,-1]*psi[2,4][1,-6,2,-2]*psi[3,4][2,-7,3,-3]*psi[4,4][3,-8,-4];
@tensor mps_top[:]:=mps_top[-1,-2,-3,-4,1,3,5,7]*psi[1,3][-9,2,1,-5]*psi[2,3][2,-10,4,3,-6]*psi[3,3][4,-11,6,5,-7]*psi[4,3][6,-12,7,-8];


@tensor mps_bot[:]:=psi[1,1][1,-1,-5]*psi[2,1][1,2,-2,-6]*psi[3,1][2,3,-3,-7]*psi[4,1][3,-4,-8];
@tensor mps_bot[:]:=mps_bot[1,3,5,7,-9,-10,-11,-12]*psi[1,2][1,2,-1,-5]*psi[2,2][2,3,4,-2,-6]*psi[3,2][4,5,6,-3,-7]*psi[4,2][6,7,-4,-8];

@tensor psi_total[:]:=mps_top[-1,-2,-3,-4,-5,-6,-7,-8,1,2,3,4]*mps_bot[1,2,3,4,-9,-10,-11,-12,-13,-14,-15,-16];
##############################################

psi_projected=deepcopy(psi_total);
for c1=1:2
    for c2=1:2
        for c3=1:2
            for c4=1:2
                for c5=1:2
                    for c6=1:2
                        for c7=1:2
                            for c8=1:2
                                for c9=1:2
                                    for c10=1:2
                                        for c11=1:2
                                            for c12=1:2
                                                for c13=1:2
                                                    for c14=1:2
                                                        for c15=1:2
                                                            for c16=1:2
                                                                if c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11+c12+c13+c14+c15+c16==(1+2)*8
                                                                else
                                                                    psi_projected[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16]=0
                                                                end

                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
@tensor H_Heisenberg[:]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4];
H_Heisenberg=TensorMap(H_Heisenberg,(ℂ^2)' ⊗ (ℂ^2)', (ℂ^2)' ⊗ (ℂ^2)');

Lx=4;
Ly=4;
coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours(Lx,Ly,"OBC");



E_total=0;
for cc=1:Lx*Ly
    for pp in NN_tuple_reduced[cc]
        # @show cc,pp
        if cc<pp
            poss=[cc,pp];
        elseif pp<cc
            poss=[pp,cc];
        end
        order1=vcat(1:poss[1]-1,poss[1]+1:poss[2]-1,poss[2]+1:Lx*Ly);
        psi_=permute(psi_total,Tuple(order1),Tuple(poss));
        rho=psi_'*psi_;
        Norm=@tensor rho[1,2,1,2];
        e0=@tensor rho[1,2,3,4]*H_Heisenberg[1,2,3,4];
        E_total+=e0/Norm;
    end
end
@show E_total


E_sz0=0;
for cc=1:Lx*Ly
    for pp in NN_tuple_reduced[cc]
        # @show cc,pp
        if cc<pp
            poss=[cc,pp];
        elseif pp<cc
            poss=[pp,cc];
        end
        order1=vcat(1:poss[1]-1,poss[1]+1:poss[2]-1,poss[2]+1:Lx*Ly);
        psi_=permute(psi_projected,Tuple(order1),Tuple(poss));
        rho=psi_'*psi_;
        Norm=@tensor rho[1,2,1,2];
        e0=@tensor rho[1,2,3,4]*H_Heisenberg[1,2,3,4];
        E_sz0+=e0/Norm;
    end
end
@show E_sz0


jldsave(filenm;psi,E_sz0=E_sz0,E=E_total);