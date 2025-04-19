using Distributed
#number of workers to add and soft restrict of memory
#addprocs(50; exeflags=["--heap-size-hint=6G"])
#addprocs(1; exeflags=["--heap-size-hint=6G"])

@everywhere using LinearAlgebra:I,diagm,diag
@everywhere using TensorKit
@everywhere using Random
@everywhere using Printf
@everywhere using DelimitedFiles
@everywhere using CSV
@everywhere using DataFrames
@everywhere using JLD2,MAT

@everywhere cd(@__DIR__)



@everywhere include("../../../../state/iPEPS_ansatz.jl")
@everywhere include("../../../../setting/Settings.jl")
@everywhere include("../../../../setting/linearalgebra.jl")
@everywhere include("../../../../setting/tuple_methods.jl")
@everywhere include("../../../../environment/MC/finite_clusters.jl")

@everywhere include("../../../../environment/MC/contract_disk.jl")
@everywhere include("../../../../environment/MC/sampling.jl")
@everywhere include("../../../../environment/MC/mps_sweep.jl")

@everywhere begin
@show const Lattice="square";#"kagome", "square"
@show const Lx = 4      # number of sites along x / number of columns in the lattice
@show const Ly = 4      # number of sites along y / number of rows in the lattice
@show const D=2;#bond dimension of state
@show const chi=10;#bond dimension of environment
@show const use_mps_sweep=true;
@show const n_mps_sweep=5;

const L = Lx*Ly # total number of lattice sites
const Nbra = L             # Inner loop size, to generate uncorrelated samples, usually must be of size O(L).
const Ne = L            # Number of electrons on the lattice (for spin models this will always be equal to L)
@show const Nsteps = 600       # Total Monte Carlo steps
@show const binn = 200          # Bin size to store the data during the monte carlo run. 
const GC_spacing = 200          # garbage collection
end

###################
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
dir=hostnm*"_"*string(Lx)*"x"*string(Ly)*"_D"*string(D)*"/";
isdir(dir) || mkdir(dir)
###################

@everywhere include("sq_constants.jl")


####################
#use single core
import LinearAlgebra.BLAS as BLAS
using Base.Threads

n_cpu=1;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);

Base.Sys.set_process_title("C"*string(n_cpu)*"_fPEPS")
pid=getpid();
println("pid="*string(pid));;flush(stdout);
####################







filenm="complex_"*"Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D);
psi,Vp=load_fPEPS(Lx,Ly,filenm);

# psi=add_noise(psi,0,true);



function compute_E(psi)
    coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced=get_neighbours_square(Lx,Ly,"OBC");

    @tensor A_1234[:]:=psi[1,1][1,-1,-5]*psi[2,1][1,2,-2,-6]*psi[3,1][2,3,-3,-7]*psi[4,1][3,-4,-8];

    @tensor A_5678[:]:=psi[1,2][-5,1,-1,-9]*psi[2,2][1,-6,2,-2,-10]*psi[3,2][2,-7,3,-3,-11]*psi[4,2][3,-8,-4,-12];

    @tensor A_9101112[:]:=psi[1,3][-5,1,-1,-9]*psi[2,3][1,-6,2,-2,-10]*psi[3,3][2,-7,3,-3,-11]*psi[4,3][3,-8,-4,-12];

    @tensor A_13141516[:]:=psi[1,4][-1,1,-5]*psi[2,4][1,-2,2,-6]*psi[3,4][2,-3,3,-7]*psi[4,4][3,-4,-8];
        
    @tensor A_total[:]:=A_13141516[1,2,3,4,-13,-14,-15,-16]*A_9101112[1,2,3,4,5,6,7,8,-9,-10,-11,-12]*A_5678[5,6,7,8,9,10,11,12,-5,-6,-7,-8]*A_1234[9,10,11,12,-1,-2,-3,-4];

    sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
    @tensor H_Heisenberg[:]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4];
    H_Heisenberg=TensorMap(H_Heisenberg,Vp*Vp,  Vp*Vp);


    psi_projected=deepcopy(A_total);
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


    E=0;
    for cn=1:length(NN_tuple_reduced)
        for ct in NN_tuple_reduced[cn]
            link=sort([cn,ct]);
            order=Tuple(vcat(link[1],link[2],1:link[1]-1,link[1]+1:link[2]-1,link[2]+1:16));
            #@show order
            psi_=permute(psi_projected,Tuple(vcat(link[1],link[2],1:link[1]-1,link[1]+1:link[2]-1,link[2]+1:16)));
            @tensor rho[:]:=psi_'[-1,-2,1,2,3,4,5,6,7,8,9,10,11,12,13,14]*psi_[-3,-4,1,2,3,4,5,6,7,8,9,10,11,12,13,14];
            E_=@tensor rho[3,4,1,2]*H_Heisenberg[3,4,1,2];
            Norm=@tensor rho[1,2,1,2];
            E=E+E_/Norm;
        end
    end
    return E
end

E0=compute_E(psi)

grad_FD=deepcopy(psi);
dt=0.00001;
for cx=1:Lx
    for cy=1:Ly
        grad_FD[cx,cy]=grad_FD[cx,cy]*0;
        T=grad_FD[cx,cy]
        if Rank(T)==3
            D1=TensorKit.dim(space(T,1));
            D2=TensorKit.dim(space(T,2));
            D3=TensorKit.dim(space(T,3));
            for d1=1:D1
                for d2=1:D1
                    for d3=1:D3
                        psi_=deepcopy(psi);
                        tt=psi_[cx,cy];
                        tt[d1,d2,d3]=tt[d1,d2,d3]+dt;
                        psi_[cx,cy]=tt;
                        Enew=compute_E(psi_);
                        Re=(Enew-E0)/dt;

                        psi_=deepcopy(psi);
                        tt=psi_[cx,cy];
                        tt[d1,d2,d3]=tt[d1,d2,d3]+dt*im;
                        psi_[cx,cy]=tt;
                        Enew=compute_E(psi_);
                        Im=(Enew-E0)/dt;

                        grad_FD[cx,cy][d1,d2,d3]=Re+im*Im;
                    end
                end
            end
            
        elseif Rank(T)==4
            D1=TensorKit.dim(space(T,1));
            D2=TensorKit.dim(space(T,2));
            D3=TensorKit.dim(space(T,3));
            D4=TensorKit.dim(space(T,4));
            for d1=1:D1
                for d2=1:D1
                    for d3=1:D3
                        for d4=1:D4
                            psi_=deepcopy(psi);
                            tt=psi_[cx,cy];
                            tt[d1,d2,d3,d4]=tt[d1,d2,d3,d4]+dt;
                            psi_[cx,cy]=tt;
                            Enew=compute_E(psi_);
                            Re=(Enew-E0)/dt;

                            psi_=deepcopy(psi);
                            tt=psi_[cx,cy];
                            tt[d1,d2,d3,d4]=tt[d1,d2,d3,d4]+im*dt;
                            psi_[cx,cy]=tt;
                            Enew=compute_E(psi_);
                            Im=(Enew-E0)/dt;

                            grad_FD[cx,cy][d1,d2,d3,d4]=Re+im*Im;
                        end
                    end
                end
            end
        elseif Rank(T)==5
            D1=TensorKit.dim(space(T,1));
            D2=TensorKit.dim(space(T,2));
            D3=TensorKit.dim(space(T,3));
            D4=TensorKit.dim(space(T,4));
            D5=TensorKit.dim(space(T,5));
            for d1=1:D1
                for d2=1:D1
                    for d3=1:D3
                        for d4=1:D4
                            for d5=1:D5
                                psi_=deepcopy(psi);
                                tt=psi_[cx,cy];
                                tt[d1,d2,d3,d4,d5]=tt[d1,d2,d3,d4,d5]+dt;
                                psi_[cx,cy]=tt;
                                Enew=compute_E(psi_);
                                Re=(Enew-E0)/dt;

                                psi_=deepcopy(psi);
                                tt=psi_[cx,cy];
                                tt[d1,d2,d3,d4,d5]=tt[d1,d2,d3,d4,d5]+im*dt;
                                psi_[cx,cy]=tt;
                                Enew=compute_E(psi_);
                                Im=(Enew-E0)/dt;

                                grad_FD[cx,cy][d1,d2,d3,d4,d5]=Re+im*Im;
                            end
                        end
                    end
                end
            end
        end

    end
end



#jldsave("grad_FD.jld2";grad_FD)




data=load("grad_4x4_D2_chi10.jld2");
Grad=data["Grad"];

ov_set=zeros(Lx,Ly)*im;
for cx=1:Lx
    for cy=1:Ly
        if Rank(grad_FD[cx,cy])==3
            ov_set[cx,cy]=dot(permute(grad_FD[cx,cy]',(1,2,3,)),Grad[cx,cy])/sqrt(dot(grad_FD[cx,cy],grad_FD[cx,cy])*dot(Grad[cx,cy],Grad[cx,cy]));
        elseif Rank(grad_FD[cx,cy])==4
            ov_set[cx,cy]=dot(permute(grad_FD[cx,cy]',(1,2,3,4,)),Grad[cx,cy])/sqrt(dot(grad_FD[cx,cy],grad_FD[cx,cy])*dot(Grad[cx,cy],Grad[cx,cy]));
        elseif Rank(grad_FD[cx,cy])==5
            ov_set[cx,cy]=dot(permute(grad_FD[cx,cy]',(1,2,3,4,5,)),Grad[cx,cy])/sqrt(dot(grad_FD[cx,cy],grad_FD[cx,cy])*dot(Grad[cx,cy],Grad[cx,cy]));
        end
        
    end
end

@show ov_set