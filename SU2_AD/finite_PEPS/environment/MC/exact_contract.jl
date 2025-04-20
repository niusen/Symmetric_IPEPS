


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


function exact_grad(psi);
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
                            if global_eltype==ComplexF64
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
                            elseif global_eltype==Float64
                                psi_=deepcopy(psi);
                                tt=psi_[cx,cy];
                                tt[d1,d2,d3]=tt[d1,d2,d3]+dt;
                                psi_[cx,cy]=tt;
                                Enew=compute_E(psi_);
                                Re=(Enew-E0)/dt;
    
            
    
                                grad_FD[cx,cy][d1,d2,d3]=Re;
                            end
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
                                if global_eltype==ComplexF64
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
                                elseif global_eltype==Float64
                                    psi_=deepcopy(psi);
                                    tt=psi_[cx,cy];
                                    tt[d1,d2,d3,d4]=tt[d1,d2,d3,d4]+dt;
                                    psi_[cx,cy]=tt;
                                    Enew=compute_E(psi_);
                                    Re=(Enew-E0)/dt;


                                    grad_FD[cx,cy][d1,d2,d3,d4]=Re;
                                end
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
                                    if global_eltype==ComplexF64
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
                                    elseif global_eltype==Float64
                                        psi_=deepcopy(psi);
                                        tt=psi_[cx,cy];
                                        tt[d1,d2,d3,d4,d5]=tt[d1,d2,d3,d4,d5]+dt;
                                        psi_[cx,cy]=tt;
                                        Enew=compute_E(psi_);
                                        Re=(Enew-E0)/dt;
    
    
                                        grad_FD[cx,cy][d1,d2,d3,d4,d5]=Re;
                                    end
                                end
                            end
                        end
                    end
                end
            end

        end
    end
    return E0,grad_FD
end



function compare_grad(grad_FD,Grad)
    ov_set=zeros(Lx,Ly)*im;
    for cx=1:Lx
        for cy=1:Ly
            if Rank(grad_FD[cx,cy])==3
                ov_set[cx,cy]=dot(permute(grad_FD[cx,cy],(1,2,3,)),Grad[cx,cy])/sqrt(dot(grad_FD[cx,cy],grad_FD[cx,cy])*dot(Grad[cx,cy],Grad[cx,cy]));
            elseif Rank(grad_FD[cx,cy])==4
                ov_set[cx,cy]=dot(permute(grad_FD[cx,cy],(1,2,3,4,)),Grad[cx,cy])/sqrt(dot(grad_FD[cx,cy],grad_FD[cx,cy])*dot(Grad[cx,cy],Grad[cx,cy]));
            elseif Rank(grad_FD[cx,cy])==5
                ov_set[cx,cy]=dot(permute(grad_FD[cx,cy],(1,2,3,4,5,)),Grad[cx,cy])/sqrt(dot(grad_FD[cx,cy],grad_FD[cx,cy])*dot(Grad[cx,cy],Grad[cx,cy]));
            end
            
        end
    end
    return ov_set
end