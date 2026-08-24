


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


function _shifted_energy(psi, cx, cy, inds, shift)
    psi_shifted=deepcopy(psi);
    T=psi_shifted[cx,cy];
    T[inds...]=T[inds...]+shift;
    psi_shifted[cx,cy]=T;
    # The variational energy is real; discard contraction roundoff in its
    # imaginary part before taking a real-coordinate finite difference.
    return real(compute_E(psi_shifted))
end


"""
    exact_grad(psi; dt=1e-5)

Central finite-difference gradient of `<psi|H|psi>/<psi|psi>` at exactly the
supplied tensor coordinates.  `psi` is treated as a frozen reference point:
this routine never calls `normalize_PEPS!` for either the reference state or
any perturbed state.

For complex tensors, the returned TensorMap stores
`dE/dRe(A) + im*dE/dIm(A)`, matching `vmc_energy_gradient`.
"""
function exact_grad(psi; dt=1e-5)
    dt>0 || throw(ArgumentError("dt must be positive"));
    E0=real(compute_E(psi));
    grad_FD=deepcopy(psi);

    for cx=1:Lx
        for cy=1:Ly
            T=psi[cx,cy];
            grad_FD[cx,cy]=T*0;
            dims=ntuple(dd -> TensorKit.dim(space(T,dd)), Rank(T));
            is_complex=eltype(T) <: Complex;

            for cart_ind in CartesianIndices(dims)
                inds=Tuple(cart_ind);
                E_plus=_shifted_energy(psi,cx,cy,inds,dt);
                E_minus=_shifted_energy(psi,cx,cy,inds,-dt);
                grad_real=(E_plus-E_minus)/(2*dt);

                if is_complex
                    E_plus_im=_shifted_energy(psi,cx,cy,inds,im*dt);
                    E_minus_im=_shifted_energy(psi,cx,cy,inds,-im*dt);
                    grad_imag=(E_plus_im-E_minus_im)/(2*dt);
                    grad_FD[cx,cy][inds...]=grad_real+im*grad_imag;
                else
                    grad_FD[cx,cy][inds...]=grad_real;
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
