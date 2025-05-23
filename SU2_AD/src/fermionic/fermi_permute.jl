function permute_neighbour_ind(A,ind1,ind2,total_ind)
    A=deepcopy(A);
    @assert ind1+1==ind2
    if total_ind==2
    elseif total_ind==3
        if (ind1==1)&&(ind2==2)
            gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3]*gate[-1,-2,1,2]; 
            A=permute(A,(2,1,3,));
        elseif (ind1==2)&&(ind2==3)
            gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2]*gate[-2,-3,1,2]; 
            A=permute(A,(1,3,2,));
        end
    elseif total_ind==4
        if (ind1==1)&&(ind2==2)
            gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4]*gate[-1,-2,1,2]; 
            A=permute(A,(2,1,3,4,));
        elseif (ind1==2)&&(ind2==3)
            gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4]*gate[-2,-3,1,2]; 
            A=permute(A,(1,3,2,4,));
        elseif (ind1==3)&&(ind2==4)
            gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2]*gate[-3,-4,1,2]; 
            A=permute(A,(1,2,4,3,));
        end
    elseif total_ind==5
        if (ind1==1)&&(ind2==2)
            gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; 
            A=permute(A,(2,1,3,4,5,));

        elseif (ind1==2)&&(ind2==3)
            gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; 
            A=permute(A,(1,3,2,4,5,));

        elseif (ind1==3)&&(ind2==4)
            gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; 
            A=permute(A,(1,2,4,3,5,));

        elseif (ind1==4)&&(ind2==5)
            gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2]; 
            A=permute(A,(1,2,3,5,4,));

        end
    elseif total_ind==6
        if (ind1==1)&&(ind2==2)
            gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5,-6]*gate[-1,-2,1,2]; 
            A=permute(A,(2,1,3,4,5,6,));

        elseif (ind1==2)&&(ind2==3)
            gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5,-6]*gate[-2,-3,1,2]; 
            A=permute(A,(1,3,2,4,5,6,));

        elseif (ind1==3)&&(ind2==4)
            gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5,-6]*gate[-3,-4,1,2]; 
            A=permute(A,(1,2,4,3,5,6,));

        elseif (ind1==4)&&(ind2==5)
            gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2,-6]*gate[-4,-5,1,2]; 
            A=permute(A,(1,2,3,5,4,6,));
        elseif (ind1==5)&&(ind2==6)
            gate=swap_gate(A,5,6); @tensor A[:]:=A[-1,-2,-3,-4,1,2]*gate[-5,-6,1,2]; 
            A=permute(A,(1,2,3,4,6,5,));

        end
    elseif total_ind==7
        if (ind1==1)&&(ind2==2)
            gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5,-6,-7]*gate[-1,-2,1,2]; 
            A=permute(A,(2,1,3,4,5,6,7,));
        elseif (ind1==2)&&(ind2==3)
            gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5,-6,-7]*gate[-2,-3,1,2]; 
            A=permute(A,(1,3,2,4,5,6,7,));
        elseif (ind1==3)&&(ind2==4)
            gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5,-6,-7]*gate[-3,-4,1,2]; 
            A=permute(A,(1,2,4,3,5,6,7,));
        elseif (ind1==4)&&(ind2==5)
            gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2,-6,-7]*gate[-4,-5,1,2]; 
            A=permute(A,(1,2,3,5,4,6,7,));
        elseif (ind1==5)&&(ind2==6)
            gate=swap_gate(A,5,6); @tensor A[:]:=A[-1,-2,-3,-4,1,2,-7]*gate[-5,-6,1,2]; 
            A=permute(A,(1,2,3,4,6,5,7,));
        elseif (ind1==6)&&(ind2==7)
            gate=swap_gate(A,6,7); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,2]*gate[-6,-7,1,2]; 
            A=permute(A,(1,2,3,4,5,7,6,));
        end

    elseif total_ind==8
        if (ind1==1)&&(ind2==2)
            gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5,-6,-7,-8]*gate[-1,-2,1,2]; 
            A=permute(A,(2,1,3,4,5,6,7,8,));
        elseif (ind1==2)&&(ind2==3)
            gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5,-6,-7,-8]*gate[-2,-3,1,2]; 
            A=permute(A,(1,3,2,4,5,6,7,8,));
        elseif (ind1==3)&&(ind2==4)
            gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5,-6,-7,-8]*gate[-3,-4,1,2]; 
            A=permute(A,(1,2,4,3,5,6,7,8,));
        elseif (ind1==4)&&(ind2==5)
            gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2,-6,-7,-8]*gate[-4,-5,1,2]; 
            A=permute(A,(1,2,3,5,4,6,7,8));
        elseif (ind1==5)&&(ind2==6)
            gate=swap_gate(A,5,6); @tensor A[:]:=A[-1,-2,-3,-4,1,2,-7,-8]*gate[-5,-6,1,2]; 
            A=permute(A,(1,2,3,4,6,5,7,8));
        elseif (ind1==6)&&(ind2==7)
            gate=swap_gate(A,6,7); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,2,-8]*gate[-6,-7,1,2]; 
            A=permute(A,(1,2,3,4,5,7,6,8));
        elseif (ind1==7)&&(ind2==8)
            gate=swap_gate(A,7,8); @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,1,2]*gate[-7,-8,1,2]; 
            A=permute(A,(1,2,3,4,5,6,8,7));
        end

    elseif total_ind==9
        if (ind1==1)&&(ind2==2)
            gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5,-6,-7,-8,-9]*gate[-1,-2,1,2]; 
            A=permute(A,(2,1,3,4,5,6,7,8,9,));
        elseif (ind1==2)&&(ind2==3)
            gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5,-6,-7,-8,-9]*gate[-2,-3,1,2]; 
            A=permute(A,(1,3,2,4,5,6,7,8,9,));
        elseif (ind1==3)&&(ind2==4)
            gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5,-6,-7,-8,-9]*gate[-3,-4,1,2]; 
            A=permute(A,(1,2,4,3,5,6,7,8,9,));
        elseif (ind1==4)&&(ind2==5)
            gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2,-6,-7,-8,-9]*gate[-4,-5,1,2]; 
            A=permute(A,(1,2,3,5,4,6,7,8,9,));
        elseif (ind1==5)&&(ind2==6)
            gate=swap_gate(A,5,6); @tensor A[:]:=A[-1,-2,-3,-4,1,2,-7,-8,-9]*gate[-5,-6,1,2]; 
            A=permute(A,(1,2,3,4,6,5,7,8,9,));
        elseif (ind1==6)&&(ind2==7)
            gate=swap_gate(A,6,7); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,2,-8,-9]*gate[-6,-7,1,2]; 
            A=permute(A,(1,2,3,4,5,7,6,8,9,));
        elseif (ind1==7)&&(ind2==8)
            gate=swap_gate(A,7,8); @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,1,2,-9]*gate[-7,-8,1,2]; 
            A=permute(A,(1,2,3,4,5,6,8,7,9,));
        elseif (ind1==8)&&(ind2==9)
            gate=swap_gate(A,8,9); @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,-7,1,2]*gate[-8,-9,1,2]; 
            A=permute(A,(1,2,3,4,5,6,7,9,8,));
        end

    elseif total_ind==10
        if (ind1==1)&&(ind2==2)
            gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5,-6,-7,-8,-9,-10]*gate[-1,-2,1,2]; 
            A=permute(A,(2,1,3,4,5,6,7,8,9,10,));
        elseif (ind1==2)&&(ind2==3)
            gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5,-6,-7,-8,-9,-10]*gate[-2,-3,1,2]; 
            A=permute(A,(1,3,2,4,5,6,7,8,9,10,));
        elseif (ind1==3)&&(ind2==4)
            gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5,-6,-7,-8,-9,-10]*gate[-3,-4,1,2]; 
            A=permute(A,(1,2,4,3,5,6,7,8,9,10,));
        elseif (ind1==4)&&(ind2==5)
            gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2,-6,-7,-8,-9,-10]*gate[-4,-5,1,2]; 
            A=permute(A,(1,2,3,5,4,6,7,8,9,10,));
        elseif (ind1==5)&&(ind2==6)
            gate=swap_gate(A,5,6); @tensor A[:]:=A[-1,-2,-3,-4,1,2,-7,-8,-9,-10]*gate[-5,-6,1,2]; 
            A=permute(A,(1,2,3,4,6,5,7,8,9,10,));
        elseif (ind1==6)&&(ind2==7)
            gate=swap_gate(A,6,7); @tensor A[:]:=A[-1,-2,-3,-4,-5,1,2,-8,-9,-10]*gate[-6,-7,1,2]; 
            A=permute(A,(1,2,3,4,5,7,6,8,9,10,));
        elseif (ind1==7)&&(ind2==8)
            gate=swap_gate(A,6,7); @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,1,2,-9,-10]*gate[-7,-8,1,2]; 
            A=permute(A,(1,2,3,4,5,6,8,7,9,10,));
        elseif (ind1==8)&&(ind2==9)
            gate=swap_gate(A,7,8); @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,-7,1,2,-10]*gate[-8,-9,1,2]; 
            A=permute(A,(1,2,3,4,5,6,7,9,8,10,));
        elseif (ind1==8)&&(ind2==9)
            gate=swap_gate(A,7,8); @tensor A[:]:=A[-1,-2,-3,-4,-5,-6,-7,-8,1,2]*gate[-9,-10,1,2]; 
            A=permute(A,(1,2,3,4,5,6,7,8,10,9,));
        end

    end

    return A
end

function permute_neighbour_ind_Rank6_fast(A,ind1,ind2)
    if (ind1==1)&&(ind2==2)
        #gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5,-6]*gate[-1,-2,1,2]; 
        P_odd,P_even=projector_parity(space(A,1));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=A[1,-2,-3,-4,-5,-6]*P_odd[-1,1];
        P_odd,P_even=projector_parity(space(A,2));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=Anew[-1,1,-3,-4,-5,-6]*P_odd[-2,1];

        A=permute(A-2*Anew,(2,1,3,4,5,6,));

    elseif (ind1==2)&&(ind2==3)
        #gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5,-6]*gate[-2,-3,1,2]; 
        P_odd,P_even=projector_parity(space(A,2));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=A[-1,1,-3,-4,-5,-6]*P_odd[-2,1];
        P_odd,P_even=projector_parity(space(A,3));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=Anew[-1,-2,1,-4,-5,-6]*P_odd[-3,1];

        A=permute(A-2*Anew,(1,3,2,4,5,6,));

    elseif (ind1==3)&&(ind2==4)
        #gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5,-6]*gate[-3,-4,1,2]; 
        P_odd,P_even=projector_parity(space(A,3));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=A[-1,-2,1,-4,-5,-6]*P_odd[-3,1];
        P_odd,P_even=projector_parity(space(A,4));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=Anew[-1,-2,-3,1,-5,-6]*P_odd[-4,1];

        A=permute(A-2*Anew,(1,2,4,3,5,6,));

    elseif (ind1==4)&&(ind2==5)
        #gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2,-6]*gate[-4,-5,1,2]; 
        P_odd,P_even=projector_parity(space(A,4));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=A[-1,-2,-3,1,-5,-6]*P_odd[-4,1];
        P_odd,P_even=projector_parity(space(A,5));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=Anew[-1,-2,-3,-4,1,-6]*P_odd[-5,1];

        A=permute(A-2*Anew,(1,2,3,5,4,6,));
    elseif (ind1==5)&&(ind2==6)
        #gate=swap_gate(A,5,6); @tensor A[:]:=A[-1,-2,-3,-4,1,2]*gate[-5,-6,1,2]; 
        P_odd,P_even=projector_parity(space(A,5));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=A[-1,-2,-3,-4,1,-6]*P_odd[-5,1];
        P_odd,P_even=projector_parity(space(A,6));
        P_odd=P_odd'*P_odd;
        @tensor Anew[:]:=Anew[-1,-2,-3,-4,-5,1]*P_odd[-6,1];

        A=permute(A-2*Anew,(1,2,3,4,6,5,));

    end
    return A
end





function LUdRD_to_LDRUd(T::TensorMap)
    T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
    T=permute_neighbour_ind(T,4,5,5);#L,U,d,D,R,
    T=permute_neighbour_ind(T,3,4,5);#L,U,D,d,R,
    T=permute_neighbour_ind(T,2,3,5);#L,D,U,d,R,
    T=permute_neighbour_ind(T,4,5,5);#L,D,U,R,d,
    T=permute_neighbour_ind(T,3,4,5);#L,D,R,U,d,
    return T
end

function LDRUd_to_LUdRD(T::TensorMap)
    #L,D,R,U,d,
    T=permute_neighbour_ind(T,3,4,5);#L,D,U,R,d,
    T=permute_neighbour_ind(T,2,3,5);#L,U,D,R,d,
    T=permute_neighbour_ind(T,4,5,5);#L,U,D,d,R,
    T=permute_neighbour_ind(T,3,4,5);#L,U,d,D,R,
    T=permute_neighbour_ind(T,4,5,5);#L,U,d,R,D,
    T=permute(T,(1,5,4,2,3,));#L,D,R,U,d
    return T
end



function fermi_rotate(T::TensorMap,deg)
    #anti-clockwise
    if deg==90
        #LDRUd=>DRULd
        T=permute_neighbour_ind(T,1,2,5);#D,L,R,U,d,
        T=permute_neighbour_ind(T,2,3,5);#D,R,L,U,d,
        T=permute_neighbour_ind(T,3,4,5);#D,R,U,L,d,
    elseif deg==180
        #LDRUd=>RULDd
        T=permute_neighbour_ind(T,1,2,5);#D,L,R,U,d,
        T=permute_neighbour_ind(T,2,3,5);#D,R,L,U,d,
        T=permute_neighbour_ind(T,3,4,5);#D,R,U,L,d,
        T=permute_neighbour_ind(T,1,2,5);#R,D,U,L,d,
        T=permute_neighbour_ind(T,2,3,5);#R,U,D,L,d,
        T=permute_neighbour_ind(T,3,4,5);#R,U,L,D,d,
    elseif deg==270
        #LDRUd=>ULDRd
        T=permute_neighbour_ind(T,3,4,5);#L,D,U,R,d,
        T=permute_neighbour_ind(T,2,3,5);#L,U,D,R,d,
        T=permute_neighbour_ind(T,1,2,5);#U,L,D,R,d,
    end
    return T
end