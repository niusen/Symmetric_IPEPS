




function H_triangle(parameters, plaquatte, sites, Lx,Ly)
    J1=parameters["J1"];
    J2=parameters["J2"];
    Jchi=parameters["Jchi"];

    # J1=1;
    # J2=0;
    # Jchi=0;

    x_range=[plaquatte[1],plaquatte[1]+1];
    y_range=[plaquatte[2],plaquatte[2]+1];

    if (1<x_range[1])&(x_range[2]<Lx)
        xp="bulk";
    elseif (x_range[1]==1)
        xp="left";
    elseif (x_range[2]==Lx)
        xp="right";
    end

    if (1<y_range[1])&(y_range[2]<Ly)
        yp="bulk";
    elseif (y_range[1]==1)
        yp="bot";
    elseif (y_range[2]==Ly)
        yp="top";
    end

    if xp=="bulk"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="top"
            J_12=J1;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1;
            J_41=J1/2;
        end
    elseif xp=="left"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1;
        elseif yp=="top"
            J_12=J1;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1;
            J_41=J1;
        end
    elseif xp=="right"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="top"
            J_12=J1;
            J_23=J1;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1;
            J_34=J1;
            J_41=J1/2;
        end
    end

    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");
    Id=unitary(space(H_Heisenberg,1),space(H_Heisenberg,1));
    @tensor op_12[:]:=H_Heisenberg[-1,-2,-4,-5]*Id[-3,-6];
    @tensor op_13[:]:=H_Heisenberg[-1,-3,-4,-6]*Id[-2,-5];
    @tensor op_23[:]:=H_Heisenberg[-2,-3,-5,-6]*Id[-1,-4];
    @tensor op_123[:]:=H123chiral[-1,-2,-3,-4,-5,-6];


    if sites=="123"
        H=J_12/2*op_12+J_23/2*op_23+J2/2*op_13+Jchi*op_123;
    elseif sites=="234"
        H=J_23/2*op_12+J_34/2*op_23+J2/2*op_13+Jchi*op_123;
    elseif sites=="341"
        H=J_34/2*op_12+J_41/2*op_23+J2/2*op_13+Jchi*op_123;
    elseif sites=="412"
        H=J_41/2*op_12+J_12/2*op_23+J2/2*op_13+Jchi*op_123;
    end

    
    # H=2*unitary(codomain(op_123),domain(op_123));
    #H=op_12
    H=permute(H,(1,2,3,),(4,5,6,));
    return H
end

function trotter_gate(H,dt)
    @assert norm(H-H')/norm(H)<1e-14
    eu,ev=eigh(H);
    @assert norm(ev*eu*ev'-H)/norm(H)<1e-14 
    gate=ev*exp(-dt*eu)*ev';
    gate_half=ev*exp(-dt*eu/2)*ev';

    #gate=unitary(codomain(H),domain(H))-dt*H+(-dt*H)*(-dt*H)/2    

    return gate, gate_half
end


function get_triangles(parameters,Lx,Ly,dt)
    triangle_set_set=Vector{Any}(undef,0);
    for cs=1:4#four sites in a plaquatte
        if cs==1
            sites="123"
        elseif cs==2
            sites="234"
        elseif cs==3
            sites="341"
        elseif cs==4
            sites="412"
        end

        triangle_set=Vector{Any}(undef,0);
        for c1=1:Lx-1
            for c2=1:Ly-1
                if mod(c1,2)==0 #type 1 triangles
                    if mod(c1+c2,2)==0
                        hh=H_triangle(parameters, [c1,c2], sites, Lx,Ly);
                        hh=permute(hh,(1,2,3,),(4,5,6,));
                        gate,gate_half=trotter_gate(hh,dt);
                        triangle=Dict{String, Any}([("plaquatte",[c1,c2]),("sites",sites),("H",hh),("gate",gate),("gate_half",gate_half)]);
                        triangle_set=vcat(triangle_set,triangle);
                    end
                end
            end
        end
        for c1=1:Lx-1
            for c2=1:Ly-1
                if mod(c1,2)==1 #type 2 triangles
                    if mod(c1+c2,2)==0
                        hh=H_triangle(parameters, [c1,c2], sites, Lx,Ly);
                        hh=permute(hh,(1,2,3,),(4,5,6,));
                        gate,gate_half=trotter_gate(hh,dt);
                        triangle=Dict{String, Any}([("plaquatte",[c1,c2]),("sites",sites),("H",hh),("gate",gate),("gate_half",gate_half)]);
                        triangle_set=vcat(triangle_set,triangle);
                    end
                end
            end
        end
        triangle_set_set=vcat(triangle_set_set,Tuple(triangle_set,));

        triangle_set=Vector{Any}(undef,0);
        for c1=1:Lx-1
            for c2=1:Ly-1
                if mod(c1,2)==0 #type 1 triangles
                    if mod(c1+c2,2)==1
                        hh=H_triangle(parameters, [c1,c2], sites, Lx,Ly);
                        hh=permute(hh,(1,2,3,),(4,5,6,));
                        gate,gate_half=trotter_gate(hh,dt);
                        triangle=Dict{String, Any}([("plaquatte",[c1,c2]),("sites",sites),("H",hh),("gate",gate),("gate_half",gate_half)]);
                        triangle_set=vcat(triangle_set,triangle);
                    end
                end
            end
        end
        for c1=1:Lx-1
            for c2=1:Ly-1
                if mod(c1,2)==1 #type 2 triangles
                    if mod(c1+c2,2)==1
                        hh=H_triangle(parameters, [c1,c2], sites, Lx,Ly);
                        hh=permute(hh,(1,2,3,),(4,5,6,));
                        gate,gate_half=trotter_gate(hh,dt);
                        triangle=Dict{String, Any}([("plaquatte",[c1,c2]),("sites",sites),("H",hh),("gate",gate),("gate_half",gate_half)]);
                        triangle_set=vcat(triangle_set,triangle);
                    end
                end
            end
        end
        triangle_set_set=vcat(triangle_set_set,Tuple(triangle_set,));
    end
    return triangle_set_set
end




function apply_triangle_op(psi,plaquatte,sites,op)
    psi=deepcopy(psi);
    x_range=[plaquatte[1],plaquatte[1]+1];
    y_range=[plaquatte[2],plaquatte[2]+1];
    T_plaquatte=psi[x_range,y_range];

    if sites=="123"
        T1=T_plaquatte[1,2];
        T2=T_plaquatte[2,2];
        T3=T_plaquatte[2,1];

        u1,s1,v1=tsvd(permute(T1,(1,2,4,),(3,5,)));#L1,D1,U1,   R1,d1
        T1_left=u1;#L1,D1,U1,newbond1
        T1_keep=s1*v1;#newbond1,R1,d1

        u3,s3,v3=tsvd(permute(T3,(4,5,),(1,2,3,)));#U3,d3    L3,D3,R3,   
        T3_keep=u3*s3;#U3,d3,newbond3
        T3_left=v3;#newbond3,L3,D3,R3,  

        @tensor T2_new[:]:=T1_keep[-1,1,-5]*T2[1,2,-3,-4,-6]*T3_keep[2,-7,-2];#newbond1,R1,d1   L2,D2,R2,U2,d2   U3,d3,newbond3  
        @tensor T2_new[:]:=T2_new[-1,-2,-3,-4,1,2,3]*op[-5,-6,-7,1,2,3];#newbond1,newbond3,R2,U2,d1,d2,d3, 
        u1,s1,v1=tsvd(permute(T2_new,(1,5,),(2,3,4,6,7,)));#newbond1,d1,   newbond3,R2,U2,d2,d3, 
        T1_keep=u1;#newbond1,d1,R1,
        @tensor T1_new[:]:=T1_left[-1,-2,-4,1]*T1_keep[1,-5,-3];#L1,D1,U1,newbond1      newbond1,d1,R1,
        T2T3=s1*v1;#L2,newbond3,R2,U2,d2,d3,
        u3,s3,v3=tsvd(permute(T2T3,(1,3,4,5,),(2,6,)));#L2,R2,U2,d2,      newbond3,d3,   
        T2_new=permute(u3*s3,(1,5,2,3,4,));#L2,R2,U2,d2,D2
        T3_keep=v3;#U3,newbond3,d3,
        @tensor T3_new[:]:=T3_keep[-4,1,-5]*T3_left[1,-1,-2,-3];#U3,newbond3,d3,  newbond3,L3,D3,R3,  

        T_plaquatte[1,2]=T1_new;
        T_plaquatte[2,2]=T2_new;
        T_plaquatte[2,1]=T3_new;
    elseif sites=="234"
        T1=T_plaquatte[2,2];
        T2=T_plaquatte[2,1];
        T3=T_plaquatte[1,1];

        u1,s1,v1=tsvd(permute(T1,(1,3,4,),(2,5,)));#L1,R1,U1,newbond1,  newbond1,D1,d1   
        T1_left=u1;#L1,R1,U1,newbond1  
        T1_keep=s1*v1;#newbond1,D1,d1  

        u3,s3,v3=tsvd(permute(T3,(3,5,),(1,2,4,)));#R3,d3,newbond3,  newbond3,L3,D3,U3 
        T3_keep=u3*s3;#R3,d3,newbond3 
        T3_left=v3;#newbond3,L3,D3,U3 

        @tensor T2_new[:]:=T1_keep[-4,1,-5]*T2[2,-2,-3,1,-6]*T3_keep[2,-7,-1];#newbond1,D1,d1,   L2,D2,R2,U2,d2,   R3,d3,newbond3   
        @tensor T2_new[:]:=T2_new[-1,-2,-3,-4,1,2,3]*op[-5,-6,-7,1,2,3];#newbond3,D2,R2,newbond1,d1,d2,d3
        u1,s1,v1=tsvd(permute(T2_new,(4,5,),(1,2,3,6,7,)));#newbond1,d1,    newbond3,D2,R2,d2,d3
        T1_keep=u1;#newbond1,d1,D1,
        @tensor T1_new[:]:=T1_left[-1,-3,-4,1]*T1_keep[1,-5,-2];#L1,R1,U1,newbond1,   newbond1,d1,D1
        T2T3=s1*v1;#U2,newbond3,D2,R2,d2,d3
        u3,s3,v3=tsvd(permute(T2T3,(1,3,4,5,),(2,6,)));#U2,D2,R2,d2,    newbond3,d3
        T2_new=permute(u3*s3,(5,2,3,1,4,));
        T3_keep=v3;#R3,newbond3,d3
        @tensor T3_new[:]:=T3_keep[-3,1,-5]*T3_left[1,-1,-2,-4];#R3,newbond3,d3    newbond3,L3,D3,U3 

        T_plaquatte[2,2]=T1_new;
        T_plaquatte[2,1]=T2_new;
        T_plaquatte[1,1]=T3_new;
    elseif sites=="341"
        T1=T_plaquatte[2,1];
        T2=T_plaquatte[1,1];
        T3=T_plaquatte[1,2];

        u1,s1,v1=tsvd(permute(T1,(1,5,),(2,3,4,)));#L1,d1,newbond1,    newbond1,D1,R1,U1   
        T1_left=v1;#newbond1,D1,R1,U1
        T1_keep=u1*s1;#L1,d1,newbond1,

        u3,s3,v3=tsvd(permute(T3,(1,3,4,),(2,5,)));#L3,R3,U3,newbond3,     newbond3, D3,d3,  
        T3_keep=s3*v3;#newbond3, D3,d3,  
        T3_left=u3;#L3,R3,U3,newbond3,

        @tensor T2_new[:]:=T3_keep[-4,1,-7]*T2[-1,-2,2,1,-6]*T1_keep[2,-5,-3];#newbond3, D3,d3,   L2,D2,R2,U2,d2,   L1,d1,newbond1,  
        @tensor T2_new[:]:=T2_new[-1,-2,-3,-4,1,2,3]*op[-5,-6,-7,1,2,3];#L2,D2,newbond1,newbond3,d1,d2,d3
        u1,s1,v1=tsvd(permute(T2_new,(1,2,4,6,7,),(3,5,)));#L2,D2,newbond3,d2,d3    newbond1,d1,
        T1_keep=v1;#L1,newbond1,d1,
        @tensor T1_new[:]:=T1_left[1,-2,-3,-4]*T1_keep[-1,1,-5];#newbond1,D1,R1,U1     L1,newbond1,d1,
        T2T3=u1*s1;#L2,D2,newbond3,d2,d3,R2
        u3,s3,v3=tsvd(permute(T2T3,(3,5,),(1,2,4,6,)));#newbond3,d3,    L2,D2,d2,R2
        T2_new=permute(s3*v3,(2,3,5,1,4,));#U2,L2,D2,d2,R2
        T3_keep=u3;#newbond3,d3,D3
        @tensor T3_new[:]:=T3_left[-1,-3,-4,1]*T3_keep[1,-5,-2];#L3,R3,U3,newbond3,    newbond3,d3,D3
        
        T_plaquatte[2,1]=T1_new;
        T_plaquatte[1,1]=T2_new;
        T_plaquatte[1,2]=T3_new;
    elseif sites=="412"
        T1=T_plaquatte[1,1];
        T2=T_plaquatte[1,2];
        T3=T_plaquatte[2,2];

        u1,s1,v1=tsvd(permute(T1,(1,2,3,),(4,5,)));#L1,D1,R1,newbond1,  newbond1,U1,d1   
        T1_left=u1;#L1,D1,R1,newbond1
        T1_keep=s1*v1;#newbond1,U1,d1  

        u3,s3,v3=tsvd(permute(T3,(1,5,),(2,3,4,)));#L3,d3,newbond3,  newbond3,D3,R3,U3 
        T3_keep=u3*s3;#L3,d3,newbond3, 
        T3_left=v3;#newbond3,D3,R3,U3 

        @tensor T2_new[:]:=T1_keep[-2,1,-5]*T2[-1,1,2,-4,-6]*T3_keep[2,-7,-3];#newbond1,U1,d1     L2,D2,R2,U2,d2,   L3,d3,newbond3, 
        @tensor T2_new[:]:=T2_new[-1,-2,-3,-4,1,2,3]*op[-5,-6,-7,1,2,3];#L2,newbond1,newbond3,U2,d1,d2,d3
        u1,s1,v1=tsvd(permute(T2_new,(2,5,),(1,3,4,6,7,)));#newbond1,d1,    L2,newbond3,U2,d2,d3
        T1_keep=u1;#newbond1,d1,U1
        @tensor T1_new[:]:=T1_left[-1,-2,-3,1]*T1_keep[1,-5,-4];#L1,D1,R1,newbond1,   newbond1,d1,U1 
        T2T3=s1*v1;#D2,L2,newbond3,U2,d2,d3
        u3,s3,v3=tsvd(permute(T2T3,(1,2,4,5,),(3,6,)));#D2,L2,U2,d2,    newbond3,d3,
        T2_new=permute(u3*s3,(2,1,5,3,4,));#D2,L2,U2,d2,R2
        T3_keep=v3;#L3,newbond3,d3,
        @tensor T3_new[:]:=T3_keep[-1,1,-5]*T3_left[1,-2,-3,-4];#L3,newbond3,d3,    newbond3,D3,R3,U3 

        T_plaquatte[1,1]=T1_new;
        T_plaquatte[1,2]=T2_new;
        T_plaquatte[2,2]=T3_new;
    end

    psi[x_range,y_range]=T_plaquatte;
    return psi
end

function compute_ov(psi1,psi2)
    psi_double=construct_double_layer(psi1,psi2);

    Lx,Ly=size(psi_double);

    #truncation method
    mpo_mps_fun=simple_truncate_to_moddle;
    #construct top and bot environment

    log_coe=0;
    trun_history=[];

    py=1;
    pxa=1;
    pxb=pxa+1;
    posy=py;
    
    if posy>1
        mps_bot=(psi_double[:,1]...,);
        mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
        trun_history=vcat(trun_history,trun_errs);
        for cy=2:min(posy-1,Ly-2)
            mpo=(psi_double[:,cy]...,);
            mps_bot,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
            log_coe=log_coe+log(coe);
            trun_history=vcat(trun_history,trun_errs);
        end
    end


    function treat_mps_top(mps)
        #convert mps_top to normal order
        mps=mps[end:-1:1];
        for cx=2:Lx-1
            mps=mps_update(mps,permute(mps[cx],(2,1,3,)),cx);
        end
        return mps
    end

    if posy<Ly
        mps_top=(psi_double[:,Ly]...,);
        mps_top=pi_rotate_mps(mps_top);
        mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
        trun_history=vcat(trun_history,trun_errs);
        for cy=Ly-1:-1:max(posy+1,3)
            mpo=pi_rotate_mpo((psi_double[:,cy]...,));
            mps_top,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
            log_coe=log_coe+log(coe);
            trun_history=vcat(trun_history,trun_errs);
        end
        mps_top=treat_mps_top(mps_top);
    end

    # println(trun_history)
    ########################################


    mps_up=mps_top;
    mpo=psi_double[:,py+1];
    mps_down=psi_double[:,py];

    @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
    for cc=Lx-1:-1:2+1
        @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
    end
    @tensor Norm[:]:=mps_up[1][8,6]*mpo[1][7,9,6]*mps_down[1][-1,7]*mps_up[2][8,1,2]*mpo[2][9,4,3,2]*mps_down[2][-2,5,4]*envR[1,3,5];

    ov=@tensor Norm[1,1];

    return ov*exp(log_coe)
end

# function compute_ov(psi1,psi2)
#     psi_double=construct_double_layer(psi1,psi2);

#     Lx,Ly=size(psi_double);

#     #truncation method
#     mpo_mps_fun=simple_truncate_to_moddle;
#     #construct top and bot environment


#     py=1;
#     pxa=1;
#     pxb=pxa+1;
#     posy=py;
    
#     if posy>1
#         mps_bot=(psi_double[:,1]...,);
#         for cy=2:min(posy-1,Ly-2)
#             mpo=(psi_double[:,cy]...,);
#             mps_bot,_=apply_mpo(mpo,mps_bot);
#         end
#     end


#     function treat_mps_top(mps)
#         #convert mps_top to normal order
#         mps=mps[end:-1:1];
#         for cx=2:Lx-1
#             mps=mps_update(mps,permute(mps[cx],(2,1,3,)),cx);
#         end
#         return mps
#     end

#     if posy<Ly
#         mps_top=(psi_double[:,Ly]...,);
#         mps_top=pi_rotate_mps(mps_top);
#         for cy=Ly-1:-1:max(posy+1,3)
#             mpo=pi_rotate_mpo((psi_double[:,cy]...,));
#             mps_top,_=apply_mpo(mpo,mps_top);
#         end
#         mps_top=treat_mps_top(mps_top);
#     end

#     # println(trun_history)
#     ########################################

#     mps_up=mps_top;
#     mpo=psi_double[:,py+1];
#     mps_down=psi_double[:,py];

#     @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
#     for cc=Lx-1:-1:2+1
#         @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
#     end
#     @tensor Norm[:]:=mps_up[1][8,6]*mpo[1][7,9,6]*mps_down[1][-1,7]*mps_up[2][8,1,2]*mpo[2][9,4,3,2]*mps_down[2][-2,5,4]*envR[1,3,5];

#     ov=@tensor Norm[1,1];

#     return ov
# end

function verify_energy(parameters,psi)
    psi0=deepcopy(psi);
    ov00=compute_ov(psi0,psi0);
    
    psi=disk_to_torus(psi);
    
    Lx,Ly=size(psi);
    dt=0.1;
    triangle_set_set=get_triangles(parameters,Lx,Ly,dt);
    E_total=0;
    E_product=1;
    for c1=1:length(triangle_set_set)
        triangle_set=triangle_set_set[c1];
        for c2=1:length(triangle_set)
            triangle=triangle_set[c2];
            psi_new=apply_triangle_op(psi,triangle["plaquatte"],triangle["sites"],triangle["gate"]);
            #psi_new=apply_triangle_op(psi,triangle["plaquatte"],triangle["sites"],triangle["H"]);

            # eu,ev=eigh(triangle["H"]);
            # println(eu)


            println(triangle["plaquatte"])
            println(triangle["sites"])


            psi_new=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_new));
            # ov11=compute_ov(psi_new,psi_new);
            ov10=compute_ov(psi_new,psi0);
            println([ov10/ov00])
            E_total=E_total+ov10/ov00;

        end
    end

    return E_total
end

