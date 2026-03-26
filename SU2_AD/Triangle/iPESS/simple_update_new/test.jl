t=1;
ϕ=pi/2;
μ=0;
U=0;
B=0;
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ), ("U",  U), ("B",  B)]);

Lx=2;
Ly=2;
dt=1;

Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_Z2();
sx_op,sy_op,sz_op=spin_operator_Z2();


parameters_site=@ignore_derivatives get_Hofstadter_coefficients(Lx,Ly,parameters,energy_setting);
tx_coe_set=parameters_site["tx_coe_set"]/2;
ty_coe_set=parameters_site["ty_coe_set"]/2;
t2_coe_set=parameters_site["t2_coe_set"]/2;
U_coe_set=parameters_site["U_coe_set"]/6;
μ_coe_set=parameters_site["μ_coe_set"]/6;


B_coe=parameters["B"]/6;
if abs(B_coe)>0
    @assert mod(Lx,3)==0;
    @assert mod(Ly,3)==0;
end

hx_set=Matrix{TensorMap}(undef,Lx,Ly);
hy_set=Matrix{TensorMap}(undef,Lx,Ly);
h2_set=Matrix{TensorMap}(undef,Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        ####################
        O1=Cdag_set[mod1(cy+1,2)];
        O2=C_set[mod1(cy+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*tx_coe_set[cx,cy];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(Cdag_set[mod1(cy,2)],2),space(Cdag_set[mod1(cy,2)],2));
        @tensor hh_tx[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        hx_set[cx,cy]=hh_tx;
        ######################
        O1=Cdag_set[mod1(cy,2)];
        O2=C_set[mod1(cy+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*ty_coe_set[cx,cy]';#be careful about the order of sites here
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(Cdag_set[mod1(cy+1,2)],2),space(Cdag_set[mod1(cy+1,2)],2));
        @tensor hh_ty[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        hy_set[cx,cy]=hh_ty;
        #####################
        O1=Cdag_set[mod1(cy+1,2)];
        O2=C_set[mod1(cy,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=-op;#!!!!!!! somehow this minus sign is required
        op=op*t2_coe_set[cx,cy];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(Cdag_set[mod1(cy+1,2)],2),space(Cdag_set[mod1(cy+1,2)],2));
        @tensor hh[:]:=hh[-1,-3,-4,-6]*Id[-2,-5];
        sgate=swap_gate(hh,2,3);
        @tensor hh_t2[:]:=sgate[-2,-3,1,2]*hh[-1,1,2,-4,3,4]*sgate'[3,4,-5,-6];
        h2_set[cx,cy]=hh_t2;
        #################
        # OU_LD=n_double_set[mod1(cy+1,2)]-(1/2)*N_occu_set[mod1(cy+1,2)]+(1/4)*Ident_set[mod1(cy+1,2)];
        # OU_RU=n_double_set[mod1(cy,2)]-(1/2)*N_occu_set[mod1(cy,2)]+(1/4)*Ident_set[mod1(cy,2)];
        # OU_RD=n_double_set[mod1(cy+1,2)]-(1/2)*N_occu_set[mod1(cy+1,2)]+(1/4)*Ident_set[mod1(cy+1,2)];
        # Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
        # Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
        # Id_RD=unitary(space(OU_RD,1),space(OU_RD,1));
        # @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
        # @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
        # @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
        # hh_U=(hh_LD+hh_RU+hh_RD)*U_coe_set[cx,cy];
        # #################
        # OU_LD=N_occu_set[mod1(cy+1,2)];
        # OU_RU=N_occu_set[mod1(cy,2)];
        # OU_RD=N_occu_set[mod1(cy+1,2)];
        # Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
        # Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
        # Id_RD=unitary(space(OU_RD,1),space(OU_RD,1));
        # @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
        # @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
        # @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
        # hh_μ=(hh_LD+hh_RU+hh_RD)*μ_coe_set[cx,cy];
        # #################
        # hh=hh_tx+hh_ty+hh_t2+hh_U-hh_μ;
        # #################
        # if abs(B_coe)>0
        #     OU_LD=N_occu_set[mod1(cy+1,2)];
        #     OU_RU=N_occu_set[mod1(cy,2)];
        #     OU_RD=N_occu_set[mod1(cy+1,2)];

        #     coord=[cx,cy+1];
        #     B_field=B_field_fun(Lx,Ly,coord)
        #     B_LD=B_field[1]*sx_op+B_field[2]*sy_op+B_field[3]*sz_op;

        #     coord=[cx+1,cy];
        #     B_field=B_field_fun(Lx,Ly,coord)
        #     B_RU=B_field[1]*sx_op+B_field[2]*sy_op+B_field[3]*sz_op;

        #     coord=[cx+1,cy+1];
        #     B_field=B_field_fun(Lx,Ly,coord)
        #     B_RD=B_field[1]*sx_op+B_field[2]*sy_op+B_field[3]*sz_op;

        #     Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
        #     Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
        #     Id_RD=unitary(space(OU_RD,1),space(OU_RD,1));
        #     @tensor hh_LD[:]:=B_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
        #     @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*B_RU[-3,-6];
        #     @tensor hh_RD[:]:=Id_LD[-1,-4]*B_RD[-2,-5]*Id_RU[-3,-6];
        #     hh_B=(hh_LD+hh_RU+hh_RD)*B_coe;
        #     hh=hh+hh_B;
        # end
        # #################
        # hh=permute(hh,(1,2,3,),(4,5,6,));
        # eu,ev=eigh(hh);
        # gate=ev*exp(-dt*eu)*ev';
        # gate_set[cx,cy]=gate;
    end
end
