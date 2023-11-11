



function plaquatte_empty()
    #the four-site plaquatte is around empty square
    global Lx,Ly
    @assert Lx==2;
    @assert Ly==2;

    println("Generate empty plaquatte state");flush(stdout);
    Vp=SU2Space(1=>1);
    Vv=SU2Space(1/2=>1);
    V0=SU2Space(0=>1);
    dimer1=TensorMap(randn,V0,Vv*Vv)*(1+0*im);
    dimer2=TensorMap(randn,V0',Vv*Vv)*(1+0*im);
    @tensor tetrahedral[:]:=dimer1[1,-1,-2]*dimer2[1,-3,-4];
    tetrahedral_1=tetrahedral/norm(tetrahedral);
    tetrahedral_2=permute(tetrahedral_1,(2,3,4,1,));
    B=TensorMap(randn,Vv*Vv,Vp)*(1+0*im);
    B=B/norm(B);
    B=permute(B,(1,2,3,));

    state=Matrix{Checkerboard_iPESS}(undef,Lx,Ly);
    state[1,1]=Checkerboard_iPESS(B,B,tetrahedral_1);
    state[1,2]=Checkerboard_iPESS(B,B,tetrahedral_2);
    state[2,1]=Checkerboard_iPESS(B,B,tetrahedral_2);
    state[2,2]=Checkerboard_iPESS(B,B,tetrahedral_1);
    return state
end

function plaquatte_cross()
        #the four-site plaquatte is around crossed square
        global Lx,Ly
        @assert Lx==2;
        @assert Ly==2;
    
        println("Generate crossed plaquatte state");flush(stdout);
        Vp=SU2Space(1=>1);
        Vv=SU2Space(1/2=>1);
        V0=SU2Space(0=>1);

        B1=TensorMap(randn,Vp*V0,Vp)*(1+0*im);
        B1=B1/norm(B1);
        B1=permute(B1,(1,2,3,));
        B2=permute(B1,(2,1,3,));

        B=TensorMap(randn,Vv*Vv',Vp)*(1+0*im);
        B=permute(B,(1,2,3,));
        @tensor AKLT_loop[:]:=B[1,2,-1]*B[2,3,-2]*B[3,4,-3]*B[4,1,-4];

        spin_zero_loop=TensorMap(randn,V0',V0*V0*V0)*(1+0*im);

    
        state=Matrix{Checkerboard_iPESS}(undef,Lx,Ly);
        state[1,1]=Checkerboard_iPESS(B2,B2,AKLT_loop);
        state[1,2]=Checkerboard_iPESS(B1,B1,spin_zero_loop);
        state[2,1]=Checkerboard_iPESS(B1,B1,spin_zero_loop);
        state[2,2]=Checkerboard_iPESS(B2,B2,AKLT_loop);
        return state
end