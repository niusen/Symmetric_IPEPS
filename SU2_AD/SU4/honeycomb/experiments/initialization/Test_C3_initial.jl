using LinearAlgebra, Random, TensorKit, SUNRepresentations, JLD2, Test
include("Test_small_D_SU4.jl")

"""C3-invariant three-leg site map with a symmetric SU(4) singlet bond.
This constrains the INITIAL state only, not the subsequent SU sweeps.
"""
function initial_C3_SU4(V;seed=1234)
    rng=MersenneTwister(seed);
    p=Rep[SU₄]("4"=>1); o=Rep[SU₄]("1"=>1);
    W=p←V'*V'*V';
    U=TensorMap(randn(rng,dim(W)),W);
    rotate(A)=TensorKit.permute(A,((1,),(3,4,2)));
    U=(U+rotate(U)+rotate(rotate(U)))/3;
    U/=norm(U);
    c3_error=norm(U-rotate(U));
    # TensorKit's canonical duality map is not itself the symmetric bond for
    # 4+4bar. Fix the relative phase of the conjugate block explicitly.
    F=unitary(V',V);
    bar4=first(sectors(Rep[SU₄]("4⁺"=>1)));
    TensorKit.block(F,bar4).*= -1;
    B=TensorKit.permute(F,((1,2),()));
    bond_error=norm(B-TensorKit.permute(B,((2,1),())));
    A=TensorKit.permute(U,((2,3,4,1),()));
    trivial=TensorMap(ones(1),o,one(o*o));
    trivial_up=TensorMap(ones(1),o',one(o*o));
    @tensor even[:]:=A[-1,-2,a,-5]*B[-3,a]*trivial_up[-4];
    @tensor odd[:]:=A[-1,a,b,-5]*B[-3,a]*B[-4,b]*trivial[-2];
    even=(1+0im)*even/norm(even); odd=(1+0im)*odd/norm(odd);
    T=Matrix{Any}(undef,2,2); lx=similar(T); ly=similar(T);
    for x=1:2,y=1:2
        T[x,y]=copy(iseven(x+y) ? even : odd);
        lx[x,y]=DiagonalTensorMap(unitary(V,V));
        vd=iseven(x+y) ? V : o;
        ly[x,y]=DiagonalTensorMap(unitary(vd',vd'));
    end
    return T,lx,ly,(c3_error=c3_error,bond_reflection_error=bond_error)
end

function run_initialization_control()
    BLAS.set_num_threads(1);
    out=joinpath(@__DIR__,"../../data","initialization_control");mkpath(out);
    path=joinpath(out,"evolution.csv");
    write(path,"initialization,stage,dt,tau,E_P_local,delta_E,torus_norm,min_D,max_D,P_x,P_y,P_z\n");
    V=honeycomb_virtual_space(["1"=>1,"4"=>1,"6"=>1,"4⁺"=>1]);
    parameters=Dict("J"=>1.0);
    for initialization in (:C3,:random)
        if initialization==:C3
            T,lx,ly,info=initial_C3_SU4(V);
            @test info.c3_error<1e-12;
            @test info.bond_reflection_error<1e-12;
        else
            T,lx,ly=initial_honeycomb_iPEPS(2,2;symmetry=:SU4,V=V,seed=1234);
            info=(c3_error=NaN,bond_reflection_error=NaN);
        end
        ob=honeycomb_energy_simple(parameters,T,lx,ly);
        if initialization==:C3
            @test maximum(values(ob.bond_energy))-minimum(values(ob.bond_energy))<1e-11;
        end
        println("INITIAL ",initialization," ",info," E(P)=",ob.energy_per_site+3/8,
            " bonds=",ob.bond_energy," norm=",torus_norm(T));flush(stdout);
        previous=ob.energy_per_site; tau=0.0; history=NamedTuple[];
        for (stage,(steps,dt)) in enumerate(((120,0.02),(120,0.01),(240,0.005),(300,0.002)))
            itebd_honeycomb(parameters,T,lx,ly,steps*dt,dt,15;verbose=false);
            ob=honeycomb_energy_simple(parameters,T,lx,ly);tau+=steps*dt;
            dims=[dim(space(s,1)) for s in real_bond_spectra(lx,ly)];
            row=(initialization=initialization,stage=stage,dt=dt,tau=tau,
                E_P_local=ob.energy_per_site+3/8,delta_E=abs(ob.energy_per_site-previous),
                torus_norm=torus_norm(T),min_D=minimum(dims),max_D=maximum(dims),
                P_x=ob.bond_energy.x+1/4,P_y=ob.bond_energy.y+1/4,P_z=ob.bond_energy.z+1/4);
            push!(history,row);
            open(path,"a") do io
                println(io,join(values(row),","));
            end
            println("GENTLE ",row);flush(stdout);
            jldsave(joinpath(out,"$(initialization).jld2");T_set=T,lambdax_set=lx,
                lambday_set=ly,parameters,history,info);
            previous=ob.energy_per_site;
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_initialization_control();
end
