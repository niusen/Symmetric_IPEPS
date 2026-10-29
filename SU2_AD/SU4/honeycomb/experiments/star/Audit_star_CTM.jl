include("Measure_star_RVB.jl")
const HCstar=HoneycombSU4;

function identity_bond_checks(T,CTM,setting)
    nx,ny=size(T);
    A=convert_honeycomb_to_iPEPS(T);
    AA=ntuple(x->ntuple(y->first(HCstar.build_double_layer(A[x][y],[])),ny),nx);
    p=space(T[1],5); I2=unitary(p*p,p*p);
    u,s,v=HCstar.tsvd(HCstar.permute(I2,(1,3,),(2,4,)));
    O1=HCstar.permute(u*s,(3,1,),(2,)); O2=HCstar.permute(v,(1,2,),(3,));
    values=ComplexF64[];
    for b in honeycomb_bonds(nx,ny)
        x,y=b.site1;cy=mod1(2-y,ny);
        value=b.axis==:horizontal ?
            HCstar.hopping_x_no_sign(CTM,O1,O2,A,AA,x-1,cy-1,setting) :
            HCstar.hopping_y_no_sign(CTM,O1,O2,A,AA,x-2,cy-1,setting);
        push!(values,value);
    end
    return maximum(abs.(values.-1));
end

function audit_star_CTM()
    BLAS.set_num_threads(4);
    # Zero-based centers (0,0),(0,1),(2,2),(2,3), translated to Julia indices.
    centers=Set([(1,1),(1,2),(3,3),(3,4)]);
    product,info=initial_star_RVB(Lx=4,Ly=4,centers=centers);
    full,_=initial_star_RVB(Lx=4,Ly=4);
    setting=LS_CTMRG_settings(CTM_ite_nums=100,CTM_conv_tol=1e-8,
        CTM_ite_info=true,CTM_conv_info=true,grad_checkpoint=false);
    out=joinpath(@__DIR__,"../../data","star_RVB");mkpath(out);
    jldsave(joinpath(out,"initial_tensors_L4.jld2");product,full,centers,info);
    for ch in (16,32,48)
        println("AUDIT START product chi=",ch);flush(stdout);
        ob=honeycomb_energy_CTM(Dict("J"=>1.),product,ch,setting;convention=:permutation);
        expected=[(b.site1 in centers || b.site2 in centers) ? -1. : .25
                  for b in ob.bonds];
        product_error=maximum(abs.(ob.energies.-expected));
        id_error=identity_bond_checks(product,ob.CTM,setting);
        row=(kind=:product,chi=ch,E_P=ob.energy_per_site,product_bond_error=product_error,
            identity_error=id_error,corner_error=ob.corner_error,iterations=ob.iterations);
        println("AUDIT RESULT ",row," bonds=",ob.energies);flush(stdout);
        jldsave(joinpath(out,"audit_product_chi$(ch).jld2");row,T=product,CTM=ob.CTM);
        if product_error<1e-6 && id_error<1e-8
            println("AUDIT START full warm start chi=",ch);flush(stdout);
            full_ob=honeycomb_energy_CTM(Dict("J"=>1.),full,ch,setting;
                convention=:permutation,CTM0=ob.CTM);
            full_id=identity_bond_checks(full,full_ob.CTM,setting);
            fullrow=(kind=:full_warm,chi=ch,E_P=full_ob.energy_per_site,
                identity_error=full_id,corner_error=full_ob.corner_error,
                min_P=minimum(full_ob.energies),max_P=maximum(full_ob.energies),
                iterations=full_ob.iterations);
            println("AUDIT RESULT ",fullrow," bonds=",full_ob.energies);flush(stdout);
            jldsave(joinpath(out,"audit_full_warm_chi$(ch).jld2");row=fullrow,T=full,CTM=full_ob.CTM);
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    audit_star_CTM();
end
