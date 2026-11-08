# Standalone integration test: portable D4 seed plus tuple/array equivalence.
# test_saved_u1.jl establishes V, N, f, env, file and before using the existing
# portable seed. No production evolution or large-D calculation is performed.
ENV["KAGOME_VARIANCE_NATIVE_ROOT"]=get(ENV,"KAGOME_VARIANCE_NATIVE_ROOT",
    normpath(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","native_measurement")))
include("test_saved_u1.jl")
cfg=V.C.LS_CTMRG_settings(CTM_trun_tol=1e-12,grad_checkpoint=false,use_sub_checkpoint=false)
array_errors=Float64[]
@testset "saved D4 tuple vs array: every corner and edge, all four directions" begin
    for orientation in (:up,:down)
        ctx,cq,cr=V.central_context(f,env,3,1,1,orientation,2)
        Q=V.C.LayerGrid(V.tuple_grid(ctx.Q));R=V.C.LayerGrid(V.tuple_grid(ctx.R))
        cs=V.tuple_grid(ctx.Cset);ts=V.tuple_grid(ctx.Tset)
        ac=V.measurement_grid(ctx.Cset,3);at=V.measurement_grid(ctx.Tset,3)
        V.set_dimensions(3,3)
        for d in (3,4,1,2)
            cs,ts=V.C.ctm_update_single_cx(8,1,3,cs,ts,Q,R,d,cfg,3,3)
            V.variance_column_update!(ac,at,Q,R,8,1,d,cfg,3)
            for x in 1:3,y in 1:3,k in 1:4
                for (a,b) in ((V.C.get_Cset(cs[x][y],k),V.C.get_Cset(ac[x][y],k)),
                              (V.C.get_Tset(ts[x][y],k),V.C.get_Tset(at[x][y],k)))
                    err=norm(a-b)/max(norm(a),eps());push!(array_errors,err)
                    @test err<1e-11
                end
            end
        end
        old=merge(ctx,(Cset=cs,Tset=ts));new=merge(ctx,(Cset=ac,Tset=at))
        # bond_expectation reads its 2x2 Q/R patch explicitly (no matrix wrap).
        for x in 1:2,y in 1:2,o in (:up,:down),edge in 1:3
            @test abs(V.bond_expectation(old,x,y,o,edge)-V.bond_expectation(new,x,y,o,edge))<1e-11
        end
        println("Compared ",orientation," center; max relative tensor difference=",maximum(array_errors));flush(stdout)
    end
end
@assert bytes2hex(sha256(read(file)))==before
println("ARRAY STORAGE PASSED; maximum relative tensor error=",maximum(array_errors))
