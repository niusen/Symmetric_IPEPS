function normalize_no_grad(T)
    Norm=norm(T);
    return T/Norm;
end
function ChainRulesCore.rrule(::typeof(normalize_no_grad), T::AbstractTensorMap)
    function normalize_no_grad_pushback(f̄wd)
        return NoTangent(), f̄wd
    end 
    return normalize_no_grad(T), normalize_no_grad_pushback
end


function show_grad(T)
    
    return T;
end
function ChainRulesCore.rrule(::typeof(show_grad), T::AbstractTensorMap)
    function show_grad_pushback(f̄wd)
        println("Grad of variable: "*string(norm(f̄wd)))
        return NoTangent(), f̄wd
    end 
    return show_grad(T), show_grad_pushback
end


function my_pinv(T::DiagonalTensorMap)
    epsilon0 = 1e-12
    epsilon=epsilon0*maximum(abs.(diag(convert(Array,T))))^2
    T_new=deepcopy(T);
    
    mm=T_new.data;
    mm = mm./(mm.^2 .+epsilon)
    T_new.data.=mm;

    return T_new
end

function my_pinv2(T::DiagonalTensorMap)
    epsilon0 = 1e-8
    epsilon=epsilon0*maximum(abs.(diag(convert(Array,T))))
    T_new=deepcopy(T);

    mm=T_new.data;
    mm = (1 ./mm).*(abs(mm).>epsilon)
    T_new.data.=mm;

    # if sectortype(space(T_new,1)) == Trivial
    #     mm=T_new.data;
    #     @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
    #     for i=1:size(mm,1)
    #         if abs(mm[i,i])>epsilon
    #             mm[i,i] = 1/mm[i,i]
    #         end
    #     end
    #     T_new=TensorMap(mm,codomain(T),domain(T));
    # else
    #     for cc=1:length(T_new.data.values)
    #         mm=T_new.data.values[cc];
    #         @assert (norm(diag(mm))-norm(mm))/norm(mm)<1e-14;
    #         for i = 1:size(mm,1)
    #             if abs(mm[i,i])>epsilon
    #                 mm[i,i] = 1/mm[i,i]
    #             end
    #         end
    #         T_new.data.values[cc]=mm;
    #     end
    # end
    return T_new
end


