using Zygote, ChainRulesCore

function checkpointed(f, args...)
    f(args...)
end



function Zygote._pullback(ctx::Zygote.AContext, ::typeof(checkpointed), f, xs...)
    y = f(xs...)
    function pullback_checkpointed(Δy)
        y, pb = Zygote._pullback(ctx, f, xs...)
        return pb(Δy)
    end
    return y, pullback_checkpointed
end


function myfun(x)
    y=x*2;
    println("aaaaa")
    return y
end



#y, pb = Zygote.pullback(checkpointed, myfun, pi/2)
function fun(ψ)
    E=Zygote.checkpointed(myfun,ψ)
    return E
end

function cfun(x)
    ψ = x




    ∂E = fun'(ψ)


    #@assert !isnan(norm(∂E))
    return ∂E
end


cfun(1)