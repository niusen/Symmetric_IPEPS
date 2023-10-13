using Zygote
using Zygote: @adjoint
Zygote.refresh()
# checkpoint(f, x) = f(x)
# @adjoint checkpoint(f, x) = f(x), ȳ -> Zygote._pullback(f, x)[2](ȳ)
#

# foo(x) = (println(x); sin(x))
# gradient(x -> checkpoint(foo, x), 1)

# foo(x) = (println(x); sin(x))
# gradient(x -> checkpoint(foo, x), 1)

function change(x)
    y=deepcopy(x);
    y=y*2
    println("aaa")
    return y;
end
Zygote.refresh()
checkpoint(change, x) = change(x);
@adjoint checkpoint(change, x) = change(x), ȳ -> Zygote._pullback(change, x)[2](ȳ)
#gradient(x -> checkpoint(myfun, x), 1)

function myfun(x)

    for cc=1:3
        #x=change(x);
        x=Zygote.checkpointed(change, x)



    end
    return x
end

function cfun(x)
    ψ = x

    function fun(ψ)
        E=myfun(ψ)
        return E
    end

    ∂E = fun'(ψ)


    #@assert !isnan(norm(∂E))
    return ∂E
end


cfun(1)
#gradient(x -> checkpoint(myfun, x), 1)
#gradient(x -> myfun(x), 1)
