# Independent occupation-bit implementation, not the production JW/kron code.
function exact_triangle(parameters,orientation; onsite_fraction=0.5)
    occupancies = (0,3,1,2)
    bits = [occupancies[a]+4occupancies[b]+16occupancies[c] for c in 1:4 for b in 1:4 for a in 1:4]
    lookup = Dict(bits[i]=>i for i in eachindex(bits))
    H = zeros(ComplexF64,64,64)
    t = parameters[orientation == :up ? "t_up" : "t_dn"]
    for (col,b) in enumerate(bits)
        n = count_ones(b)
        d = sum(((b >> (2i)) & 3) == 3 for i in 0:2)
        interaction = get(parameters,"centered",true) ? d-n/2+3/4 : d
        H[col,col] += onsite_fraction*(parameters["U"]*interaction-parameters["μ"]*n)
        for (k,(i,j)) in enumerate(((1,2),(2,3),(3,1))), spin in 0:1
            for (dst,src,amplitude) in ((2(i-1)+spin,2(j-1)+spin,-t[k]),
                                        (2(j-1)+spin,2(i-1)+spin,-conj(t[k])))
                iszero(b & (1 << src)) && continue
                b1 = b ⊻ (1 << src)
                !iszero(b1 & (1 << dst)) && continue
                sign = (-1)^(count_ones(b & ((1 << src)-1))+count_ones(b1 & ((1 << dst)-1)))
                row = lookup[b1 ⊻ (1 << dst)]
                H[row,col] += amplitude*sign
            end
        end
    end
    return H
end

function atomic_state(occupation)
    s = initial_iPESS(1,1,Rep[ℤ₂](0=>1);rng=MersenneTwister(2))
    for b in 1:3
        data = zeros(ComplexF64,1,1,4)
        data[1,1,occupation] = 1
        s.B[1,1,b] = TensorMap(data,codomain(s.B[1,1,b]),domain(s.B[1,1,b]))
    end
    return s
end
