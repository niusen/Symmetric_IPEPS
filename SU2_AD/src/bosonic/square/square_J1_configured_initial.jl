"""Configured parity-resolved 2×2 initial states for square-J1 optimization."""

function square_J1_configured_matching_cell(
    matching::Symbol,
    even_multiplets,
    odd_multiplets;
    seed::Integer=666,
)
    isempty(even_multiplets) && throw(ArgumentError("even_multiplets cannot be empty"))
    isempty(odd_multiplets) && throw(ArgumentError("odd_multiplets cannot be empty"))
    all(pair -> last(pair) isa Integer && last(pair) > 0, even_multiplets) ||
        throw(ArgumentError("even multiplet multiplicities must be positive integers"))
    all(pair -> last(pair) isa Integer && last(pair) > 0, odd_multiplets) ||
        throw(ArgumentError("odd multiplet multiplicities must be positive integers"))

    Veven = SU2Space(even_multiplets...)
    Vodd = SU2Space(odd_multiplets...)
    all(sector -> isodd(dim(sector)), sectors(Veven)) ||
        throw(ArgumentError("Veven must contain integer-spin SU(2) sectors only"))
    all(sector -> iseven(dim(sector)), sectors(Vodd)) ||
        throw(ArgumentError("Vodd must contain half-integer-spin SU(2) sectors only"))

    Random.seed!(seed)
    return square_J1_matching_cell(matching, Veven, Vodd)
end

"""Load an existing Simple Update, variational, or Full Update iPEPS cell."""
function square_J1_load_configured_cell(filename::AbstractString, cell_Lx::Int, cell_Ly::Int)
    data = load(filename)
    value = if haskey(data, "A_set")
        data["A_set"]
    elseif haskey(data, "T_set")
        data["T_set"]
    elseif haskey(data, "A_cell")
        data["A_cell"]
    elseif haskey(data, "x")
        data["x"]
    elseif haskey(data, "A")
        stored_A = data["A"]
        stored_A isa AbstractMatrix ? stored_A : fill(stored_A, cell_Lx, cell_Ly)
    else
        throw(ArgumentError("initial state must contain A_set, T_set, A_cell, x, or A"))
    end
    raw = value isa Tuple ? square_fu_cell_to_matrix(value) : value
    size(raw) == (cell_Lx, cell_Ly) || throw(DimensionMismatch(
        "initial state cell has size $(size(raw)); expected ($cell_Lx, $cell_Ly)",
    ))
    A_set = [begin
        entry = raw[cx, cy]
        A = entry isa TensorMap ? entry :
            (hasproperty(entry, :T) ? getproperty(entry, :T) :
             throw(ArgumentError("cannot extract tensor at ($cx, $cy)")))
        A / norm(A)
    end for cx in 1:cell_Lx, cy in 1:cell_Ly]
    _square_fu_validate_cell(A_set)
    return A_set
end
