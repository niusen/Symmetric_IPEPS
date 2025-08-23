using MAT


#lines = readlines("your_file.out")
filenm="2025_08_04_23_11_57.out";
file_content = readlines(filenm);



#store line index
line_index=Vector{Int64}(undef,0);
for cc=1:length(file_content)
    if occursin("optim iteration", file_content[cc]);
        if cc+1<=length(file_content)
            push!(line_index,cc+1)
        end
    end
end



E_set=Vector{Float64}(undef,0);
ex_set=Vector{Vector{ComplexF64}}(undef,0);
ey_set=Vector{Vector{ComplexF64}}(undef,0);
e_diagonal1_set=Vector{Vector{ComplexF64}}(undef,0);
e0_set=Vector{Vector{ComplexF64}}(undef,0);
eU_set=Vector{Vector{ComplexF64}}(undef,0);


pattern = r"(\w+)\s*=\s*([^\[\],]+(?:\[[^\]]*\])?)"

step=1;
for cc in line_index
    line=file_content[cc];
    # println(line)
    vars = Dict{String, Any}()

    for m in eachmatch(pattern, line)
        key = m.captures[1]
        val_str = strip(m.captures[2])
        val = Meta.parse(val_str) |> eval
        vars[key] = val
    end
    push!(E_set,vars["E"]);
    push!(ex_set,vars["ex_set"]);
    push!(ey_set,vars["ey_set"]);
    push!(e_diagonal1_set,vars["e_diagonal1_set"]);
    push!(e0_set,vars["e0_set"]);
    push!(eU_set,vars["eU_set"]);


end



matwrite(filenm*".mat", Dict(
    "E_set" => E_set,
    "ex_set" => ex_set,
    "ey_set" => ey_set,
    "e_diagonal1_set" => e_diagonal1_set,
    "e0_set" => e0_set,
    "eU_set" => eU_set,
))