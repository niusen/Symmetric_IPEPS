# Read the list of numbers from the CSV file
outputname="test.csv"
data = open(outputname, "r") do file
    #[parse(ComplexF64, line) for line in readlines(file)]
    dset1=Vector{ComplexF64}(undef,0);
    dset2=Vector{ComplexF64}(undef,0);
    dset3=Vector{ComplexF64}(undef,0);
    dset4=Vector{ComplexF64}(undef,0);
    for line in readlines(file)
        tokens = strip.(split(line, ","))
        complex_numbers = parse.(Complex{Float64}, tokens)
        push!(dset1,complex_numbers[1]);
        push!(dset2,complex_numbers[2]);
        push!(dset3,complex_numbers[3]);
        push!(dset4,complex_numbers[4]);
    end
    [dset1,dset2,dset3,dset4]
end