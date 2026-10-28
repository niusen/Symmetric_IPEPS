# No TensorKit dependency: compare diagnostics generated in two environments.
using TOML, Test
length(ARGS) == 2 || error("usage: compare_compatibility.jl old.toml new.toml")
old,new = TOML.parsefile.(ARGS)
cases = sort([key for (key,value) in old if value isa AbstractDict])
max_energy_difference = 0.0
max_spectrum_difference = 0.0
@testset "Old/new TensorKit numerical agreement" begin
    @test cases == sort([key for (key,value) in new if value isa AbstractDict])
    for key in cases
        a,b = old[key],new[key]
        @test a["dimensions"] == b["dimensions"]
        for field in ("up_energy","down_energy","lambda_spectra")
            @test length(a[field]) == length(b[field])
            difference = maximum(abs.(a[field]-b[field]))
            @test difference < 1e-11
            if field == "lambda_spectra"
                global max_spectrum_difference = max(max_spectrum_difference,difference)
            else
                global max_energy_difference = max(max_energy_difference,difference)
            end
        end
    end
end
println("TensorKit versions: ",old["TensorKit"]," / ",new["TensorKit"])
println("maximum SU energy-estimate difference: ",max_energy_difference)
println("maximum lambda-spectrum difference: ",max_spectrum_difference)
