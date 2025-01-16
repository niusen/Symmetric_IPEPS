

function data_analysis()


    # Read the list of numbers from the CSV file
    data = open("test.csv", "r") do file
        [parse(Float64, line) for line in readlines(file)]
    end

    total_data_size = length(data)
    println(total_data_size)

    # Output file
    outputname = "errors_varying_bin_sizes.csv"
    open(outputname, "w") do outfile
        bin_size = 1

        while bin_size < total_data_size
            # Bin the data
            binned_real = [mean(real(data[i:min(i+bin_size-1, total_data_size)])) for i in 1:bin_size:total_data_size]
            
            # Compute mean energy per site
            energypersite = mean(binned_real) / L
            
            # Compute standard deviation
            std_dev = std(binned_real; corrected=false) / (L*sqrt(length(binned_real)))

            # Write results to the output file
            @printf(outfile, "%10d %30.7f %30.7f\n", bin_size, energypersite, std_dev)

            # Double the bin size
            bin_size *= 2
        end
    end

end