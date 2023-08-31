cd(@__DIR__)
init=load("fail_CTM_SingleLayer_D28_chi160.jld2")
CTM=init["init"]["CTM"]
Cset=CTM["Cset"]


corner=1;

part=1;

println(space(Cset[1][1,1]))
if part==1
    println(spectrum_conv_check([1],Cset[corner][1,1])[2][1:5])
    println(spectrum_conv_check([1],Cset[corner][1,2])[2][1:5])
    println(spectrum_conv_check([1],Cset[corner][2,1])[2][1:5])
    println(spectrum_conv_check([1],Cset[corner][2,2])[2][1:5])
elseif part==2
    println(spectrum_conv_check([1],Cset[corner][1,1])[2][end-5:end])
    println(spectrum_conv_check([1],Cset[corner][1,2])[2][end-5:end])
    println(spectrum_conv_check([1],Cset[corner][2,1])[2][end-5:end])
    println(spectrum_conv_check([1],Cset[corner][2,2])[2][end-5:end])
end


println("11111")

init=load("CTM_SingleLayer_D28_chi240.jld2")
CTM=init["init"]["CTM"]
Cset=CTM["Cset"]

println(space(Cset[1][1,1]))
if part==1
    println(spectrum_conv_check([1],Cset[corner][1,1])[2][1:5])
    println(spectrum_conv_check([1],Cset[corner][1,2])[2][1:5])
    println(spectrum_conv_check([1],Cset[corner][2,1])[2][1:5])
    println(spectrum_conv_check([1],Cset[corner][2,2])[2][1:5])
elseif part==2
    println(spectrum_conv_check([1],Cset[corner][1,1])[2][end-5:end])
    println(spectrum_conv_check([1],Cset[corner][1,2])[2][end-5:end])
    println(spectrum_conv_check([1],Cset[corner][2,1])[2][end-5:end])
    println(spectrum_conv_check([1],Cset[corner][2,2])[2][end-5:end])
end