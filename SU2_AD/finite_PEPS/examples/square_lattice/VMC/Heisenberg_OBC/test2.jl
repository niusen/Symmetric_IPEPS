using Distributed
#number of workers to add and soft restrict of memory
#addprocs(50; exeflags=["--heap-size-hint=6G"])

@everywhere using LinearAlgebra:I,diagm,diag
@everywhere using TensorKit
@everywhere using Random
@everywhere using Printf
@everywhere using DelimitedFiles
@everywhere using CSV
@everywhere using DataFrames
@everywhere using JLD2,MAT

@everywhere cd(@__DIR__)



a=randn(3,3);
jldsave("test.jld2";a);

outputname="test.jld2";

jldopen(outputname, "a+") do file
    @show file["a"]
    file["b"]=zeros(2,2);
    @show haskey(file,"b")
end
