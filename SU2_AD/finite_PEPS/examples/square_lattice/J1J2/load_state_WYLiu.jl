using HDF5, JLD2, MAT
cd(@__DIR__)

file = matopen("WYLiu_D2.mat")
A=read(file, "A")
close(file)