using HDF5, JLD2
using LinearAlgebra
cd(@__DIR__)
chi=40;

a=load("test.jld2")
AA=a["init"]["AA"];
M1=a["init"]["M1"];
M2=a["init"]["M2"];
M3=a["init"]["M3"];
M5=a["init"]["M5"];
M7=a["init"]["M7"];
M8=a["init"]["M8"];
M1_=a["init"]["M1_"];
M2_=a["init"]["M2_"];
M3_=a["init"]["M3_"];
M5_=a["init"]["M5_"];
M7_=a["init"]["M7_"];
M8_=a["init"]["M8_"];
PM=a["init"]["PM"];
PM_inv=a["init"]["PM_inv"];

PM0=a["init"]["PM0"];
PM_inv0=a["init"]["PM_inv0"];

