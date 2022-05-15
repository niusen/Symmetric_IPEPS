#!/bin/bash
##### Example for 1 node and 18 tasks 1 GPU #####
#SBATCH -J test_tn-torch_cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH -B 1:18:1

#SBATCH --mem=140000
#SBATCH --threads-per-core=1
#SBATCH --time=96:00:00
#SBATCH --out=myJob-cpu-%j.out
#SBATCH --err=myJob-cpu-%j.err
# --qos=<qos_name>
# -B S[:C[:T]]  Combined shortcut option for --sockets-per-node, --cores-per-cpu, --threads-per-core
# --cpus-per-task=18

# the necessary modules to run the peps-torch
module purge

# set how many threads/cores are allowed for Lapack/BLAS to be used
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cores=$OMP_NUM_THREADS




/tmpdir/niu/julia-1.7.2/bin/julia test_threads.jl --threads 3
