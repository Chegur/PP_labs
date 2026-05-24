#!/bin/bash
#SBATCH --job-name=lab5_mpi
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=batch
#SBATCH --output=slurm_%j.out

module load intel/mpi5

mkdir -p results

echo "=== Start Benchmarks ==="
for np in 1 2 4 8; do
    echo "--- $np processes ---"
    srun -n $np ./lab5_mpi > results/out_${np}.txt 2>&1
    echo "Done."
done
echo "=== All Finished ==="