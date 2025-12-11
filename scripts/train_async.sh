#!/bin/bash
#SBATCH --job-name=cso
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=750G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

# Loop over 10
for i in $(seq 1 10)
do
  echo "Run number: $i"
  # Kill any process using port 8080 after 10 hours
  ( sleep 36000 && fuser -k 8080/tcp ) & \
  bash scripts/run_async_training.sh "$@"
done
