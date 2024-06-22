#!/bin/bash
#SBATCH --job-name=yh
#SBATCH -o output/run_%j.out
#SBATCH -e output/run_%j.err
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # total number of tasks across all nodes
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

python run_data_collector.py -cf cfg/data_collection_pathfinder_aimos.yaml --path_id_offset 0 --seed 64
