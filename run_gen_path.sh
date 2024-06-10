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

source ~/.bashrc_dcs
conda activate habitat
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NOD


echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date

srun python run_data_collector_pathfinder.py -cf cfg/data_collection_pathfinder_aimos.yaml --path_id_offset 0 --seed 64

echo "Run completed at:- "
date