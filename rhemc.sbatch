#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=fc_popgen
#
# Partition:
#SBATCH --partition=savio2
#
# Tasks per node
#SBATCH --ntasks-per-node=5
#
# Nodes
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=0:59:30
#
## Command(s) to run:
module load gcc openmpi # or module load intel openmpi
ht_helper.sh -m "python/3.6" -t rhemc.cmds 
