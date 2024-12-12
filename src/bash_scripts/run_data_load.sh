#!/bin/bash

#SBATCH --job-name=EXAMPLE

module purge
module load miniconda

conda activate pes

# Retrieve .env variables
if [ -f DeepForestcast/src/.env ]; then
  export $(grep -v '^#' DeepForestcast/src/.env | xargs)
fi

python DeepForestcast/src/main_data_loading.py