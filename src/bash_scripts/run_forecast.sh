#!/bin/bash

#SBATCH --job-name=FORECAST

module purge
module load miniconda

conda activate pes

# Retrieve .env variables
if [ -f DeepForestcast/src/.env ]; then
  export $(grep -v '^#' DeepForestcast/src/.env | xargs)
fi

python DeepForestcast/src/forecasting.py
