#!/bin/bash

#SBATCH --job-name=EXAMPLE

module purge
module load miniconda

conda activate pes

# Retrieve .env variables
if [ -f DeepForestcast/src/.env ]; then
  export $(grep -v '^#' DeepForestcast/src/.env | xargs)
fi

python DeepForestcast/src/model.py --config DeepForestcast/src/configs.py --config_object hyperparameter_defaults_3D
