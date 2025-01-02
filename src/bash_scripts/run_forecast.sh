#!/bin/bash

#SBATCH --job-name=forecast
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mem=80G
#SBATCH --nodes=2
#SBATCH --gpus-per-node=3

module purge
module load miniconda

conda activate pes

# Retrieve .env variables
if [ -f DeepForestcast/src/.env ]; then
  export $(grep -v '^#' DeepForestcast/src/.env | xargs)
fi

python DeepForestcast/src/forecasting.py --config DeepForestcast/src/configs.py --config_object forecast_config --modelpath /gpfs/gibbs/project/pande/rtt8/storage/DeepForestcast/models/meghalaya_only_models/3D/torch.nn.parallel.data_parallel.DataParallel/torch.nn.parallel.data_parallel.DataParallel_7.12.24_14.39_10232281.pt
