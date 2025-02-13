# DeepFore[st]cast'

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Leveraging deep convolutional neural networks to forecast tropical deforestation.

## Citation

Please cite:

Thorstenson, R. (2024). Forecasting Deforestation in India with Deep Learning for the GREEN Meghalaya PES Program. Yale University.

Ball, J. G. C., Petrova, K., Coomes, D. A., & Flaxman, S. (2022). Using deep convolutional neural networks to forecast spatial patterns of Amazonian deforestation. *Methods in Ecology and Evolution*, 13, 2622– 2634. [https://doi.org/10.1111/2041-210X.13953](https://doi.org/10.1111/2041-210X.13953)


## Requirements
- Python 3.8+
- scikit-learn
- torch 1.9.0
- torchaudio 0.9.0
- torchvision 0.10.0

See `src/requirements/environment.yml` for the complete list.

## Getting started

First, create the directories and download the appropriate data. See [https://github.com/PatBall1/DeepForestcast/tree/master](https://github.com/PatBall1/DeepForestcast/tree/master) for details. Also see the `Makefile` in the root, `src/bash_scripts/run_data_load.sh`, and `src/main_data_load.py`. 

Then, update the `.env` file. See `src/.env.example`. Depending on your use-case, not all are required. 

Next, build a config in `src/configs.py`, then call `python3 src/models.py`. It can take the parameters below. If preferred, use the bash scripts `src/bash_scripts/`.

`--debug`:
Enables debugging mode by overriding model settings for a simpler and faster configuration. Disables Weights & Biases integration if `--wandb_project` is specified.

`--skip_wandb`:
Disables Weights & Biases logging for the run.

`--test_only`:
Skips training and runs only testing. Requires a pretrained model specified via `--init_model`.

`--wandb_project`:
Specifies a custom W&B project for logging.

`--init_model` (str):
Path to a pretrained model to initialize training or testing. The file should have a .pt extension.

`--test_epochs` (int):
Specifies how often (in terms of epochs) to test the model during training.

`--config` (str):
Path to the configuration file for the model.

`--config_object` (str):
Specifies a Python object within the file provided by `--config` to load the model configuration.