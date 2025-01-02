import os
import json, yaml
from models.utils import get_model_type
import torch
import numpy as np
import random
import importlib.util
import argparse
import random
import wandb
import argparse
import pprint
import warnings
from configs import hyperparameter_debug_3D

def fix_randomness(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multiple GPUs

def compute_input_dimensions(data_layers):
    # Compute the input dimensions
    input_dim_x = 2 + ( sum(data_layers["static"].values()) if "static" in data_layers else 0 )
    input_dim_y = 8 + ( sum(data_layers["time"].values()) if "time" in data_layers else 0 )

    # add extra channels for multibands
    if "static" in data_layers:
        if "agroclimate_zones_reproj_one_hot" in data_layers["static"] and data_layers["static"]["agroclimate_zones_reproj_one_hot"]:
            input_dim_x += 11 # 12 total
        if "block_boundaries_reproj_one_hot" in data_layers["static"] and data_layers["static"]["block_boundaries_reproj_one_hot"]:
            input_dim_x += 38 # 39 total 
        if "updated_block_boundaries_reproj_one_hot" in data_layers["static"] and data_layers["static"]["updated_block_boundaries_reproj_one_hot"]:
            input_dim_x += 45 # 46 total 
        if "updated_roads_reproj_one_hot" in data_layers["static"] and data_layers["static"]["updated_roads_reproj_one_hot"]:
            input_dim_x += 11 # 12 total 

    input_dim_3D = (input_dim_x, input_dim_y)
    input_dim_2D = input_dim_x + input_dim_y

    print("Data dimensions:", input_dim_2D, "(2D) and ", input_dim_3D, "(3D)")
    return input_dim_2D, input_dim_3D


def load_config_from_object(file_path, object_name):
    # Dynamically load the module from the file
    spec = importlib.util.spec_from_file_location("config", file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Get the desired object
    config = getattr(config_module, object_name, None)
    if config is None:
        raise ValueError(f"Object '{object_name}' not found in {file_path}")
    return config

def get_paths(dir, load_region, modeltype, save_region=None):
    """ just pass in region unless you want to load data from a region different than that of the model (ie train on nagaland, then meghalaya)
    """

    if save_region is None:
        save_region = load_region

    # WHERE TO IMPORT DATA FROM
    sourcepath = dir + "/outputs/" + load_region
    wherepath = dir + "/outputs/" + save_region + "/tensors"
    savepath = dir + "/outputs/" + load_region + "/out"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # WHERE TO SAVE MODEL CHECKPOINT
    modelpath = dir + "/models/" + load_region + "_models/" + modeltype
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    # WHERE TO SAVE IMAGES TRACKING TRAINING PROCESS
    picspath = dir + "/models/" + load_region + "_models/" + modeltype + "/pics"
    if not os.path.exists(picspath):
        os.makedirs(picspath)

    # WHERE TO SAVE MODEL PERFORMANCE OF EACH JOB FOR TRAIN, VAL AND TEST DATA
    file = dir + "/models/" + load_region + "_models/" + modeltype + "/grid_summary/ConvRNN.Conv_" + modeltype + ".mem.txt"
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    
    return sourcepath, wherepath, savepath, modelpath, picspath, file

def get_job_details():
    if "$SLURM_ARRAY_TASK_ID" in os.environ:
        job = int(os.environ["$SLURM_ARRAY_TASK_ID"])
    else:
        job = 1
    if "SLURM_JOBID" in os.environ:
        job_id = str(os.environ["SLURM_JOBID"])
    else:
        job_id = "1"

    return job, job_id

def load_config(file_path, object_name=None):
    if object_name and not file_path:
        raise ValueError("--config missing, required if specifying --object_name")

    if isinstance(file_path, dict):
        return file_path # already loaded
    
    if file_path.endswith('.py'):
        return load_config_from_object(file_path, object_name)

    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith(('.yaml', '.yml')):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file type")
    
def get_arguments(DEFAULT_CONFIG, WANDB_PROJECT_ID, TEST_EPOCHS):

    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--debug', action='store_true', help='Override model settings for fast, debugging model', required=False, default=False)
    parser.add_argument('--skip_wandb', action='store_true', help='Do not run Weights & Biases', required=False, default=False)
    parser.add_argument('--test_only', action='store_true', help='Skip training, only test', required=False, default=False)
    parser.add_argument('--wandb_project', nargs=1,  help='Weights and Biases project', required=False)
    parser.add_argument('--init_model', type=str, nargs=1, help='Use preexisting model', default=None)
    parser.add_argument( '--test_epochs', type=int, nargs=1, help='To test year-ahead after every x epochs.', required=False, default=0)
    parser.add_argument('--config', type=str, nargs=1, help='Path to the configuration file', required=False)
    parser.add_argument('--config_object', type=str, nargs=1, help='Object within the Python filepath specified in --config', required=False)
    # parser.add_argument('--sweep', type=str, nargs=1, help='W&B sweep id (alternative to --config)', required=False) # TODO this should be separate; function which calls this and passes along relevant args
    args = parser.parse_args()

    # Iteratively unpack each argument
    for arg_name in ["wandb_project", "init_model", "test_epochs", "config", "config_object"]:
        value = getattr(args, arg_name, None)
        if isinstance(value, list) and len(value) == 1:
            setattr(args, arg_name, value[0])

    if args.wandb_project and args.skip_wandb:
        raise ValueError("Please specify only one of --skip_wandb and --wandb_project")

    
    if args.test_only and not args.init_model:
        raise ValueError("You have specified --test_only; please include a model to test with --init_model.")

    # Handle init_model to start training from
    if args.init_model != "" and args.init_model is not None:
        init_bestmodelpath = args.init_model
        if not init_bestmodelpath.endswith(".pt"):
            init_bestmodelpath += ".pt"
    else:
        init_bestmodelpath = None

    # Handle debugging model (parameter overrides provided later on)
    if args.debug and args.wandb_project:
        warnings.warn("W&B Project is specified, but debug flag is specified so it will be ignored.")

    if not args.wandb_project:
        args.wandb_project = WANDB_PROJECT_ID

    if args.debug or args.skip_wandb:
        if args.debug: print("Debugging mode is enabled, overriding model parameters in favor of fast, simple model. W&B will be also disabled.")
        if args.config:
            print("Attempting config load from:", args.config, args.config_object)
            config = load_config(args.config, args.config_object)
        else:
            print("Loading debug config: hyperparameter_debug_3D")
            config = load_config(hyperparameter_debug_3D)
    else:
        # TODO not working as intended (but sweep overrides configurations below, so not a problem)
        # # Currently running W&B sweep
        # if wandb.run is not None and wandb.run.sweep_id is not None:
        #     print(f"Running in a W&B sweep with ID: {wandb.run.sweep_id}")
        
        # Set default value
        if not args.config:
            print("Loading default config: hyperparameter_debug_3D")
            args.config = DEFAULT_CONFIG
        else:
            print("Attempting config load from:", args.config, " : ", args.config_object)

        wandb.init(config=load_config(args.config, args.config_object), 
                settings=wandb.Settings(start_method="fork"), # https://docs.wandb.ai/support/initstarterror_error_communicating_wandb_process
                project=args.wandb_project)

        config = wandb.config
    pprint.pprint(config)

    # Override TEST_EPOCHS based on arguments
    if args.test_epochs <= 0:
        TEST_EPOCHS = 0
    if args.test_epochs > config["n_epochs"]:
        raise Exception("Test epochs greater than configured number of total epochs.")
    else:
        TEST_EPOCHS = args.test_epochs

    return args, init_bestmodelpath, config, TEST_EPOCHS

def get_forecast_arguments():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config', type=str, nargs=1, help='Path to the configuration file', required=True)
    parser.add_argument('--config_object', type=str, nargs=1, help='Object within the Python filepath specified in --config', required=True)
    parser.add_argument('--modelpath', type=str, nargs=1, help='Use preexisting model', required=True)
    args = parser.parse_args()

    # Map [arg] -> arg
    for arg_name in ["config", "config_object", "modelpath"]:
        value = getattr(args, arg_name, None)
        if isinstance(value, list) and len(value) == 1:
            setattr(args, arg_name, value[0])

    config = load_config(args.config, args.config_object)
    if config["end_year"] + 1 is not config["forecast_year"]:
        raise Exception("Forecast year should immediately follow data.")
    
    return args, config, args.modelpath