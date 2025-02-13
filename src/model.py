from data_loading.get_setup import get_paths, get_job_details, fix_randomness, get_arguments, compute_input_dimensions
from models.train_3DCNN import train_3DCNN
from models.test_3DCNN import test_3DCNN
from models.train_2DCNN import train_2DCNN
from models.test_2DCNN import test_2DCNN
from configs import hyperparameter_defaults_3D, hyperparameter_defaults_2D, wandb_data_layers
import time 
import math
import pprint
import os

fix_randomness()

USER = os.environ.get('USER')
DIRECTORY = os.environ.get('DeepForestcast_PATH')
DEFAULT_CONFIG = hyperparameter_defaults_2D
WANDB_PROJECT_ID = "sweep" # overridden by args parser
TEST_EPOCHS = 0 # to test year-ahead after every [1, n_epochs] or <= 0 to test after training n_epochs # overridden by args parser

start = time.time()

job, job_id = get_job_details()

args, init_bestmodelpath, config, TEST_EPOCHS = get_arguments(DEFAULT_CONFIG, WANDB_PROJECT_ID, TEST_EPOCHS)

REGION = config["region"]
sourcepath, wherepath, savepath, modelpath, picspath, file = get_paths(DIRECTORY, REGION, config["modeltype"])

# Compute year offset to avoid train/test data overlap
testing_years_ahead_offset = config["years_ahead"] - 1
if "train_years" in config:
    testing_years_ahead_offset += (config["train_years"] - 1)

# Percentile to for threshold selection
# Advisable to be 100*times/(times+1)
perc = (100 * config["train_times"]) / (
    config["train_times"] + 1
)  

# Debugging override
if args.debug:
    hidden_dim = [1, 1, 1]
    batch_size = 2
    size = 7
else:
    hidden_dim = [config["hidden_dim1"], 
                    config["hidden_dim2"], 
                    config["hidden_dim3"]]
    batch_size = config["batch_size"]

data_layers = config["data_layers"] if "data_layers" in config else wandb_data_layers
input_dim_2D, input_dim_3D = compute_input_dimensions(data_layers)

print("is data layers in config:", "data_layers" in config)

pprint.pprint(config)
pprint.pprint(data_layers)

def wrapped_train_3DCNN(config, n_epochs, bestmodelpath=None):
    return train_3DCNN(
        # Static
        wherepath=wherepath, 
        modelpath=modelpath, 
        picspath=picspath, 
        file=file,
        model_type_name=config["modeltype"],

        # Computed from config
        input_dim=input_dim_3D,
        perc=perc,

        # Config-dependent or overrode for debugging
        hidden_dim=hidden_dim,
        batch_size=batch_size,

        data_layers=data_layers,

        start_year=config["start_year"]-1,
        end_year=config["end_year"]-1,
        kernel_size=config["kernel_size"],
        # stride=config["stride"],
        # padding=config["padding"],
        levels=config["levels"],
        train_times=config["train_times"],
        test_times=config["test_times"],
        AUC=config["AUC"],
        BCE_Wloss=config["BCE_Wloss"],
        FNcond=config["FNcond"],
        w=config["w"],
        pos_weight=config["pos_weight"],
        lr=config["lr"],
        dropout=config["dropout"],
        size=config["size"],
        weight_decay=config["weight_decay"],
        n_splits=config["n_splits"],
        n_epochs=n_epochs,
        patience=config["patience"],
        training_time=config["training_time"],
        stop_batch=config["stop_batch"],
        print_batch=config["print_batch"],

        years_ahead=config["years_ahead"],
        job_id=job_id,
        job=job,
        sourcepath=sourcepath, # TODO extract, root in moving AUC-threshold calculation outside
        pretrained_path=bestmodelpath,
        train_years=config["train_years"] if "train_years" in config else 1,
    )

def wrapped_test_3DCNN(config, bestmodelpath):
    return test_3DCNN(
        # Static
        wherepath=wherepath, 
        modelpath=modelpath, 
        picspath=picspath,
        model_type_name=config["modeltype"],

        region=REGION,
        start_year=config["start_year"] + testing_years_ahead_offset,
        end_year=config["end_year"] + testing_years_ahead_offset,

        # Computed from config
        input_dim=input_dim_3D,
        perc=perc,

        # Config-dependent or overrode for debugging
        hidden_dim=hidden_dim,
        batch_size=batch_size,

        data_layers=data_layers,

        kernel_size=config["kernel_size"],
        # stride=config["stride"],
        # padding=config["padding"],
        levels=config["levels"],
        test_times=config["test_times"],
        w=config["w"],
        pos_weight=config["pos_weight"],
        dropout=config["dropout"],
        size=config["size"],
        stop_batch=config["stop_batch"],
        print_batch=config["print_batch"],
        years_ahead=config["years_ahead"],
        checkpoint=bestmodelpath,
        sourcepath=sourcepath + "/",
    )

def wrapped_train_2DCNN(config, n_epochs, bestmodelpath=None):
    return train_2DCNN(
        # Static
        wherepath=wherepath, 
        modelpath=modelpath, 
        picspath=picspath, 
        file=file,
        model_type_name=config["modeltype"],

        # Computed from config
        input_dim=input_dim_2D,
        perc=perc,

        # Config-dependent or overrode for debugging
        batch_size=batch_size,

        data_layers=data_layers,

        hidden_dim1=config["hidden_dim1"],
        hidden_dim2=config["hidden_dim2"],
        hidden_dim3=config["hidden_dim3"],
        hidden_dim4=config["hidden_dim4"],
        start_year=config["start_year"]-1,
        end_year=config["end_year"]-1,
        kernel_size=config["kernel_size"],
        stride=config["stride"],
        padding=config["padding"],
        levels=config["levels"],
        train_times=config["train_times"],
        test_times=config["test_times"],
        AUC=config["AUC"],
        BCE_Wloss=config["BCE_Wloss"],
        FNcond=config["FNcond"],
        w=config["w"],
        pos_weight=config["pos_weight"],
        lr=config["lr"],
        dropout=config["dropout"],
        size=config["size"],
        weight_decay=config["weight_decay"],
        n_splits=config["n_splits"],
        n_epochs=n_epochs,
        patience=config["patience"],
        stop_batch=config["stop_batch"],
        training_time=config["training_time"],
        print_batch=config["print_batch"],
        job_id=job_id,
        years_ahead=config["years_ahead"],
        pretrained_path=bestmodelpath,
    )

def wrapped_test_2DCNN(config, bestmodelpath):
    return test_2DCNN(
        # Static
        wherepath=wherepath, 
        modelpath=modelpath, 
        model_type_name=config["modeltype"],

        region=REGION,
        start_year=config["start_year"] + testing_years_ahead_offset,
        end_year=config["end_year"] + testing_years_ahead_offset,

        # Computed from config
        input_dim=input_dim_2D,
        perc=perc,

        # Config-dependent or overrode for debugging
        batch_size=batch_size,

        data_layers=data_layers,

        hidden_dim1=config["hidden_dim1"],
        hidden_dim2=config["hidden_dim2"],
        hidden_dim3=config["hidden_dim3"],
        hidden_dim4=config["hidden_dim4"],
        kernel_size=config["kernel_size"],
        stride=config["stride"],
        padding=config["padding"],
        levels=config["levels"],
        test_times=config["test_times"],
        w=config["w"],
        pos_weight=config["pos_weight"],
        dropout=config["dropout"],
        size=config["size"],
        stop_batch=config["stop_batch"],
        print_batch=config["print_batch"],
        years_ahead=config["years_ahead"],

        checkpoint=bestmodelpath,
        # sourcepath=sourcepath + "/",
    )

# Select correct train/test functions
n_epochs = config["n_epochs"]
def train(config, n_epochs, bestmodelpath):
    if config["modeltype"].startswith("3D"):
        return wrapped_train_3DCNN(config, n_epochs, bestmodelpath)
    elif config["modeltype"].startswith("2D"):
        return wrapped_train_2DCNN(config, n_epochs, bestmodelpath)

def test(config, bestmodelpath):
    if config["modeltype"].startswith("3D"):
        return wrapped_test_3DCNN(config, bestmodelpath)
    elif config["modeltype"].startswith("2D"):
        return wrapped_test_2DCNN(config, bestmodelpath)

# Train/test
if args.test_only:
    test(config, init_bestmodelpath)

else:
    if TEST_EPOCHS <= 0:
        bestmodelpath = train(config, n_epochs, init_bestmodelpath)
        bestmodelpath = modelpath + "/torch.nn.parallel.data_parallel.DataParallel/" + bestmodelpath + ".pt"
        test(config, bestmodelpath)
    else:
        bestmodelpath = init_bestmodelpath
        for _ in range(math.ceil(n_epochs / TEST_EPOCHS)):
            bestmodelpath = train(config, TEST_EPOCHS, bestmodelpath)
            bestmodelpath = modelpath + "/torch.nn.parallel.data_parallel.DataParallel/" + bestmodelpath + ".pt"
            test(config, bestmodelpath)

print("\n\n\n\n\n\n\n\n\n\n")