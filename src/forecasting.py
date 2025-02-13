from models.utils import get_model_type
from preprocessing.Data_maker_loader import with_DSM
from models.Testing import forecasting, heatmap
from data_loading.get_setup import compute_input_dimensions, fix_randomness, get_forecast_arguments, get_paths
import torch

fix_randomness()

USER = "rtt8"
DIRECTORY = f"/gpfs/gibbs/project/pande/{USER}/storage/DeepForestcast"

args, config, forecast_modelpath = get_forecast_arguments()
REGION = config["region"]

sourcepath, wherepath, savepath, modelpath, picspath, file = get_paths(DIRECTORY, REGION, config["modeltype"])
input_dim_2D, input_dim_3D = compute_input_dimensions(config["data_layers"])

modeltype = get_model_type(config["modeltype"], config["start_year"], config["end_year"])
hidden_dim = [config["hidden_dim1"], config["hidden_dim2"], config["hidden_dim3"]]
model = modeltype(
        input_dim=input_dim_3D,
        hidden_dim=hidden_dim,
        kernel_size=config['kernel_size'],
        levels=config['levels'],
        dropout=config["dropout"],
)

model = torch.nn.DataParallel(model)
modelname = forecast_modelpath.split("/", -1)[-1]

print("Forecast model path: " + forecast_modelpath)
# Load to GPU is possible, else CPU
if torch.cuda.is_available() is False:
    checkpoint = torch.load(forecast_modelpath, map_location=torch.device('cpu'))
else:
    checkpoint = torch.load(forecast_modelpath)
    
model.load_state_dict(checkpoint["model_state_dict"])

Data = with_DSM(
    size=int(config["size"] / 2),
    start_year=config["start_year"],
    end_year=config["end_year"],
    wherepath=wherepath,
    data_layers=config["data_layers"],
    years_ahead=1,
    forecasting=True,
)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.empty_cache()
if use_cuda:
    print("Device accessed:", torch.cuda.get_device_name(0))
    print("Number of devices:", torch.cuda.device_count())
model = model.to(device)
model.eval()

# Verify DataParallel is being used
if isinstance(model, torch.nn.DataParallel):
    print(f"Model is running on {len(model.device_ids)} GPUs.")
else:
    print("Model is not running on multiple GPUs.")

savepath = DIRECTORY + "/forecasts/" + REGION

with torch.no_grad():
    outputs, coordinates = forecasting(
        model=model,
        device=device,
        Data=Data,
        year=config["forecast_year"],
        batch_size=config["batch_size"],
        start_batch=config["start_batch"],
        stop_batch=config["stop_batch"],
        print_batch=config["print_batch"],
        save_batch=config["save_batch"],
        save=True,
        path=savepath,
        name=modelname,
    )

heatmap(
    end_year=config["end_year"],
    outputs=outputs,
    coordinates=coordinates,
    sourcepath=sourcepath,
    wherepath=wherepath,
    savepath=savepath,
    name=modelname,
    output_year=config["forecast_year"],
    msk_yr=config["end_year"],
)