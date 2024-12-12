import os
from models.Testing import forecasting, heatmap
from preprocessing.Data_maker_loader import with_DSM
from models.ConvRNN import *
import random
from random import uniform
import torch

random.seed(300)

region = os.environ.get('REGION')

USER = os.environ.get('USER')

# WHERE TO IMPORT DATA FROM
dir = os.environ.get('DeepForestcast_PATH')
wherepath = dir + "/outputs/" + region + "/tensors"
savepath = dir + "/outputs/" + region + "/out"
modelpath = dir + "/models/" + region + "_models/3D"
picspath = dir + "/models/" + region + "_models/3D/pics"
file = dir + "/models/" + region + "_models/3D/grid_summary/ConvRNN.Conv_3D.mem.txt"

# MAY NEED TO CONFIG THIS DEPENDING ON YOUR SYSTEM
sourcepath = "/home/" + os.environ.get('DeepForestcast_PATH') + "/project/storage/DeepForestcast/outputs/" + region + "/"

start_year = 19
end_year = 23
forecast_year = 24

bestmodel = "torch.nn.parallel.data_parallel.DataParallel_23.8.24_21.3_1.pt"
modelpath = modelpath + "/torch.nn.parallel.data_parallel.DataParallel/"
checkpoint = modelpath + bestmodel
modelname = checkpoint.split("/", -1)[-1]

# set CNN model parameters
size = 45
DSM = False
data_layers = {}
years_ahead = 1

Data = with_DSM(
    size=int(size / 2),
    start_year=start_year,
    end_year=end_year,
    wherepath=wherepath,
    DSM=DSM,
    data_layers=data_layers,
    years_ahead=years_ahead,
)

start_batch = None
stop_batch = None
print_batch = 300
batch_size = 2048 # 1024 might be better; 32 means the post-processing takes ages...sum(coords, []) is not performant # 4096 is too large to fit on GPU memory
save_batch = 1200

if DSM:
    input_dim = (3, 8)
else:
    input_dim = (2, 8)

input_dim = (20, 8)
hidden_dim = [64, 128, 128]
kernel_size = [(3, 3), (2, 3, 3), (3, 3)]
dropout = 0.8 # round(uniform(0.1, 0.8), 1)
levels = 10

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.empty_cache()
if use_cuda:
    print("Device accessed:", torch.cuda.get_device_name(0))
    print("Number of devices:", torch.cuda.device_count())

if (start_year - end_year) % 2 == 0:
    model = Conv_3Dodd(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        levels=[levels],
        dropout=dropout,
    )
else:
    model = Conv_3Deven(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        levels=[levels],
        dropout=dropout,
    )

model = torch.nn.DataParallel(model)
model.to(device)

print("Checkpoint: " + checkpoint)
checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint["model_state_dict"])

# Verify DataParallel is being used
if isinstance(model, torch.nn.DataParallel):
    print(f"Model is running on {len(model.device_ids)} GPUs.")
else:
    print("Model is not running on multiple GPUs.")


savepath = dir + "/forecasts/" + region

outputs, coordinates = forecasting(
    model=model,
    device=device,
    Data=Data,
    year=forecast_year,
    batch_size=batch_size,
    start_batch=start_batch,
    stop_batch=stop_batch,
    print_batch=print_batch,
    save_batch=save_batch,
    save=True,
    path=savepath,
    name=modelname,
)

heatmap(
    end_year=end_year,
    outputs=outputs,
    coordinates=coordinates,
    sourcepath=sourcepath,
    wherepath=wherepath,
    savepath=savepath,
    name=modelname,
    output_year=forecast_year,
    msk_yr=end_year,
)