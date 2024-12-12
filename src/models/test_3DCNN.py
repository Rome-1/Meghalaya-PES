"""
SCRIPT FOR TESTING 3DCNN MODELS
"""
import time
import torch
import numpy as np
import rasterio
from rasterio.transform import rowcol
from datetime import datetime
from models.ConvRNN import *
from models.Training import ImbalancedDatasetUnderSampler
from models.Training import test_model
from models.Testing import *
from preprocessing.Data_maker_loader import *
from models.utils import get_model_type


def test_3DCNN(
    wherepath,
    modelpath,
    picspath,
    region,
    start_year,
    end_year,
    input_dim,
    hidden_dim,
    kernel_size,
    levels,
    dropout,
    size,
    pos_weight,
    checkpoint,
    test_times,
    w,
    perc,
    batch_size,
    stop_batch,
    print_batch,
    DSM=False,
    data_layers={},
    years_ahead=1,
    sourcepath=None,
    model_type_name="3D",
):
    modelname = checkpoint.split("/", -1)[-1]

    modeltype = get_model_type(model_type_name, start_year, end_year)
    model = modeltype(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            levels=levels,
            dropout=dropout,
    )

    model = torch.nn.DataParallel(model)

    print("Checkpoint: " + checkpoint)
    # Load to GPU is possible, else CPU
    if torch.cuda.is_available() is False:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint)
        
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="mean", pos_weight=torch.tensor(pos_weight)
    )

    Data = with_DSM(
        size=int(size / 2),
        start_year=start_year,
        end_year=end_year,
        wherepath=wherepath,
        DSM=DSM,
        data_layers=data_layers,
        years_ahead=years_ahead,
    )
    indices = np.array(range(0, len(Data)))

    test_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=indices, times=test_times
    )
    print(datetime.now())
    print("Region: " + region)

    print(
        "percentage valid pixels in year 20%d with label 1: " % (end_year + 1),
        test_sampler.count[1] / sum(test_sampler.count) * 100, "%"
    )
    print("Which correspond to %d number of 1 labeled pixels" % (test_sampler.count[1]))


    start = time.time()

    outputs, targets, coordinates = testing(
        model=model,
        Data=Data,
        criterion=criterion,
        test_sampler=test_sampler,
        w=w,
        perc=perc,
        batch_size=batch_size,
        stop_batch=stop_batch,
        print_batch=print_batch,
        name=modelname,
        path=modelpath,
        region=region,
        save=True,
        valid_pixels_path=wherepath + f"/pixels_cord_{end_year}.pt",
        alt_thresholds=[],#[0.75, 0.9, 1.1, 1.25],
        sourcepath = sourcepath + f"/datamask_2023.tif",
        modelname=modelname,
    )

    heatmap(
        end_year=end_year,
        outputs=outputs, # outputs / scores
        coordinates=coordinates, # coordinates / valid_pixels
        sourcepath=sourcepath, 
        wherepath=wherepath,
        savepath=picspath,
        name=modelname + "_heatmap" + "_forecast_from",
    )

    # outputs, coordinates = forecasting(model = model,
    #                                   Data = Data,
    #                                   year = end_year,
    #                                   batch_size = batch_size,
    #                                   stop_batch = stop_batch,
    #                                   print_batch = print_batch)

    # print("outputs")
    # print(outputs.shape)

    # print("coordinates")
    # print(coordinates.shape)

    # for illustrative purposes create mock outputs and coordinates
    # from torch.distributions import Uniform

    # valid_pixels = torch.load(wherepath + "/pixels_cord_%d.pt" % (end_year))
    # m = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
    # scores = m.sample((len(valid_pixels),))
    # scores = scores.squeeze(dim=1)

    # print("scores")
    # print(scores.shape)

    # print("valid_pixels")
    # print(valid_pixels.shape)


    # # Heatmap

    # sourcepath = '/rds/general/user/jgb116/home/repos/deforestation_forecasting/data/Hansen'
    # heatmap(end_year = end_year,
    #        outputs = outputs, # was scores, but this is just noise, right?
    #        coordinates = valid_pixels,
    #        sourcepath = sourcepath,
    #        wherepath = wherepath,
    #        name = modelname+"mock")

    print("\n\nEND!Total time (in h):", (time.time() - start) / 3600)
