"""
SCRIPT FOR TESTING 2DCNN MODELS
"""
import time
import torch
import numpy as np
from datetime import datetime
from models.CNN import CNNmodel
from models.Training import ImbalancedDatasetUnderSampler
from models.Testing import *
from preprocessing.Data_maker_loader import *

def test_2DCNN(
    wherepath,
    modelpath,
    start_year,
    end_year,
    input_dim,
    hidden_dim1,
    hidden_dim2,
    hidden_dim3,
    hidden_dim4,
    region,
    kernel_size,
    stride,
    padding,
    levels,
    test_times,
    perc,
    w,
    pos_weight,
    dropout,
    size,
    stop_batch,
    batch_size,
    print_batch,
    checkpoint,
    data_layers={},
    years_ahead=1,
    model_type_name="2D",
):
    
    start = time.time()
    modelname = checkpoint.split("/", -1)[-1]

    # Set up model
    hidden_dim = [
        hidden_dim1,
        hidden_dim2,
        hidden_dim3,
        hidden_dim4,
    ]

    model = CNNmodel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        levels=levels,
        dropout=dropout,
        stride=stride,
        padding=padding,
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
        data_layers=data_layers,
        years_ahead=years_ahead,
        type="2D",
    )
    indices = np.array(range(0, len(Data)))

    test_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=indices, times=test_times
    )

    print(datetime.now())
    print("Region: " + region)
    print(modelname)
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
        path=modelpath + "/torch.nn.parallel.data_parallel.DataParallel/CNNmodel",
        save=True,
    )

    # outputs, coordinates = forecasting(model = model,
    #                                   Data = Data,
    #                                   year = end_year,
    #                                   batch_size = batch_size,
    #                                   stop_batch = stop_batch,
    #                                   print_batch = print_batch)

    print("\n\nEND!Total time (in h):", (time.time() - start) / 3600)
