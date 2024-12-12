import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from models.utils import parse_model_forward
import scikitplot
from models.Training import AUC_CM
from scipy.special import expit as Sigmoid
import os
import time
from preprocessing.Data_maker_loader import *

import rasterio as rio
from rasterio.enums import Resampling
import matplotlib
import matplotlib.pyplot as plt
import itertools

RECON_CRITERION = torch.nn.MSELoss()
RECON_LOSS_WEIGHT = 0.5

def testing(
    model,
    Data,
    criterion,
    test_sampler,
    w,
    perc,
    batch_size,
    stop_batch,
    print_batch,
    name=None,
    path=None,
    region=None,
    save=False,
    valid_pixels_path=None,
    alt_thresholds=[],
    sourcepath=None,
    modelname=None,
    recon_loss_weight=RECON_LOSS_WEIGHT,
    reconstruction_criterion=RECON_CRITERION,
):
    print("\nTESTING ROUTINE COMMENCING", flush=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    if use_cuda:
        print("Device accessed:", torch.cuda.get_device_name(0), flush=True)
        print("Number of devices:", torch.cuda.device_count(), flush=True)
    model.eval()

    Test_sampler = SubsetRandomSampler(test_sampler.classIndexes_unsampled)
    test_loader = DataLoader(
        Data, sampler=Test_sampler, batch_size=batch_size, drop_last=True
    )
    require_sigmoid = isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss)

    # initialize lists to monitor test loss, accuracy and coordinates
    losses = []
    outputs = []
    targets = []
    coordinates = []

    test_start = time.time()
    for batch, (data, target, cor) in enumerate(test_loader):

        if type(data) == type([]):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
        else:
            data = data.to(device)
        target = target.to(device)
        cor = cor.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        model_output = model.forward(data, sigmoid=not require_sigmoid)
        output, loss, _, _ = parse_model_forward(model_output, target, data, criterion, reconstruction_criterion, recon_loss_weight)

        losses.append(loss.item())
        outputs.append(list(output.cpu().data))
        targets.append(list(target.cpu().data))
        coordinates.append(list(cor.cpu().data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break
        # =====================================================================
        # =====================================================================
        if print_batch:
            if batch % print_batch == 0:
                print("\tBatch:", batch, "\tLoss:", loss.item(), flush=True)
        # =====================================================================

    outputs = sum(outputs, []) # TODO see heatmap for more performant (and pythonic) version
    targets = sum(targets, []) # TODO see heatmap for more performant (and pythonic) version
    coordinates = sum(coordinates, []) # TODO see heatmap for more performant (and pythonic) version
    t_time = time.time() - test_start
    print(
        "\n\tTime to load %d test batches of size %d : %3.4f hours\n"
        % (batch, batch_size, t_time / (3600)), flush=True
    )

    losses = np.average(losses)

    print("\tTest Loss: ", losses)
    AUC, cost = AUC_CM(targets, outputs, w, perc, sigmoid=require_sigmoid, coordinates=coordinates, savepath=path + f"/outputs_{region}/", label = "test", valid_pixels_path=valid_pixels_path, alt_thresholds=alt_thresholds, transform_crs_source=sourcepath, modelname=modelname)
    if require_sigmoid:
        outputs = np.array(Sigmoid(outputs))
    probas_per_class = np.stack((1 - outputs, outputs), axis=1)
    roc = scikitplot.metrics.plot_roc(np.array(targets), probas_per_class)

    coordinates = torch.stack(coordinates, dim=0)
    outputs = torch.tensor(outputs)

    if save:
        path = path + "/outputs"
        if region is not None:
            path = path + "_" + region
        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory ", path, " Created ")
        roc.get_figure().savefig(
            os.path.join(path, name + "_Test_ROC.png"), bbox_inches="tight"
        )
        torch.save(
            {"model_outputs": outputs, "targets": targets, "coordinates": coordinates},
            os.path.join(path, name + "_outputs.pt"),
        )
        print("saved at :", os.path.join(path, name + "_outputs.pt"))

    return outputs, targets, coordinates


def forecasting(
    model,
    device,
    Data,
    year,
    batch_size,
    start_batch=None,
    stop_batch=None,
    print_batch=None,
    save_batch=None,
    name=None,
    path=None,
    save=False,
    recon_loss_weight=RECON_LOSS_WEIGHT,
    reconstruction_criterion=RECON_CRITERION,
):  
    path = path + "/outputs"

    predict_loader = DataLoader(
        Data, shuffle=False, batch_size=batch_size, drop_last=False
    )

    outputs = []
    coordinates = []

    for batch, (data, target, cor) in enumerate(predict_loader):
        if start_batch and batch < start_batch:
            continue

        if type(data) == type([]):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
        else:
            data = data.to(device)
        cor = cor.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        model_output = model.forward(data, sigmoid=True)
        output, loss, _, _ = parse_model_forward(model_output, target, data, None, None, None)

        outputs.append(list(output.cpu().data))
        coordinates.append(list(cor.cpu().data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break
        # =====================================================================
        # =====================================================================
        if print_batch:
            if batch % print_batch == 0:
                print("\tBatch:", batch, "/", len(predict_loader))
        # =====================================================================
        # =====================================================================
        # Try saving checkpoints
        try:
            if save and save_batch:
                if batch % save_batch == 0:
                    if not os.path.exists(path):
                        os.mkdir(path)
                        print("Directory ", path, " Created ")
                    torch.save(
                        {"model_outputs": outputs, "coordinates": coordinates},
                        os.path.join(path, name + "_forecast_outputs_from%d_checkpoint%d.pt" % (year, batch)),
                    )
                    print(
                        "saved checkpoint at :",
                        os.path.join(path, name + "_forecast_outputs_from%d_checkpoint%d.pt" % (year, batch)),
                    )
        except:
            pass
        # =====================================================================

    # Flush 
    print("Predicting finished", flush=True)

    outputs = sum(outputs, [])
    coordinates = sum(coordinates, [])
    coordinates = torch.stack(coordinates, dim=0)
    outputs = torch.tensor(outputs)

    print("Saving...", flush=True)

    if save:
        if not os.path.exists(path):
            os.mkdir(path)
            print("Directory ", path, " Created ")
        torch.save(
            {"model_outputs": outputs, "coordinates": coordinates},
            os.path.join(path, name + "_forecast_outputs_from%d.pt" % (year)),
        )
        print(
            "forecasts saved at :",
            os.path.join(path, name + "_forecast_outputs_from%d.pt" % (year)),
            flush=True
        )

    return outputs, coordinates


def forecasting_split(
    model,
    Data,
    year,
    batch_size,
    stop_batch,
    print_batch,
    name=None,
    path=None,
    save=False,
    slice=1,
):

    predict_loader = DataLoader(
        Data, shuffle=False, batch_size=batch_size, drop_last=False
    )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    outputs = []
    coordinates = []

    for batch, (data, cor) in enumerate(predict_loader):

        if type(data) == type([]):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
        else:
            data = data.to(device)
        cor = cor.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data, sigmoid=True)
        outputs.append(list(output.cpu().data))
        coordinates.append(list(cor.cpu().data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break
        # =====================================================================
        # =====================================================================
        if print_batch:
            if batch % print_batch == 0:
                print("\tBatch:", batch)
        # =====================================================================

    outputs = sum(outputs, [])
    coordinates = sum(coordinates, [])
    coordinates = torch.stack(coordinates, dim=0)
    outputs = torch.tensor(outputs)

    if save:
        path = path + "/forecasts"
        if not os.path.exists(path):
            os.mkdir(path)
            print("Directory ", path, " Created ")
        torch.save(
            {"model_outputs": outputs, "coordinates": coordinates},
            os.path.join(
                path, name + "Madre_forecast_outputs_from%d_%d.pt" % (year, slice)
            ),
        )
        print(
            "saved at :",
            os.path.join(
                path, name + "Madre_forecast_outputs_from%d_%d.pt" % (year, slice)
            ),
        )
        np.savetxt(
            os.path.join(
                path, name + "Madre_forecast_outputs_from%d_%d" % (year, slice) + ".csv"
            ),
            np.c_[coordinates.numpy(), outputs.numpy()],
            delimiter=",",
        )
        print(
            "saved: ",
            name + "Madre_forecast_outputs_from%d_%d" % (year, slice) + ".csv",
        )

    return outputs, coordinates



def heatmap(
    end_year, outputs, coordinates, sourcepath, wherepath, savepath, name, output_year=None, msk_yr=23
):
    if output_year is None:
        output_year = end_year

    print("\nHeatmap:", flush=True)


    datamask = to_Tensor(sourcepath, "datamask_20%d.tif" % (msk_yr))
    print("Loading pixels...", flush=True)
    valid_pixels = torch.load(wherepath + "/pixels_cord_%d.pt" % min(end_year, 23))
    print("Loaded pixels. Setting up datamask..", flush=True)

    datamask[valid_pixels[:, 0], valid_pixels[:, 1]] = 0

    # Create Heatmap
    print("Valid pixels to predict in year 20%d" % (end_year))
    colors = ["white", "green", "grey", "blue"]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.matshow(datamask, cmap=matplotlib.colors.ListedColormap(colors))
    fig.show()
    fig.savefig(savepath + "/" +  name + f"_valid_pixels_20{output_year:d}.png")
    print("png valid pixels saved at: ", savepath + "/" + name + f"_valid_pixels_20{output_year:d}.png")

    # print("Stacking...", flush=True)
    # coordinates = torch.stack(coordinates, dim=0)
    # print("Coords shape:", coordinates.shape)
    # # if require_sigmoid:
    # # outputs = np.array(Sigmoid(outputs))
    # outputs = torch.tensor(outputs)

    heatmap = torch.ones(datamask.shape) * (-1)
    heatmap[coordinates[:, 0], coordinates[:, 1]] = outputs
    heatmap = heatmap.numpy()
    heatmap[heatmap == -1] = None

    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(111)
    ax.matshow(
        heatmap, cmap=matplotlib.cm.Spectral_r, interpolation="none", vmin=0, vmax=1
    )
    fig.show()
    fig.savefig(savepath + "/" +  name + f"_heatmap_20{output_year:d}.png")
    print("PNG heatmap saved at: ", savepath + "/" + name + f"_heatmap_20{output_year:d}.png")

    print("Saving to tif...", flush=True)
    heatmap[heatmap == None] = np.nan
    with rio.open(sourcepath + "datamask_20%d.tif" % (msk_yr)) as src:
        ras_data = src.read()
        ras_meta = src.profile
        
        # # For downsampled copy
        # new_height = src.height // 64
        # new_width = src.width // 64

    # make any necessary changes to raster properties, e.g.:
    ras_meta.update({
        "dtype": "float32",       # Consider float16 if supported
        "nodata": np.nan,
        # "compress": "lzw"         # or 'deflate' for DEFLATE compression
    })

    # # Update the metadata
    # ras_meta_downsampled = ras_meta.copy()
    # ras_meta_downsampled.update({
    #     "height": src.height // 64,
    #     "width": src.width // 64,
    #     "transform": src.transform * src.transform.scale(
    #         (src.width / new_width),
    #         (src.height / new_height)
    #     )
    # })

    # where to save the output .tif
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print("Directory ", savepath, " Created ")

    print("Actually saving to tif...", flush=True)
    # Save heatmap
    with rio.open(savepath + "/" +  name + f"_20{output_year:d}.tif", "w", **ras_meta) as dst:
        dst.write(heatmap, 1)

    # # Make a second, downsampled heatmap
    # with rio.open(savepath + "/" +  name + f"_20{end_year:d}.tif", "w", **ras_meta) as src:

    #     # Resample data to target shape
    #     heatmap_downsized = src.read(
    #         out_shape=(src.count, new_height, new_width),
    #         resampling=Resampling.bilinear
    #     )

    # # Save downsampled heatmap
    # with rio.open(savepath + "/" +  name + f"_downsampled_20{end_year:d}.tif", "w", **ras_meta) as dst:
    #     dst.write(heatmap_downsized, 1)

    print(
        "Heatmap min: %.4f, max: %.4f, mean: %.4f;"
        % (np.nanmin(heatmap), np.nanmax(heatmap), np.nanmean(heatmap)), flush=True
    )
    print("heatmap saved at: ", savepath + "/" + name + f"_20{output_year:d}.tif")
    # print("downsampled heatmap saved at: ", savepath + name + f"_downsampled_20{end_year:d}.tif")


# For saving as .csv batches
# Seems incomplete to me
def forecasting2(
    model, Data, year, batch_size, stop_batch, print_batch, name=None, path=None
):

    predict_loader = DataLoader(Data, shuffle=True, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    outputs = []
    coordinates = []

    for batch, (data, cor) in enumerate(predict_loader):

        if type(data) == type([]):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
        else:
            data = data.to(device)
        cor = cor.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data, sigmoid=True)
        #        outputs.append(list(output.cpu().data))
        #        coordinates.append(list(cor.cpu().data))
        # =====================================================================
        if stop_batch:
            if batch == stop_batch:
                break
        # =====================================================================
        # =====================================================================
        print("\tBatch:", batch)
        np.savetxt(
            path + str(batch) + "batch.csv",
            np.c_[cor.cpu().data, output.cpu().data],
            delimiter=",",
        )
        print("saved: ", path + str(batch) + "batch.csv")
        # =====================================================================
