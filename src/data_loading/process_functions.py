from torchvision.transforms import ToTensor
from PIL import Image

# import rasterio
import numpy as np
import torch
import os.path
import rasterio

def to_Tensor(path, name):
    """
    Load Tiff files as tensors
    """
    t = Image.open(path + "/" + name)
    t = ToTensor()(t)
    t = t.squeeze(dim=0)
    return t

'''
def to_Tensor(path, name):
    """
    Load Tiff files as tensors
    """
    t = rasterio.open(path + "/" + name)
    t = ToTensor()(t)
    t = t.squeeze(dim=0)
    return t
'''


def last_to_image(path, year):
    """
    Given path to folder having tiff files for each last band for given year
    returns Tensors with chanels == bands and year as requested in the path
    """
    image = []
    for b in range(1, 5):
        band = Image.open(path + "/last_20%d_%d.tif" % (year, b))
        band = ToTensor()(band)
        image.append(band)
    image = torch.cat(image, dim=0)
    image = image.float()
    return image

def print_pixel_value_percentages(image):
    # Flatten the image to handle all pixels together
    flattened_image = image.view(-1)
    total_pixels = flattened_image.numel()
    
    # Calculate counts for each condition
    num_at_minus_one = (flattened_image == -1).sum().item()
    num_between_minus_one_and_zero = ((flattened_image > -1) & (flattened_image < 0)).sum().item()
    num_at_zero = (flattened_image == 0).sum().item()
    num_greater_than_zero = (flattened_image > 0).sum().item()
    
    # Calculate percentages
    percent_at_minus_one = (num_at_minus_one / total_pixels) * 100
    percent_between_minus_one_and_zero = (num_between_minus_one_and_zero / total_pixels) * 100
    percent_at_zero = (num_at_zero / total_pixels) * 100
    percent_greater_than_zero = (num_greater_than_zero / total_pixels) * 100

    # Print the results
    print(f"Percentage of pixels at -1: {percent_at_minus_one:.2f}%")
    print(f"Percentage of pixels between -1 and 0: {percent_between_minus_one_and_zero:.2f}%")
    print(f"Percentage of pixels at 0: {percent_at_zero:.2f}%")
    print(f"Percentage of pixels greater than 0: {percent_greater_than_zero:.2f}%")


def rescale_image(image, log=False):
    # detach and clone the image so that you don't modify the input, but are returning new tensor.
    rescaled_image = image.data.clone()
    if len(image.shape) == 2:
        rescaled_image = rescaled_image.unsqueeze(dim=0)

    # Compute mean and std only from non masked pixels
    # Spatial coordinates of this pixels are:
    pixels = (rescaled_image[0, :, :] != -1).nonzero()
    pos_pixels = (rescaled_image[0, :, :] >= 0).nonzero()
    
    if log:
        # Add a small epsilon to avoid issues with log(0)
        epsilon = 1e-8
        rescaled_image[:, pos_pixels[:, 0], pos_pixels[:, 1]] = torch.log(
            rescaled_image[:, pos_pixels[:, 0], pos_pixels[:, 1]] + epsilon
        )

    mean = rescaled_image[:, pixels[:, 0], pixels[:, 1]].mean(1, keepdim=True)
    std = rescaled_image[:, pixels[:, 0], pixels[:, 1]].std(1, keepdim=True)
    rescaled_image[:, pixels[:, 0], pixels[:, 1]] -= mean
    rescaled_image[:, pixels[:, 0], pixels[:, 1]] /= std

    if len(image.shape) == 2:
        rescaled_image = rescaled_image.squeeze(dim=0)
        mean = mean.squeeze(dim=0)
        std = std.squeeze(dim=0)

    return (rescaled_image, mean, std)


def if_def_when(lossyear, year, cutoffs=[2, 5, 8]):
    """
    Creates categorical variables for deforestration event given cutoffs.
    Values in cutoffs define the time bins
    Returns len(cutoffs) + 1 categorical layers:
    Example: cutoffs = [2,5,8], num of layers = 4 , considered year = year
    Categories:
    0) if year - lossyear is in [0,2)
    1) if year - lossyear is in [2,5)
    2) if year - lossyear is in [5,8)
    3) 8 years ago or more
    No prior knowledge:
        if loss event is in year > considered year or pixel is non deforested up to 2018+, all categories have value 0
    """
    cutoffs.append(year)
    cutoffs.insert(0, 0)
    lossyear[(lossyear > year)] = 0
    losses = []
    for idx in range(0, len(cutoffs) - 1):
        deff = torch.zeros(lossyear.size())
        deff[
            (cutoffs[idx] <= (year - lossyear)) & ((year - lossyear) < cutoffs[idx + 1])
        ] = 1
        losses.append(deff.float())
    losses = torch.stack(losses)
    # Return Nan values encoded as needed:
    losses[:, (lossyear == -1)] = -1
    return losses

def create_tnsors_pixels(
    year,
    latest_yr,
    tree_p=30,
    cutoffs=[2, 5, 8],
    sourcepath=None,
    rescale=True,
    wherepath=None,
    nightlight_metrics=["median"],
):
    """
    Given year, and cutoffs as defined above returns (and save if wherepath!= None)
        Static tensor,
        Non static tensor,
        list of valid pixels coordinates,
        list of labels corresponding to this valid cordinates

    sourcepath = path to tiff files
    wherepath = in not None, path to where to save the tensors

    Static tensor is identical for any year, hence save only once
    Static tensor has datamask layer and treecover

    Nonstatic tensor has if_deff_when categorical layers and the image landset 7 bands stacked

    Valid pixels are these that meet all the following conditions :
     1. datamask == 1 , eg                        land not water body
     2. tree_cover > tree_p   or   gain == 1      if tree canopy in 2000 > tree_p or became forest up to 2012
     3. lossyear > year   or   lossyear == 0      experienced loss only after that year (or not at all in the study period)
     4. buffer == 0                               is in Madre de Dios area

    for each valid pixel assign label 1 if it is deforested in exactly in year+1 or zero otherwise

    All pixels in the rasters and produced tensors have value 111 in the locations outside Area of Interest and its buffer
    """
    buffer = to_Tensor(sourcepath, "buffer.tif")
    gain = to_Tensor(sourcepath, "gain_20" + str(latest_yr) + ".tif")
    lossyear = to_Tensor(sourcepath, "lossyear_20" + str(latest_yr) + ".tif")
    datamask = to_Tensor(sourcepath, "datamask_20" + str(latest_yr) + ".tif")
    tree_cover = to_Tensor(sourcepath, "treecover2000_20" + str(latest_yr) + ".tif")
    tree_cover = tree_cover.float()
    datamask = datamask.float()

    # skip if no nightlight_metric provided (ie nightlight data not downloaded)
    nightlights = []
    if nightlight_metrics:
        for metric in nightlight_metrics: 
            nightlights.append(to_Tensor(sourcepath, f"/nightlight/nightlight_20{str(year)}_{metric}.tif"))
    
    # Create list of valid pixels coordinates
    pixels = (
        (datamask == 1)
        & ((tree_cover > tree_p) | (gain == 1))  # land (not water body)
        & (  # if forest in 2000 or became forest up to 2012
            (lossyear > year) | (lossyear == 0)
        )
        & (  # experienced loss only after that year (or not at all in the study period)
            buffer == 0
        )
    ).nonzero()  # In area of interest

    # Deforestation in future years 1-5
    future_losses = []
    last_year_of_data = torch.max(lossyear)
    for years_ahead in range(1,5+1):
        if years_ahead + year > last_year_of_data: # labels for year X are deforestation events in X+1
            break
        future_losses.append(torch.logical_and(
            lossyear[pixels[:, 0], pixels[:, 1]] >= (year + 1),
            lossyear[pixels[:, 0], pixels[:, 1]] <= (year + years_ahead)
        )) # can be change to >= (year+1) & <111
    labels = future_losses[0] if len(future_losses) > 0 else [] # easy access to next year's loss

    when = if_def_when(lossyear, year, cutoffs=cutoffs)
    image = last_to_image(sourcepath, year) 

    if rescale: # does not apply to external files, which have already been rescaled
        # Rescale datamask to have values -1 for nan, 0 for land, 1 for water
        datamask[datamask != -1] = datamask[datamask != -1] - 1
        # Rescale tree_cover to have values in [0, 1] and -1 for nan
        tree_cover[tree_cover != -1] = tree_cover[tree_cover != -1] * 0.01

        # Normalize image by channel with -1 values for nan
        image, _, _ = rescale_image(image)

        # skip if no nightlight_metric provided (ie nightlight data not downloaded)
        rescaled_nightlights = []
        if nightlight_metrics:
            for metric_index, nightlight in enumerate(nightlights):
                print_pixel_value_percentages(nightlight)
                norm_nightlight, mean, std = rescale_image(nightlight)
                log_nightlight, log_mean, log_std = rescale_image(nightlight, log=True)
                nightlight = torch.unsqueeze(norm_nightlight, 0)
                log_nightlight = torch.unsqueeze(log_nightlight, 0)
                rescaled_nightlights.append([nightlight, log_nightlight])

                print(f"Nightlight {nightlight_metrics[metric_index]} in 20{year} has mean {mean} and std {std}.")
                print(f"Log Nightlight {nightlight_metrics[metric_index]} in 20{year} has mean {log_mean} and std {log_std}.\n")

    # Create non Static tensor
    image = torch.cat((when, image), dim=0)

    # Creates static tensor
    static_tensors = [datamask, tree_cover]
    static = torch.stack(static_tensors)

    if wherepath:
        # # Will need to delete static.pt if computation is changed
        if not os.path.isfile(wherepath + "/" + "static.pt"):
            torch.save(static, wherepath + "/" + "static.pt")

        torch.save(image, wherepath + "/" + "tensor_%d.pt" % (year))
        torch.save(pixels, wherepath + "/" + "pixels_cord_%d.pt" % (year))
        
        # skip if no nightlight_metric provided (ie nightlight data not downloaded)
        if nightlight_metrics:
            for metric_index, nightlight_data in enumerate(rescaled_nightlights):
                nightlight, log_nightlight = nightlight_data
                torch.save(nightlight, f"{wherepath}/nightlight_{nightlight_metrics[metric_index]}_{year}.pt")
                torch.save(log_nightlight, f"{wherepath}/nightlight_log_{nightlight_metrics[metric_index]}_{year}.pt")
        
        # We have a last year with all data, and we have the year after with everything but labels
        # because labels_X are deforestation events in X+1 (and labelsY_X are for year X+Y)
        for years_ahead, future_loss in enumerate(future_losses, 1):
            if years_ahead + year > last_year_of_data: 
                break
            if years_ahead == 1: # 1 year ahead is just labels (not labels1)
                torch.save(future_loss, wherepath + f"/labels_{year}.pt")
            else: # N years ahead is labelsN
                torch.save(future_loss, wherepath + f"/labels{years_ahead}_{year}.pt")

    return static, image, pixels, labels
