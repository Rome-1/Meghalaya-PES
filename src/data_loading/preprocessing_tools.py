import os
import json
import glob
import shutil
from tempfile import NamedTemporaryFile
from pprint import pprint

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
from torchvision.transforms import ToTensor

import rasterio
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

import matplotlib.pyplot as plt
import matplotlib

from dsm import reproj_match, to_Tensor


DeepForestcast_path = os.environ.get('DeepForestcast_PATH')
dir = os.environ.get('FEATURES_PATH')
region = os.environ.get('REGION')

boundaryPath = DeepForestcast_path + "/data/" + region + "/boundary/boundary.shp"
save_dest = DeepForestcast_path + f"/src/data_loading/testing/"
tif_right = DeepForestcast_path + "/outputs/meghalaya_only/buffer.tif"

def visualize_pt(pt_path, label = ""):
    # Load the .pt file
    data = torch.load(pt_path)

    # Convert the tensor to a NumPy array for visualization
    data_np = data.numpy()

    # Plot the entire 2D matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(data_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(f"{data_np.shape} Matrix Visualization")
    plt.show()
    plt.savefig(DeepForestcast_path + f"/src/data_loading/testing/{label}.png", format="png")


def visualize_tif(tif_path, label = "", PLOT_DIST = True, high_res=False): # assumed single-band
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        print("Label:", label)
        print("Shape:", data.shape)
        print("Min:", np.min(data))
        print("Max:", np.max(data))
        print("Unique values:", np.unique(data), "(" + str(np.unique(data).shape[0]) + ")")

        height, width = data.shape
        aspect_ratio = width / height
        scale = 5

        # Set nan values to -2
        data[np.isnan(data)] = -2

        if high_res:
            scale = 50

        # Plot using matplotlib
        plt.figure(figsize=(scale * aspect_ratio, scale))
        plt.title(f"{data.shape}: {tif_path}")
        plt.axis('off')
        # show(data, cmap='gray', title=tif_path)
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.savefig(save_dest + label + ".png", format="png")
        plt.close()

        # Plot distribution of values
        if PLOT_DIST:
            plt.figure(figsize=(10, 6))
            flattened_array = data.flatten()
            flattened_array = flattened_array[flattened_array != -1] # skip -1 values
            plt.hist(flattened_array, bins=50, edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of ' + label)
            plt.savefig(save_dest + label + "_dist.png", format="png")
            plt.show()
            plt.close()

def reproject_tif(tif_wrong, tif_right, tif_corrected):

    # Open the wrong GeoTIFF
    with rasterio.open(tif_wrong) as src_wrong:
        # right_dtype = src_wrong.dtype
        # Open the right GeoTIFF to get the correct CRS and resolution
        with rasterio.open(tif_right) as src_right:
            # Get the metadata and CRS of the right GeoTIFF
            right_meta = src_right.meta.copy()
            right_crs = src_right.crs
            right_transform = src_right.transform
            right_width = src_right.width
            right_height = src_right.height

            # Calculate the transform and dimensions for the wrong GeoTIFF to match the right one
            transform, width, height = calculate_default_transform(
                src_wrong.crs, right_crs, src_wrong.width, src_wrong.height, *src_wrong.bounds)

            # Update metadata of the wrong GeoTIFF to match the right one
            wrong_meta = src_wrong.meta.copy()
            wrong_meta.update({
                'crs': right_crs,
                'transform': right_transform,
                'width': right_width,
                'height': right_height
            })

            # Reproject and resample the wrong GeoTIFF to match the right one
            with rasterio.open(tif_corrected, 'w', **wrong_meta) as dst:
                for i in range(1, src_wrong.count + 1):
                    reproject(
                        source=rasterio.band(src_wrong, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src_wrong.transform,
                        src_crs=src_wrong.crs,
                        dst_transform=right_transform,
                        dst_crs=right_crs,
                        resampling=Resampling.bilinear,
                        # dtype=right_dtype
                    )
                
    print(f"Reprojected and resampled GeoTIFF saved to {tif_corrected}")

def crop_tif_with_mask(input_tif, mask_tif, output_tif): # assuming buffer is 0 and 1s
    with rasterio.open(input_tif) as src:
        image = src.read(1)
        metadata = src.meta.copy()
    
    with rasterio.open(mask_tif) as mask_src:
        mask = mask_src.read(1)  # Reading the first band of the mask (should be 0s and 1s)

    # Where the mask is 0, set the corresponding pixel in the image to -1
    cropped_image = np.where(mask == -1, -1, image)
    
    metadata.update({
        'dtype': 'float32'  # Assuming float32 since we're adding -1; adjust based on your input
    })

    with rasterio.open(output_tif, 'w', **metadata) as dst:
        dst.write(cropped_image.astype(np.float32), 1)  # Write the first band

    print(f"Modified image saved to {output_tif}")

def apply_transforms_in_place(tif_path, epsilon=1e-10):
    # Create a temporary file to store the log-transformed data
    with NamedTemporaryFile(delete=False, suffix='.tif') as tmpfile:
        temp_output_path = tmpfile.name

    # Open the original file and apply the log transformation
    with rasterio.open(tif_path) as src:
        data = src.read(1)  # Read the first band (assuming single-band TIFF)
        
        print("Original Data:")
        print("Min:", np.min(data))
        print("Max:", np.max(data))
        print("Count of unique values:", np.unique(data).shape[0])
        data_transformed = data
        
        # # Apply logarithmic transformation, handling non-positive values with epsilon
        # print("Applying Log")
        # data_transformed = np.where(data_transformed > 0, np.log(data_transformed), np.log(epsilon))
        # # Clip at > e50
        # data_transformed = np.where(data_transformed < 50, data_transformed, 0)

        # Clip at > 1e30
        clip_at = 1e30
        print("Clipping at", clip_at)
        data_transformed = np.where(data_transformed < clip_at, data_transformed, 0)

        print("Transformed Data:")
        print("Min:", np.min(data_transformed))
        print("Max:", np.max(data_transformed))
        print("Count of unique values after transformations:", np.unique(data_transformed).shape[0])
        
        # Copy metadata (profile) from the original file
        profile = src.profile
        profile.update(dtype=rasterio.float32)  # Update to floating point type

        # Write log-transformed data to the temporary file
        with rasterio.open(temp_output_path, 'w', **profile) as dst:
            dst.write(data_transformed.astype(rasterio.float32), 1)

    # Replace the original file with the transformed file
    os.remove(tif_path)  # Delete the original file
    shutil.move(temp_output_path, tif_path)  # Rename temp file to original file name

def visualize_multiband_tif(input_path, label = ""):

    with rasterio.open(input_path) as src:
        num_bands = src.count
        print("File:", input_path)
        print("Channels:", num_bands)

        n_cols = int(np.ceil(np.sqrt(num_bands)))
        n_rows = max(int(np.ceil(num_bands / n_cols)), 2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(50, 25))  # Create subplots
        
        for i in range(1, num_bands + 1):  # Band indexing starts at 1 in rasterio
            band = src.read(i)
            # print("Count unique", np.unique(band)) # Can be slow
            row = (i-1) // n_cols
            col = (i-1) % n_cols
            ax = axes[row, col] if num_bands > 1 else axes  # Handle single band case
            ax.imshow(band, cmap="gray")
            ax.set_title(f'Band {i}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.savefig(DeepForestcast_path + f"/src/data_loading/testing/{label}_{i}.png", format="png")

    print(f"{label} visualized at:", DeepForestcast_path + f"/src/data_loading/testing/{label}_{i}.png")

def rescale_tif(input_tif, output_tif):
    print("Rescaling", input_tif)
    
    # Step 1: Open the input TIFF
    with rasterio.open(input_tif) as src:
        image = src.read(1)  # Reading the first (and possibly only) band
        
        mask = image != -1  # Boolean mask: True for pixels that are not -1
        
        # Get the min and max of the pixels that are not -1
        min_val = image[mask].min()
        max_val = image[mask].max()
        
        print(f"Original Min (excluding -1): {min_val}, Max (excluding -1): {max_val}")
        
        # Step 3: Rescale only the pixels that are not -1
        rescaled_image = image.astype(np.float32)  # Ensure the image is float32
        
        # Apply the rescaling only to the valid pixels (those not equal to -1)
        rescaled_image[mask] = (image[mask] - min_val) / (max_val - min_val)
        
        # Step 4: Prepare the metadata for the output file
        metadata = src.meta.copy()
        metadata.update({
            'dtype': 'float32',  # Save as float32 since the values are continuous
            'driver': 'GTiff'
        })
        
    # Save the rescaled image to the output TIFF
    with rasterio.open(output_tif, 'w', **metadata) as dst:
        dst.write(rescaled_image.astype(np.float32), 1)  # Write the first band
    
    print(f"Rescaled image saved to {output_tif}")

def convert_to_one_hot_tif(input_tif, output_tif, num_classes=13):
    # Step 1: Read the input TIFF
    with rasterio.open(input_tif) as src:
        # Read the image data
        
        image = src.read(1)  # Assuming it's a single-band classification TIFF (1-indexed in rasterio)
        
        # Extract height and width of the image
        height, width = image.shape
        unique_vals = np.unique(image).shape[0]

        # Double check formatting of image is correct
        # Because of how this is set up, we use values to index to one-hot encoding
        # so value 3 will correspond to 0010, for example
        if unique_vals != num_classes or num_classes != np.max(image) + 1:
            print("Max+1, unique, and num_clases do not agree")
            print("Max:", np.max(image) + 1)
            print("Num classes:", num_classes)
            print("Unique values:", np.unique(image), "(" + str(np.unique(image).shape[0]) + ")")
            raise Exception("Disagreement in one-hot encoding values")

        # Save up to 50 bands
        max_bands = 50
        if num_classes > max_bands:
            print("Image would have one-hot encoding of max:", np.max(image), "unique:", np.unique(image))
            print("...which is greater than max:", max_bands)
            raise Exception("Too many values for one-hot encoding in " + input_tif)
        
        # Step 2: Convert to one-hot encoding using np.eye
        # Create a one-hot encoded matrix with shape (height, width, num_classes)
        one_hot_encoded_image = np.eye(num_classes)[image]  # Shape: (height, width, num_classes)
        
        # Convert to uint8 (assuming the values are either 0 or 1)
        one_hot_encoded_image = one_hot_encoded_image.astype(np.uint8)
        
        # Step 3: Define the metadata for the output TIFF file
        transform = src.transform  # Use the same transform as the input file
        metadata = src.meta.copy()
        metadata.update({
            'driver': 'GTiff',
            'dtype': 'uint8',   # One-hot encoded values are either 0 or 1, so uint8 is sufficient
            'count': num_classes - 1,  # Number of bands is equal to the number of classes (- 1 invalid pixels with value of 0)
            'width': width,
            'height': height,
        })
        
    # Step 4: Write the one-hot encoded image to the output TIF
    with rasterio.open(output_tif, 'w', **metadata) as dst:
        for i in range(1, num_classes): # skipping 0th channel, which defines invalid pixels in agroclimate
            dst.write(one_hot_encoded_image[:, :, i], i)  # Write each band (i+1 because bands are 1-indexed in rasterio)

    print("Converted to one-hot encodings at:", output_tif)


def tensor_distribution(tensor, bins=10, take_log=False, show_histogram=False, save_path="tensor_distribution"):
    """
    Computes and saves distribution statistics for a PyTorch tensor.

    Parameters:
    tensor (torch.Tensor): Input tensor.
    bins (int): Number of bins for the histogram.
    show_histogram (bool): If True, saves the histogram as an image.
    save_path (str): Base path for saving statistics and histogram.

    """
    if tensor.dtype == torch.bool:
        tensor = tensor.float()

    if take_log is True:
        epsilon = 1e-10

        num_values_in_range = ((tensor > 0) & (tensor <= epsilon)).sum().item()
        total_values = ((tensor != -1)).sum().item()
        percent_in_range = (num_values_in_range / total_values) * 100
        print("Percent under epsilon:", percent_in_range, "(", num_values_in_range, ")")

        tensor = torch.where(tensor > 0, torch.log(tensor + epsilon), tensor)

    # Calculate summary statistics
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    # Create output dictionary
    stats = {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}

    pprint(stats)
    
    # Optional: Save histogram if show_histogram is True
    if show_histogram:
        plt.hist(tensor.numpy().flatten(), bins=bins, color='blue', alpha=0.7)
        plt.title('Tensor Value Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        hist_path = f"{save_path}_histogram.png"
        plt.savefig(hist_path)
        plt.close()
        print(f"Histogram saved to {hist_path}")
    
def rescale_image(image):
    # detach and clone the image so that you don't modify the input, but are returning new tensor.
    rescaled_image = image.data.clone()
    if len(image.shape) == 2:
        rescaled_image = rescaled_image.unsqueeze(dim=0)
    # Compute mean and std only from non masked pixels
    # Spatial coordinates of this pixels are:
    pixels = (rescaled_image[0, :, :] != -1).nonzero()
    mean = rescaled_image[:, pixels[:, 0], pixels[:, 1]].mean(1, keepdim=True)
    std = rescaled_image[:, pixels[:, 0], pixels[:, 1]].std(1, keepdim=True)
    rescaled_image[:, pixels[:, 0], pixels[:, 1]] -= mean
    rescaled_image[:, pixels[:, 0], pixels[:, 1]] /= std
    if len(image.shape) == 2:
        rescaled_image = rescaled_image.squeeze(dim=0)
        mean = mean.squeeze(dim=0)
        std = std.squeeze(dim=0)
    return (rescaled_image, mean, std)
