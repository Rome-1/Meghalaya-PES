import os
import glob
import re
import gzip
import shutil
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt


def crop_reformat_tif(datapath, savename, savepath, refRasterPath, boundaryPath):
    """
    Load tif, crop to shp, and make it match raster (CRS, resolution, coords, etc) without changing dtype
    Assumes prepare_data in main data load has already been run to generate a buffer.tif, used as the reference raster to determine CRS, etc
    """
    boundary = gpd.read_file(boundaryPath)

    # Open the reference file to get the metadata
    with rasterio.open(refRasterPath) as ref:
        ref_raster = ref.meta.copy()

    with rasterio.open(datapath) as src:
        clip, out_trans = mask(src, boundary.geometry, nodata=-1, crop=True)
        dtype = src.meta['dtype']

    meta = {
        'driver': 'GTiff',
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': ref_raster["transform"],
        'height': ref_raster['height'],
        'width': ref_raster['width'],
        'dtype': dtype,
    }

    with rasterio.open(savepath + savename, "w", **meta) as out:
        out.write(clip)
    print("Saved:", savepath + savename)

def save_for_visual_inspection(loadpath, savepath, savename):
    """TODO
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Save for visual inspection
    with rasterio.open(loadpath + savename) as saved_data:
        raster = saved_data.read(1)
        raster = np.where(raster > 0, np.log(raster), raster)
        
        plt.figure(figsize=(50, 25))
        plt.title(f"{raster.shape}: {savename}")
        plt.axis('off')
        show(raster, cmap='gray', title=savename)
        plt.tight_layout()
        plt.savefig(savepath + savename + ".png", format="png")

def prepare_nightlight_data(years, nightlightDwnldPath, savepath, refBoundary, referenceShp, metrics=["median", "maximum"]):
    """
    Prepares VIIRS nightlight data for analysis by downloading, unzipping, 
    cropping, and reformatting the data based on the provided reference boundary 
    and shapefile. The processed data is saved for further use and visual inspection.
    - This function requires the nightlight data to be downloaded using 
      `download_nightlight_data` and expects the data to be available in `.tif` or `.tif.gz` format.
    - The processed files are saved with a naming convention `nightlight_<year>_<metric>.tif`.
    
    Parameters:
    - years (list[int]): List of years to process, e.g., [2012, 2013, ..., 2023].
    - nightlightDwnldPath (str): Path where the downloaded nightlight data is stored.
    - savepath (str): Path where processed data should be saved.
    - refBoundary (str): Path to the reference boundary file for cropping.
    - referenceShp (str): Path to the shapefile for spatial reference.
    - metrics (list[str]): List of nightlight metrics to process, e.g., ["median", "maximum"].
    """

    for metric in metrics:
        for year in years:

            # Input Validation
            if year in set(range(12,23+1)): # Convert year from 2000 to year from 0
                year += 2000
            if year < 2012 or year > 2023:
                raise Exception(f"VIIRS nighttime satellite data not available for {year}")

            # Filenaming
            datapath = nightlightDwnldPath + f"viirs_{year}_{metric}.tif" # set by download_nightlight_data
            savename = f"nightlight_{year}_{metric}.tif"

            # Uncompress file if necessary
            if not os.path.exists(datapath):
                if not os.path.exists(datapath + ".gz"):
                    raise Exception(f"Download nightlight data first! Could not find: {datapath + ".gz"}")
                
                print("Unzipping:", datapath + ".gz")
                # Decompress the zipped .tif.gz file
                with gzip.open(datapath + ".gz", 'rb') as f_in:
                    with open(datapath, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove the zipped .tif.gz file
                os.remove(datapath + ".gz")
                print("Deleted:", datapath + ".gz")

            # Transform the raster
            crop_reformat_tif(datapath, savename, savepath, refBoundary, referenceShp)

            # Save for visual inspection
            save_for_visual_inspection(savepath,savepath + "/images/", savename)
