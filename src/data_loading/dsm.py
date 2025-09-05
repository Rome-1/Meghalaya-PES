from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import torch
from preprocessing.Data_maker_loader import rescale_image
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
import geopandas as gpd

def to_Tensor(path,name):
    """
    Load Tiff files as tensors
    """
    t = Image.open(path+"/"+name)
    t = ToTensor()(t)
    t = t.squeeze(dim = 0)
    return(t)  


def reproj_match(infile, match, outfile):
    """Reproject a file to match the shape and projection of existing raster. 
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 0})
        print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    

def DSM_to_tensor(dwnldpath, wherepath, outpath, boundaryPath):
    datamask = to_Tensor(wherepath,'datamask_2023.tif')
    boundary = gpd.read_file(boundaryPath)

    datamask[datamask != -1] = datamask[datamask != -1] - 1

    # Reproject DSM to match the shape of the datamask
    src = rasterio.open(dwnldpath + '/DSM.tif')
    reproj_match(infile=wherepath + '/datamask_2023.tif', match=dwnldpath + '/DSM.tif', outfile=dwnldpath + '/DSM_reproj.tif') 

    src = rasterio.open(dwnldpath + '/DSM_reproj.tif')
    DSM, out_trans = mask(src, boundary.geometry, nodata=-1, crop=True)

    DSM = ToTensor()(DSM).float()
    DSM = DSM.squeeze(dim = 0)

    # If positive skewed distribution of the values:
    min_val  = DSM[DSM != -1 ].min().numpy()
    print("Min value of the elevation: ",min_val)
    # if there is negative values and zero values log transform must be applied after a shift to positive values only.
    # log(0) = -Inf
    if min_val > 0:
        DSM[DSM != -1] = np.log(DSM[DSM != -1])
    else:
        DSM[DSM != -1] = np.log(DSM[DSM != -1] + min_val + 1)
    #Normalise:
    DSM, DSMmean, DSMstd = rescale_image(DSM)
    print("Extracted mean: ",DSMmean)
    print("Devided std: ",DSMstd)

    # write to wherepath 
    torch.save(DSM, outpath + '/DSM.pt')