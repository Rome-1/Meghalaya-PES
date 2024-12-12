import os
from data_loading.prepare_nightlight_data import prepare_nightlight_data
from data_loading.load_nightlight_data import download_nightlight_data
from data_loading.Download_functions import download_data, import_boundary
from data_loading.prepare_data import prepare_data
from data_loading.to_tensors import to_tensors
# from data_loading.ee_boundary import download_boundary
from data_loading.dsm import DSM_to_tensor

USER = os.environ.get('USER')
dir = os.environ.get('DeepForestcast_PATH')
old_region = os.environ.get('REGION')
region = os.environ.get('REGION')

# Import global grid
gridPath = dir + "/data/gfc_tiles/gfc_tiles.shp"

years = list(range(14, 23+1))

types_static = ["treecover2000", "datamask", "gain"]
types_dynamic = ["lossyear", "last"]


dwnldPath = dir + "/data/" + region + "/"
if not os.path.exists(dwnldPath):
    os.makedirs(dwnldPath)

outPath = dir + "/outputs/" + region + "/"
if not os.path.exists(outPath):
    os.makedirs(outPath)

nightlight_configured = 'VIIRS_USERNAME' in os.environ and 'VIIRS_PASSWORD' in os.environ

if nightlight_configured:
    nightlight_metric = ["median", "maximum"]
    nightlightDwnldPath = dwnldPath + "nightlight/"
    if not os.path.exists(nightlightDwnldPath):
        os.makedirs(nightlightDwnldPath)
    nightlightOutPath = outPath + "nightlight/"
    if not os.path.exists(nightlightOutPath):
        os.makedirs(nightlightOutPath)


# Import boundary data (requires Google Earth Engine Config)
download_boundary(dir, region_upper) # cannot be run with sbatch as it requires user input

boundPathTight = dir + "/data/" + old_region + "/boundary/boundary.shp"
buffsize = 0.09 # this is the original buffer set by Ball et al.
boundaryPath = dir + "/data/" + region + "/boundary/boundary.shp"
if not os.path.exists(dir + "/data/" + region + "/boundary"):
    os.makedirs(dir + "/data/" + region + "/boundary")
buffer = import_boundary(boundPathTight, buffer=buffsize, filename=boundaryPath)

download_data(buffer, dwnldPath, years, types_static, types_dynamic, gridPath)

if nightlight_configured:
    download_nightlight_data(years, nightlightDwnldPath, metrics = nightlight_metric)

prepare_data(types_static, types_dynamic, dwnldPath, outPath, boundPathTight, boundaryPath=boundaryPath, buffer=buffer, buffsize=buffsize, gridPath=gridPath)
prepare_nightlight_data(years, nightlightDwnldPath, nightlightOutPath, outPath + "buffer.tif", boundaryPath, metric=nightlight_metric) # will overwrite if data already exists

to_tensors(years=years,
    region=region,
    sourcepath=outPath,
    wherepath=outPath + "/tensors",
    nightlight_metrics=nightlight_metric if nightlight_configured else None # TODO median will be parameter
)

# DSM_to_tensor(dwnldpath=dwnldPath, file_name="Meghalaya_DEM", wherepath=outPath, outpath=outPath + "/tensors", boundaryPath=boundaryPath)
