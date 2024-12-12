import geopandas as gpd
import json
from shapely.geometry import Polygon
import os

# Requires .json file at FeatureCollectionPath
# which can be downloaded at 
# https://data.apps.fao.org/map/catalog/srv/eng/catalog.search#/metadata/f8065ff5-bae1-4d61-bbcd-654f036b2c4a
# Alternative to ee_boundary.py to download from Google Earth Engine
# which requires setup and API auth

region = 'Manipur'

def select_feature_by_name(geojson_path, name):
    with open(geojson_path) as f:
        geojson_data = json.load(f)
    
    # Filter features by NAME_1
    selected_features = [feature for feature in geojson_data['features'] if feature['properties']['NAME_1'] == name]
    
    # Create a new FeatureCollection with the selected feature
    selected_geojson = {
        "type": "FeatureCollection",
        "features": selected_features
    }
    
    return selected_geojson

def save_selected_feature_as_geojson(selected_geojson, output_path):
    with open(output_path, 'w') as f:
        json.dump(selected_geojson, f)

def save_geojson_as_shp(geojson_path, output_path):
    gdf = gpd.read_file(geojson_path)
    gdf.to_file(output_path, driver='ESRI Shapefile')


user = os.environ.get('USER')
FeatureCollectionPath = '/home/' + user + '/project/storage/DeepForestcast/data/geojsons/gadm41_IND_1.json'
outputGeoJsonPath = '/home/' + user + '/project/storage/DeepForestcast/data/geojsons/' + region + '.json'
outputShpPath = '/home/' + user + '/project/storage/DeepForestcast/data/' + region.lower() + '/boundary/boundary.shp'

if not os.path.exists(os.path.dirname(outputShpPath)):
    os.makedirs(os.path.dirname(outputShpPath))

selected_geojson = select_feature_by_name(FeatureCollectionPath, region)
save_selected_feature_as_geojson(selected_geojson, outputGeoJsonPath)
save_geojson_as_shp(outputGeoJsonPath, outputShpPath)