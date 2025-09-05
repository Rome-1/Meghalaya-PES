import requests
import json
import os

# https://eogdata.mines.edu/products/vnl/ -> Annual VNL V2
base_data_url = 'https://eogdata.mines.edu/nighttime_light/annual'
extensions = '.tif.gz'

def download_nightlight_data(years, savepath, metrics=["median", "maximum"]):
    # year: list with elements in [2012, 2023] or [12, 23]
    # extension one of: median, average, max, min, etc.
    # Requires .env variables env.VIIRS_USERNAME, env.VIIRS_PASSWORD
    # Downloads night data:  VIIRS Annual VNL V2
    # https://eogdata.mines.edu/products/register/index.html


    # Retrieve access token
    params = {    
        'client_id': 'eogdata_oidc',
        'client_secret': '2677ad81-521b-4869-8480-6d05b9e57d48', # same for all users
        'username': os.environ['VIIRS_USERNAME'],
        'password': os.environ['VIIRS_PASSWORD'],
        'grant_type': 'password'
    }
    token_url = 'https://eogauth.mines.edu/auth/realms/master/protocol/openid-connect/token'
    response = requests.post(token_url, data=params)
    
    if response.status_code != 200:
        raise Exception("Failed to retrieve authorization token to download VIIRS data:", response.error)
    access_token_dict = json.loads(response.text)
    access_token = access_token_dict.get('access_token')

    # Submit request with token bearer
    auth = 'Bearer ' + access_token
    headers = {'Authorization' : auth}

    for metric in metrics:
        for year in years:

            # Input validation
            if year in set(range(12,23+1)): # Convert year from 2000 to year from 0
                year += 2000

            if 'VIIRS_USERNAME' not in os.environ or 'VIIRS_PASSWORD' not in os.environ:
                raise Exception("VIIRS_USERNAME and/or VIIRS_PASSWORD missing from .env")
            if year < 2012 or year > 2023:
                raise Exception(f"VIIRS nighttime satellite data not available for {year}")
            
            # If data already exists, skip
            file_savename = f"viirs_{year}_{metric}{extensions}"
            file_unzipped_name = f"viirs_{year}_{metric}.tif"
            if os.path.exists(f"{savepath}/{file_savename}") or os.path.exists(f"{savepath}/{file_unzipped_name}"):
                print(f"VIIRS {year} already exists, skipping...")
                continue

            # Construct URL of data
            # https://eogdata.mines.edu/products/vnl/#:~:text=as%20shown%20below.-,File%20Naming,-The%20GeoTIFF%20files
            # Tile of India ID: 60E75N
            # https://eogdata.mines.edu/products/vnl/#:~:text=GTiff%20%3Cinput_file%3E%20%3Coutput_file%3E-,Tiles,-For%20those%20products
            if year <= 2021:
                create_date = "c202205302300"
                version = "v21"
                version_satellite = version + "_npp"
            elif year == 2022:
                create_date = "c202303062300"
                version = "v22"
                version_satellite = version + "_npp-j01"
            elif year == 2023:
                create_date = "v2_c202402081600"
                version = "v22"
                version_satellite = "npp"
            data_url = f"{base_data_url}/{version}/{year}/VNL_{version_satellite}_{year}_global_vcmslcfg_{create_date}.{metric}.dat{extensions}" 
            
            print(f"Downloading VIIRS {year}: {data_url}")
            response = requests.get(data_url, headers = headers)

            # Write response to output file
            with open(f"{savepath}/{file_savename}",'wb') as f:
                f.write(response.content)
            print(f"Saved VIIRS {year} nightlight data to: {savepath}/{file_savename}")

