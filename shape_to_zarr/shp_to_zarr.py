import geopandas as gpd
import xarray as xr
import numpy as np
import rasterio.features
import rasterio.transform
import os
import fiona
import pandas as pd
import stat
import os
import sys
import dask
from datetime import datetime, date
import requests
from zipfile import ZipFile
import shutil
from shapely.geometry import box, shape
from tempfile import TemporaryDirectory
from tqdm import tqdm
# Suppress pandas SettingWithCopyWarning

pd.set_option('mode.chained_assignment', None)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def attributes_update(dataset, title, resolution, zip_url):
        latitudeattrs = {'_CoordinateAxisType': 'Lat', 
                            'axis': 'Y', 
                            'long_name': 'latitude', 
                            'max': dataset.latitude.values.max(), 
                            'min': dataset.latitude.values.min(), 
                            'standard_name': 'latitude', 
                            'step': (dataset.latitude.values.max() - dataset.latitude.values.min()) / dataset.latitude.values.shape[0], 
                            'units': 'degrees_north'
            }
        longitudeattrs = {'_CoordinateAxisType': 'Lon', 
                        'axis': 'X', 
                        'long_name': 'longitude',
                        'max': dataset.longitude.values.max(),
                        'min': dataset.longitude.values.min(),
                        'standard_name': 'longitude', 
                        'step': (dataset.longitude.values.max() - dataset.longitude.values.min()) / dataset.longitude.values.shape[0], 
                        'units': 'degrees_east'
        }
        dataset.latitude.attrs.update(latitudeattrs)
        dataset.longitude.attrs.update(longitudeattrs)

        # Set the CRS as an attribute
        dataset.attrs['proj:epsg'] = 4326
        dataset.attrs['resolution'] = resolution
        dataset.attrs.update({
            'geospatial_lat_min': dataset['latitude'].min().item(),
            'geospatial_lat_max': dataset['latitude'].max().item(),
            'geospatial_lon_min': dataset['longitude'].min().item(),
            'geospatial_lon_max': dataset['longitude'].max().item()
        })
        dataset.attrs['history'] = f'Converted on {date.today()}'
        dataset.attrs['title'] = title
      
        dataset.attrs['Comment'] = f"Downloaded from {zip_url} Converted from data product {title}.shp on {datetime.today()}"
        return dataset
# Function to download and extract the zip file
def download_and_extract_zip(zip_url):
    zip_file_path = 'zipfiles/temp.zip'
    # Download the zip file
    response = requests.get(zip_url)
    with open(zip_file_path, "wb") as f:
        f.write(response.content)
    # Extract the contents of the zip file
    with ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall('zipfiles')
    # Find the .shp file
    shp_file_path = None
    for file in os.listdir('zipfiles'):
        if file.endswith(".shp"):
            shp_file_path = os.path.join('zipfiles', file)
            break
    return shp_file_path
    
    
def gdf2zarrconverter(shp_file_path, variable, resolution, arco_asset_tmp_path, zip_url):
    def cleaner(data):
        if isinstance(data, str):
            if data == '0' or data =='' or data == np.nan or data == 'nan' or data == "" or data == " ":
                data = 'None'
        return data

    def encode_categorical(data):
        if isinstance(data[0], str):
            data[data ==''] = 'None'
            data[data == '0'] = 'None'
            unique_categories = np.unique(data)
            category_mapping = {'None': 1}
            counter = 2
            for category in unique_categories:
                if category!= 'None':
                    category_mapping[category] = counter
                    counter += 1
            encoded_data = np.array([category_mapping.get(item, np.nan) for item in data])
        else:
            encoded_data = data.astype(np.float32)
            category_mapping = {}
        return encoded_data, category_mapping

    resolution = float(resolution)
    title = os.path.splitext(os.path.basename(shp_file_path))[0]
    with fiona.open(shp_file_path, 'r') as src:
        crs = src.crs
        total_bounds = src.bounds
        lon_min, lat_min, lon_max, lat_max = total_bounds
        width = int(np.ceil((lon_max - lon_min) / resolution))
        height = int(np.ceil((lat_max - lat_min) / resolution))
        raster_transform = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
        raster = np.zeros((height, width), dtype=np.float32)
        data = []
        geometries = []
        with tqdm(total=len(src), desc=f"Processing features of {variable}") as pbar:
            for feature in src:
                value = cleaner(feature['properties'][variable])
                data.append(value)
                geometries.append(feature['geometry'])
                pbar.update()
        data = np.array(data)
        encoded_data, category_mapping = encode_categorical(data)

    # Create a grid for each GDF
    grid = np.zeros((height, width), dtype=np.float32)

    # Iterate over each polygon and overlay it on the grid
    with tqdm(total=len(geometries), desc=f"Rasterizing geometries") as pbar:
        for geom, value in zip(geometries, encoded_data):
            geom = shape(geom)
            # Rasterize the polygon on the grid
            rasterio.features.rasterize(
                [(geom, value)],
                out=grid,
                transform=raster_transform,
                merge_alg=rasterio.enums.MergeAlg.replace,
                dtype=np.float32,
            )
            pbar.update()
    chunk_size = 200
    # Create an xarray dataset from the grid
    dataset = xr.Dataset(coords={'latitude':  np.round(np.linspace(lat_max, lat_min, height, dtype=float), decimals=4),
                                 'longitude': np.round(np.linspace(lon_min, lon_max, width, dtype=float), decimals=4)})
    dataset[variable] = (['latitude', 'longitude'], grid)
    
    # first chunk for the heavy sorting operation
    dataset = dataset.chunk({'latitude': chunk_size, 'longitude': chunk_size})
    dataset = dataset.sortby('latitude')

    # rechunk to corrects even sorted latitude chunks
    dataset = dataset.chunk({'latitude': chunk_size, 'longitude': chunk_size})
    
    if category_mapping:
        # save the mappig dictionary with the variable attributes
        dataset[variable].attrs['categorical_encoding']= category_mapping

    dataset = attributes_update(dataset, title, resolution, zip_url)

    zarr_var_path = f"{arco_asset_tmp_path}/{title}_{variable}.zarr"
    dataset.to_zarr(zarr_var_path, mode='w', consolidated=True)
    return zarr_var_path



if __name__ == "__main__":

    # if len(sys.argv) != 3:
    #     print(f"Usage: python shp_to_zarr.py <zip_url> <resolution>")
    #     sys.exit(1)
    # zip_url = sys.argv[1]
    # arco_asset_temp_dir = os.environ.get('ARCO_ASSET_TEMP_DIR')
    # resolution = sys.argv[2]
    
    # defaults for testing
    zip_url = "https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/archive/human_activities_windfarms/EMODnet_HA_Energy_WindFarms_polygons_20231124.zip"
    arco_asset_temp_dir = 'data'
    resolution = "0.01" 
    
    # Download and extract the zip file, then get the path to the .shp file
    shp_file_path = download_and_extract_zip(zip_url)
    
    print(shp_file_path)
    permissions = stat.filemode(os.stat(shp_file_path).st_mode)
    print("File Permissions:", permissions)
    # Convert the .shp file to zarr using gdf2zarrconverter function
    if shp_file_path:
        
        title = os.path.splitext(os.path.basename(shp_file_path))[0]
        combined_dataset = xr.Dataset()
        
        variables = fiona.open(shp_file_path).meta['schema']['properties'].keys()
    
        zarr_vars_paths = []
        # Process each variable in the .shp file or choose your own
        for variable in variables:
            try:
                print(f"Processing {variable}")
                zarr_var_path = gdf2zarrconverter(shp_file_path, variable, resolution, arco_asset_temp_dir, zip_url )
                zarr_vars_paths.append(zarr_var_path)
            except Exception as e:
                print(f"Failed to process {variable}: {e}")
                continue
        
        # join all the zarr datasets into a single dataset
        with dask.config.set(scheduler='single-threaded'):
            for path in zarr_vars_paths:
                try:
                    dataset = xr.open_dataset(path, chunks={})  # Use Dask to lazily load the dataset
                    dataset = dataset.chunk({'latitude': 'auto', 'longitude': 'auto'}) 
                    combined_dataset = xr.merge([combined_dataset, dataset], compat='override', join='outer')
                except Exception as e:
                    print(f"Failed to combine zarr dataset {path}: {e}")
                    continue

        # add applicable categorical encodings
        categorical_encodings_dict = {}
        for var in combined_dataset.variables:
            if 'categorical_encoding' in combined_dataset[var].attrs:
                categorical_encodings_dict[var] = combined_dataset[var].attrs['categorical_encoding']

        combined_dataset.attrs['categorical_encoding'] = categorical_encodings_dict

        # rechunk and save the final dataset
        with dask.config.set(scheduler='single-threaded'):
            try:    
                final_dataset = combined_dataset.chunk({'latitude': 'auto', 'longitude': 'auto'})  # for var in dataset.variables:
                combined_zarr = f"{title}_res{resolution}.zarr"
                final_dataset.to_zarr(f"{arco_asset_temp_dir}/{combined_zarr}", mode = 'w')

                # Cleanup: delete all zarr files except the final one
                for file in os.listdir(arco_asset_temp_dir):
                    if file.endswith(".zarr") and file != combined_zarr:
                        shutil.rmtree(os.path.join(arco_asset_temp_dir, file))
                            
            except Exception as e:
                print(f"final zarr dataset did not save {title}: {e}")

    # Print the combined dataset
    print(combined_dataset)

