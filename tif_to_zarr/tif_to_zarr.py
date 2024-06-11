import rioxarray as rxr
import rasterio
import xarray as xr
import dask
import dask.array as da
import pandas as pd
from datetime import datetime, date
import numpy as np
import os 
import logging
import requests
from zipfile import ZipFile
import os
import stat
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

arcologger = logging.getLogger(__name__)


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
      
        dataset.attrs['Comment'] = f"Downloaded from {zip_url} Converted from data product {title}.tif on {datetime.today()}"
        return dataset
def tif2zarr(file_path, band_variable, arco_asset_temp_dir, zip_url):
    

    with rasterio.open(file_path) as src:
        band_count = src.count
        band_descriptions = src.descriptions
    chunk_size = 200
    # Extract the title without the .tif extension
    title = os.path.splitext(os.path.basename(file_path))[0]
    # Open the file in chunks
    da = rxr.open_rasterio(file_path, chunks=chunk_size)
    # Assign a name to the DataArray
    da.name = band_variable
    # Convert the DataArray to a Dataset
    ds = da.to_dataset()
    # Rename the 'band' dimension to 'kdpar'
    ds = ds.rename({ 'x': 'longitude', 'y': 'latitude'})
    # Rechunk the data
    # Sort by latitude
    ds = ds.sortby('latitude')

    ds = ds.chunk({'latitude': chunk_size, 'longitude': chunk_size})
    ds[band_variable]= ds[band_variable].chunk({'latitude' : chunk_size, 'longitude': chunk_size})
    # Get the band number and variable names
    variable_names = list(ds.data_vars)
    resolution = abs(ds.latitude.values[0] - ds.latitude.values[1])
    # Add attributes to the dataset
    ds.attrs['band_count'] = band_count
    ds.attrs['band_descriptions'] = band_descriptions
    ds.attrs['variables'] = variable_names
    ds = attributes_update(ds, title, resolution, zip_url)
    zarr_path = f'{arco_asset_temp_dir}/{band_variable}.zarr'
    with dask.config.set(scheduler='threads'):  # use the threaded scheduler
        ds.to_zarr(zarr_path, mode='w')
    return zarr_path
def download_and_extract_zip(zip_url):
    zip_file_path = '/zipfiles/temp.zip'
    # Download the zip file
    response = requests.get(zip_url)
    with open(zip_file_path, "wb") as f:
        f.write(response.content)
    # Extract the contents of the zip file
    with ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall('/zipfiles')
    # Find the .shp file
    tif_file_path = None
    for root, dirs, files in os.walk('/zipfiles'):
        
        for file in files:
            if file.endswith(".tif"):
                tif_file_path = os.path.join(root, file)
                break
    return tif_file_path

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('Usage: python tif_to_zarr.py <zip_url> <band_variable>')
        sys.exit(1)
    zip_url = sys.argv[1]
    band_variable = sys.argv[2]
    arco_asset_temp_dir = os.environ.get('ARCO_ASSET_TEMP_DIR')

    # defaults for testing
    # zip_url = "https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_seabed_habitats/environmental_variables_that_influence_habitat_type_optical_properties/ratio_of_depth_to_seabed_secchi_disk_depth_in_baltic_sea/baltic_secchi_disk_depth_ratio.zip"
    # band_variable = 'secchi_disk_water_depth_ratio"
    # arco_asset_temp_dir = 'data'
    # zipdir = 'zipfiles'
    # os.makedirs(zipdir, exist_ok=True)

    tif_file_path = download_and_extract_zip(zip_url)
    print(tif_file_path)
    permissions = stat.filemode(os.stat(tif_file_path).st_mode)
    if tif_file_path:
        metadata_dict = {}  # Add your metadata here
        zarr_path = tif2zarr(tif_file_path, band_variable, arco_asset_temp_dir, zip_url)
        print(f'Zarr file saved at {zarr_path}')






