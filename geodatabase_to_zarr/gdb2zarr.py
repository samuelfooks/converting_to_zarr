# %% [markdown]
# # Creating a Rasterized ARCO Version of a Geodatabase
# 
# This Jupyter Notebook guides you through the process of converting a geodatabase into a rasterized ARCO version. The geodatabase contains multiple layers of geological seabed substrate data obtained from EMODnet Geology.
# 
# The conversion process is divided into several steps, each utilizing different Python packages:
# 
# 1. **Reading Geospatial Data**: We use the `fiona` package to read the geospatial data from the geodatabase.
# 
# 2. **Data Manipulation**: The `geopandas` package allows us to manipulate the vector geospatial data as needed.
# 
# 3. **Raster Operations**: We use the `rasterio` package to perform raster operations to rasterize the geodataframes.
# 
# 4. **Working with Multi-dimensional Arrays**: The `xarray` package enables us to work with multi-dimensional arrays, which is crucial for handling geospatial data.
# 
# 5. **Data Storage**: Finally, we use the `zarr` package to store the processed data in a compressed format.  And we store the data in s3 storage, allowing us to subset the data in the cloud
# 

# %%
import fiona
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
import zarr
import xarray as xr
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime, date
import urllib.request
import zipfile
import dask
import shutil

# %% [markdown]
# To update the attributes of a dataset, the `attributes_update()` function is used. This function takes in the dataset, title, resolution, and metadata dictionary as input. It updates the latitude and longitude attributes, sets the CRS, adds spatial extent information, includes the resolution, history, title, comment, and sources attributes. The function ensures that the dataset has accurate and informative attributes for better understanding and analysis of the data.
# 

# %%
def attributes_update(dataset, title, resolution, zipurl):
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
        dataset.attrs['resolution'] = resolution
        #include where the data comes and when its been converted
        dataset.attrs['History'] = f'Zarr dataset converted from {title}.gdb, downloaded from {zipurl}, on {date.today()}'
        
        #add any other attributes you think necessary to include in the metadata of your zarr dataset
        #dataset.attrs['sources'] = source
    

        return dataset

# %% [markdown]
# The `gdf2zarrconverter` function converts spatial data from a GeoDataFrame into a Zarr store.

### Parameters
# - `file_path`: Path to the input file.
# - `native_var`: Variable in the data to be processed.
# - `title`: Title for the output dataset.
# - `layer`: Layer of the geospatial data to be processed.
# - `arco_asset_tmp_path`: Temporary path for storing the output Zarr file.
# - `zipurl`: URL of the zip file containing the data.

# ### Returns
# - `zarr_var_path`: Path to the output Zarr file.

# ### Description
# This function converts a geospatial data file into a Zarr file. It cleans and encodes the data, rasterizes the geometries, creates an xarray dataset, and saves it as a Zarr file.
# The resolution is set at 0.01 degrees, but can be adjusted.  Be aware this will consume more memory and processing time.
# %%
def gdf2zarrconverter(file_path, native_var, title, layer, arco_asset_tmp_path, zipurl):

    def cleaner(data):
        if isinstance(data, str):
            if data == '0' or data == ' ' or data == np.nan or data == 'nan' or data == "" or data == " ":
                data = 'None'
        return data

    def encode_categorical(data):
        if isinstance(data[0], str):
            data = pd.Series(data)
            data = data.fillna('None')  # replace None values with 'None'
            
            data[data == ' '] = 'None'
            data[data == '0'] = 'None'
            data = data.values 
            unique_categories = np.unique(data)
            category_mapping = {'None': 1}
            counter = 2
            for category in unique_categories:
                if category != 'None':
                    category_mapping[category] = counter
                    counter += 1
            encoded_data = np.array([category_mapping.get(item, np.nan) for item in data])
        else:
            encoded_data = data.astype(np.float32)
            category_mapping = {}
        return encoded_data, category_mapping

    with fiona.open(file_path, 'r', layer=layer) as src:
        crs = src.crs
        total_bounds = src.bounds
        lon_min, lat_min, lon_max, lat_max = total_bounds
        resolution = 0.01
        width = int(np.ceil((lon_max - lon_min) / resolution))
        height = int(np.ceil((lat_max - lat_min) / resolution))
        raster_transform = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
        raster = np.zeros((height, width), dtype=np.float32)
        data = []
        geometries = []
        with tqdm(total=len(src), desc=f"Processing features of {layer} - {native_var}") as pbar:
            for feature in src:
                value = cleaner(feature['properties'][native_var])
                data.append(value)
                geometries.append(feature['geometry'])
                pbar.update()
        data = np.array(data)
        encoded_data, category_mapping = encode_categorical(data)
        with tqdm(total=len(geometries), desc="Rasterizing") as pbar:
            rasterio.features.rasterize(
                ((geom, value) for geom, value in zip(geometries, encoded_data)),
                out=raster,
                transform=raster_transform,
                merge_alg=rasterio.enums.MergeAlg.replace,
                dtype=np.float32,
            )
            pbar.update()
        
        # make xarray dataset, arrange latitude from max to min since rasterio makes rasters from top left to bottom right
        dataset = xr.Dataset(coords={'latitude':  np.round(np.linspace(lat_max, lat_min, height, dtype=float), decimals=4),
                                    'longitude': np.round(np.linspace(lon_min, lon_max, width, dtype=float), decimals=4)})
        dataset[native_var] = (['latitude', 'longitude'], raster)
        dataset = dataset.sortby('latitude')

        if category_mapping:
            # save the mappig dictionary with the variable attributes
            dataset[native_var].attrs['categorical_encoding']= category_mapping

        dataset = attributes_update(dataset, title, resolution, zipurl)
        zarr_var_path = f"{arco_asset_tmp_path}/{title}_{native_var}.zarr"
        dataset.to_zarr(zarr_var_path, mode='w', consolidated=True)
        return zarr_var_path


# %% [markdown]
# This Python script performs the following tasks:

# 1. **Download and Extract Geodatabase**: Downloads a zip file from a specified URL and extracts the geodatabase file.
# 2. **Geodatabase to Zarr Conversion**: Converts each layer and variable within the geodatabase into a Zarr dataset. This is done using a helper function `gdf2zarrconverter`.
# 3. **Rechunking with Dask**: The script uses Dask to rechunk the variables for efficient computation and compatibility of the Zarr datasets. This lays the groundwork for distributed computations if necessary.
# 4. **Combining Datasets**: The converted datasets are combined into a single Zarr dataset using `xr.merge`. This is done variable by variable to ensure compatibility.
# 6. **Saving the Final Dataset**: The combined dataset is rechunked again using Dask and saved as a Zarr file.


# %%

# Download the zip file
zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_geology/seabed_substrate/multiscale_folk_5/EMODnet_GEO_Seabed_Substrate_All_Res.zip'
geodatabase = 'EMODnet_Seabed_Substrate_1M.gdb'
zip_file = os.path.basename(zipurl)
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=zip_file) as t:
    urllib.request.urlretrieve(zipurl, filename=zip_file, reporthook=t.update_to)

# Extract the geodatabase from the zip file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('extracted_files')

for root, dirs, files in os.walk('extracted_files'):
    for dir in dirs:
        if dir.endswith('.gdb') and os.path.basename(dir) == geodatabase:
            gdb_path = os.path.join(root, dir)
            break

# %% [markdown]
#  4. Geodatabase to zarr
# 
#  Geodatabases can often contain multiple layers, and often contain a number of columns(variables).
# 
#  We simplify the burden of conversion by converting layers and one variable from each layer into a zarr dataset. Then we combine them into a single zarr dataset using dask to rechunk the variables and ensure both compatibility of the zarr datasets, and lay the ground work for distributed computations if necessary.

# %%
temp_zarr_path = 'converted_zarr_files'

os.makedirs(temp_zarr_path, exist_ok=True)
title = os.path.splitext(os.path.basename(geodatabase))[0]

# Get the layers from the geodatabase
layers = fiona.listlayers(gdb_path)

# Create an empty xarray dataset to hold the combined data
combined_dataset = xr.Dataset()

# Process each layer and each variable using gdf2zarr
for layer in layers:
    # Get the variables from the layer
    variables = fiona.open(gdb_path, layer=layer).meta['schema']['properties'].keys()
    
    zarr_vars_paths = [] # replace with your column names
    for variable in variables:
        try:
            print(f"Processing {layer} - {variable}")
            zarr_var_path = gdf2zarrconverter(gdb_path, variable, title, layer, temp_zarr_path, zipurl)
            zarr_vars_paths.append(zarr_var_path)
        except Exception as e:
            print(f"Failed to process {layer} - {variable}: {e}")
            continue

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

    with dask.config.set(scheduler='single-threaded'):
        try:    
            final_dataset = combined_dataset.chunk({'latitude': 'auto', 'longitude': 'auto'})  # for var in dataset.variables:
            zarr_path = f"{layer}.zarr"
            final_dataset.to_zarr(zarr_path, mode = 'w')
            shutil.rmtree(temp_zarr_path)
        except Exception as e:
            print(f"final zarr dataset did not save {layer}: {e}")
            continue

# Print the combined dataset
print(combined_dataset)


