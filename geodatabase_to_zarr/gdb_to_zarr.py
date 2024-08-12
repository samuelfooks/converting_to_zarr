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
from pyproj import CRS

class ZarrConverter:
    def __init__(self, zipurl, geodatabase, resolution=0.01, variables=None):
        self.zipurl = zipurl
        self.geodatabase = geodatabase
        self.resolution = resolution
        self.variables = variables
        self.zip_file = os.path.basename(zipurl)
        self.temp_zarr_path = 'geodatabase_to_zarr/converted_zarr_files'
        os.makedirs(self.temp_zarr_path, exist_ok=True)
        self.gdb_path = None
        self.title = os.path.splitext(os.path.basename(geodatabase))[0]
        self.crs = None

# clean categorical data values
    def clean_categorical_data(self, data):
        if isinstance(data, str):
            if data in [' ', 'nan', "", " ", '']:
                data = 'None'
        return data

# map each value of categorical columns to a numerical value
    def encode_categorical_column(self, data):
        if isinstance(data[0], str):
            data = pd.Series(data).fillna('None')  # fill 'None'
            unique_categories = data.unique()
            category_mapping = {category: idx + 1 for idx, category in enumerate(unique_categories)}
            encoded_data = data.map(category_mapping).values
        else:
            encoded_data = data.astype(np.float32)
            category_mapping = {}
        return encoded_data, category_mapping


# add necessary metadata via the attributes of the zarr dataset
    def attributes_update(self, dataset):
        lon_min, lat_min, lon_max, lat_max = dataset.longitude.values.min(), dataset.latitude.values.min(), dataset.longitude.values.max(), dataset.latitude.values.max()
        latitudeattrs = {'_CoordinateAxisType': 'Lat',
                            'axis': 'Y',
                            'long_name': 'latitude',
                            'max': lat_max,
                            'min': lat_min,
                            'standard_name': 'latitude',
                            'step': (lat_max - lat_min) / dataset.latitude.values.shape[0],
                            'units': 'degrees_north'
            }
        longitudeattrs = {'_CoordinateAxisType': 'Lon',
                        'axis': 'X',
                        'long_name': 'longitude',
                        'max': lon_max,
                        'min': lon_min,
                        'standard_name': 'longitude',
                        'step': (lon_max - lon_min) / dataset.longitude.values.shape[0],
                        'units': 'degrees_east'
        }
        dataset.latitude.attrs.update(latitudeattrs)
        dataset.longitude.attrs.update(longitudeattrs)
        # Set the CRS as an attribute
        dataset.attrs['proj:epsg'] = 4326
        dataset.attrs['title'] = self.title
        dataset.attrs['geographical_extent'] = [lon_min, lat_min, lon_max, lat_max]
        dataset.attrs['Conventions'] = 'CF-1.8'
        # Include where the data comes and when its been converted
        dataset.attrs['History'] = f'Zarr dataset converted from {self.title}.gdb, downloaded from {self.zipurl}, on {date.today()}'
        return dataset


# function used to add _CRS attribute to each variable in the zarr dataset
    def update_crs(self, final_dataset):
        if isinstance(self.crs, dict):
            crs_wkt = CRS.from_dict(self.crs).to_wkt()
        else:
            crs_wkt = self.crs.to_wkt() 
        for var in final_dataset.data_vars:
            final_dataset[var].attrs['_CRS'] = {"wkt": crs_wkt}
        return final_dataset

# download the zip file and find the specified geodatabase, and the geodatabase layers
    def download_and_extract(self):
        class TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=self.zip_file) as t:
            urllib.request.urlretrieve(self.zipurl, filename=self.zip_file, reporthook=t.update_to)

        with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
            zip_ref.extractall('extracted_files')

        for root, dirs, files in os.walk('extracted_files'):
            for dir in dirs:
                if dir.endswith('.gdb') and os.path.basename(dir) == self.geodatabase:
                    self.gdb_path = os.path.join(root, dir)
                    self.gdblayers = fiona.listlayers(self.gdb_path)
                    break

# for a given geodatabase layer and a variable, convert that variable from that layer into a zarr dataset 
    def gdf2zarrconverter(self, file_path, native_var, layer):
        with fiona.open(file_path, 'r', layer=layer) as src:
            self.crs = src.crs
            self.total_bounds = src.bounds
            
            lon_min, lat_min, lon_max, lat_max = self.total_bounds
            width = int(np.ceil((lon_max - lon_min) / self.resolution))
            height = int(np.ceil((lat_max - lat_min) / self.resolution))
            raster_transform = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
            raster = np.zeros((height, width), dtype=np.float32)
            data = []
            geometries = []
            with tqdm(total=len(src), desc=f"Processing features of {layer} - {native_var}") as pbar:
                for feature in src:
                    value = self.clean_categorical_data(feature['properties'][native_var])
                    data.append(value)
                    geometries.append(feature['geometry'])
                    pbar.update()
            data = np.array(data)
            encoded_data, category_mapping = self.encode_categorical_column(data)
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
            dataset = xr.Dataset(coords={
                'latitude':  np.round(np.linspace(lat_max, lat_min, height, dtype=float), decimals=4),
                'longitude': np.round(np.linspace(lon_min, lon_max, width, dtype=float), decimals=4)
            })
            dataset[native_var] = (['latitude', 'longitude'], raster)
            dataset[native_var].attrs['standard_name'] = native_var
            dataset = dataset.sortby('latitude')

            if category_mapping:
                dataset[native_var].attrs['categorical_encoding'] = category_mapping

            dataset = self.attributes_update(dataset)
            zarr_var_path = f"{self.temp_zarr_path}/{self.title}_{layer}_{native_var}.zarr"
            dataset.to_zarr(zarr_var_path, mode='w', consolidated=True)
            
            return zarr_var_path

# open a geodatabase at the chosen layer, convert each variable or all variables into zarr datasets
# append each zarr path into a list, to be combined
    def process_gdblayer(self, gdblayer):
        with fiona.open(self.gdb_path, layer=gdblayer) as src:
            all_variables = src.schema['properties'].keys()
            variables_to_process = self.variables if self.variables else all_variables
            self.zarr_vars_paths = []
            for variable in variables_to_process:
                if variable in all_variables:
                    try:
                        print(f"Processing {gdblayer} - {variable}")
                        zarr_var_path = self.gdf2zarrconverter(self.gdb_path, variable, gdblayer)
                        self.zarr_vars_paths.append((zarr_var_path, variable))
                    except Exception as e:
                        print(f"Failed to process {gdblayer} - {variable}: {e}")
                        continue


# create an empty dataset, open each variable zarr dataset, chunk them by lat and lon, merge datasets
    def combine_and_rechunk(self):
        self.combined_dataset = xr.Dataset()
        with dask.config.set(scheduler='single-threaded'):
            for path, var in self.zarr_vars_paths:
                try:
                    dataset = xr.open_dataset(path, chunks={})
                    dataset = dataset.chunk({'latitude': 'auto', 'longitude': 'auto'}) 
                    self.combined_dataset = xr.merge([self.combined_dataset, dataset], compat='override', join='outer')
                                        
                except Exception as e:
                    print(f"Failed to combine zarr dataset {path}: {e}")
                    continue                  

# update the final combined dataset from each gdb layer with necessary metadata and attributes
    def update_and_finalize(self, gdblayer):
        categorical_encodings_dict = {}
        for var in self.combined_dataset.variables:
            if 'categorical_encoding' in self.combined_dataset[var].attrs:
                categorical_encodings_dict[var] = self.combined_dataset[var].attrs['categorical_encoding']

        self.combined_dataset.attrs['categorical_encoding'] = categorical_encodings_dict

        with dask.config.set(scheduler='single-threaded'):
            try:
                final_dataset = self.combined_dataset.chunk({'latitude': 'auto', 'longitude': 'auto'})
                final_dataset = self.update_crs(final_dataset)
                zarr_path = f"geodatabase_to_zarr/{gdblayer}_{self.title}.zarr"
                final_dataset = self.attributes_update(final_dataset)
                final_dataset.to_zarr(zarr_path, mode='w', consolidated=True)
            except Exception as e:
                print(f"Failed to save final zarr dataset: {e}")

# the run process from the ZarrConverter Object
    def run(self):
        self.download_and_extract()
        for layer in self.gdblayers:

            self.process_gdblayer(layer)
            self.combine_and_rechunk()
            self.update_and_finalize(layer)


# zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_geology/seabed_substrate/multiscale_folk_5/EMODnet_GEO_Seabed_Substrate_All_Res.zip'
# geodatabase = 'EMODnet_Seabed_Substrate_1M.gdb'
# Download the zip file
zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_human_activities/energy/wind_farms_points/EMODnet_HA_Energy_WindFarms_20240508.zip'
geodatabase = 'EMODnet_HA_Energy_WindFarms_20240508.gdb'

# resolution ~1 km
resolution = 0.01

# for specific columns/variables
variables = []

# establish a conversion object, and process each geodatabase layer into zarr datasets
converter = ZarrConverter(zipurl, geodatabase, resolution=resolution, variables=variables)
converter.run()