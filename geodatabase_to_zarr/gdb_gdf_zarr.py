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
from shapely.geometry import box, shape
from rtree import index as rtree_index
import shapely.ops
import shapely
from rasterio.windows import Window
import dask.array as da
import dask.delayed
from affine import Affine
import logging
import sys
from dask.distributed import LocalCluster, Client
import psutil
import time
import threading
import logging
from memory_profiler import profile
import gc
def make_logger(name, log_file_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent logging from propagating to the root logger

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def log_memory_usage(logger, interval=1):
    """
    Logs memory usage every `interval` seconds.
    Runs in a separate thread.
    """
    process = psutil.Process()
    while True:
        mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        logger.info(f"Current Memory Usage: {mem:.2f} MB")
        time.sleep(interval)

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
logger = make_logger('converter', f'{log_dir}/converter.log')



class ZarrConverter:
    def __init__(self, zipurl, geodatabase, layer_index, resolution=0.01, variables=None):
        self.zipurl = zipurl
        self.zip_file = os.path.basename(zipurl)
        self.geodatabase = geodatabase
        self.layer_index = layer_index
        self.resolution = resolution
        self.variables = variables
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = f"{self.script_dir}/../data"
        self.temp_zarr_path = f"{self.data_dir}/converted_zarr_files"
        self.extract_dir = f"{self.data_dir}/extracted_files"
        for dir in [self.data_dir, self.temp_zarr_path, self.extract_dir]:
            os.makedirs(dir, exist_ok=True)
        # clean up the directories
        
        for dir in [self.temp_zarr_path, self.extract_dir]:
            self.clean_dir(dir)
        self.zarr_vars_paths = []
        self.gdb_path = None
        self.title = os.path.splitext(os.path.basename(geodatabase))[0]
        self.crs = None

    def clean_dir(self, targetdir):
        
        for root, dirs, files in os.walk(targetdir, topdown=False):
            # Remove all files
            for file in files:
                os.remove(os.path.join(root, file))
            
            # Remove all directories
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):  # Check if the directory is empty
                    os.rmdir(dir_path)
            
        print(f'Cleaned output directory: {targetdir}')

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
            urllib.request.urlretrieve(self.zipurl, filename=f"{self.data_dir}/{self.zip_file}", reporthook=t.update_to)

        with zipfile.ZipFile(f"{self.data_dir}/{self.zip_file}", 'r') as zip_ref:
            zip_ref.extractall(self.extract_dir)

        for root, dirs, files in os.walk(self.extract_dir):
            for dir in dirs:
                if dir.endswith('.gdb') and os.path.basename(dir) == self.geodatabase:
                    self.gdb_path = os.path.join(root, dir)
                    self.gdblayers = fiona.listlayers(self.gdb_path)
                    break

    def rasterize_chunk(self, chunk, raster_shape, transform):
            raster_chunk = np.zeros(raster_shape, dtype=np.float32)

            valid_geoms = (geom for geom in chunk['geometries'] if geom.is_valid)
            if not valid_geoms:
                return raster_chunk

            shapes = ((geom, value) for geom, value in zip(valid_geoms, chunk['encoded_data']))
            rasterio.features.rasterize(
                shapes,
                out=raster_chunk,
                transform=transform,
                merge_alg=rasterio.enums.MergeAlg.replace,
                dtype=np.float32,
            )

            return raster_chunk

    def cleaner(self, data):
        if isinstance(data, str):
            if data == '0' or data.strip() == '' or pd.isna(data):
                return 'None'
        return data

    def encode_categorical(self, data, category_mapping):
        if data.dtype == 'object':
            data = pd.Series(data).fillna('None').replace({' ': 'None', '': 'None', '0': 'None'})
            unique_categories = np.unique(data)
            for category in unique_categories:
                if category not in category_mapping:
                    category_mapping[category] = len(category_mapping) + 1

            encoded_data = data.map(category_mapping).values
        else:
            data = data.astype(np.float32)
            data[np.isnan(data)] = 0
            encoded_data = data
        return encoded_data, category_mapping

    @profile
    def single_vector_variable_to_zarr(self, vector_file, layer, native_var, temp_zarr_path, resolution=0.01):
        """
        Converts a specific variable within a vector layer to Zarr format.
        
        Parameters:
            native_var: The variable within the vector data to convert.
            layer: The specific layer within the vector data that contains the variable. Default is None.
            
        Returns:
            The path to the converted Zarr file.
        """
        arco_dir = temp_zarr_path
        title = os.path.splitext(os.path.basename(vector_file))[0]
        if layer is None:
            layer = 0

        resolution = 0.01

        with fiona.open(vector_file, 'r', layer=layer) as src:
            self.crs = src.crs
            self.total_bounds = src.bounds
            lon_min, lat_min, lon_max, lat_max = self.total_bounds
            width = int(np.ceil((lon_max - lon_min) / resolution))
            height = int(np.ceil((lat_max - lat_min) / resolution))
            raster_transform = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
            chunk_width = 1000
            chunk_height = 1000
            category_mapping = {}
            row_num= 0
            for i in range(0, height, chunk_height):
                remainder_y = height - i
                window_height = min(chunk_height, remainder_y)
                row_rasters = []
                for j in range(0, width, chunk_width):
                    print(f"Processing window at i={i}, from height {height}, j={j}, from width {width}")
                    remainder_x = width - j
                    window_width = min(chunk_width, remainder_x)
                    window = Window(j, i, window_width, window_height)
                    chunk_geoms = []
                    chunk_data = []
                    window_geom = box(
                        *rasterio.windows.bounds(window, transform=raster_transform)
                    )
                    with tqdm(total=len(src), desc=f"Processing features of {layer} - {native_var}") as pbar:
                        for feature in src:
                            geom = feature['geometry']
                            if isinstance(geom, fiona.model.Geometry):
                                geom = shapely.geometry.shape(geom)
                            if geom.intersects(window_geom):
                                chunk_geoms.append(geom)
                                chunk_data.append(feature['properties'][native_var])
                            pbar.update(1)

                    if not chunk_geoms:
                        empty_raster = np.zeros((window_height, window_width))
                        lazy_raster = da.from_array(empty_raster, chunks=(chunk_height, chunk_width))
                        
                    else:
                        chunk_data = pd.Series(chunk_data)
                        dtype = chunk_data.dtype
                        if dtype == 'object':
                            for k in range(len(chunk_data)):
                                if isinstance(chunk_data[k], str):
                                    chunk_data[k] = self.cleaner(chunk_data[k])
                        
                        chunk_data = np.array(chunk_data)
                        encoded_data, category_mapping = self.encode_categorical(chunk_data, category_mapping) 

                        # Calculate the top-left corner of the chunk
                        chunk_x = raster_transform.c + window.col_off * raster_transform.a
                        chunk_y = raster_transform.f + window.row_off * raster_transform.e

                        # Create a new transform for the chunk
                        chunk_transform = Affine(raster_transform.a, raster_transform.b, chunk_x,
                                                raster_transform.d, raster_transform.e, chunk_y)

                        # Use the chunk_transform in the delayed function
                        lazy_raster = dask.delayed(self.rasterize_chunk)(
                            {'geometries': chunk_geoms, 'encoded_data': encoded_data},
                            (window_height, window_width),
                            chunk_transform
                        )
                        lazy_raster = da.from_delayed(lazy_raster, shape=(window_height, window_width), dtype=np.float32)
                                            
                    row_rasters.append(lazy_raster)

                # Instead of appending to row_rasters, create a DataArray and add it to the Dataset
                if row_rasters:
                    row_raster = da.concatenate(row_rasters, axis=1)
                    row_raster_computed = row_raster.compute()  # Compute the raster to check values
                    print(np.max(row_raster_computed))
                    data_array = xr.DataArray(
                        row_raster,
                        dims=['latitude', 'longitude'],
                        name=native_var
                    ).chunk({'latitude': 500, 'longitude': 500})

                    data_array.to_zarr(f'{arco_dir}/rows/row_{row_num}.zarr', mode='w')
                    row_num += 1
                    logger.info(f"Finished processing window at i={i}")
                    
                    del row_raster
                    del row_raster_computed
                    del data_array
                    gc.collect()

            # Load the zarr files and concatenate them
            datasets = [xr.open_zarr(f'{arco_dir}/rows/row_{i}.zarr') for i in range(row_num)]
            dataset = xr.concat(datasets, dim='latitude')

            latitudes = np.round(np.linspace(lat_max, lat_min, height, dtype=float), decimals=4)
            longitudes = np.round(np.linspace(lon_min, lon_max, width, dtype=float), decimals=4)

            dataset = dataset.assign_coords({'latitude': latitudes, 'longitude': longitudes})
            dataset = dataset.sortby('latitude')
            dataset = dataset.chunk({'latitude': 'auto', 'longitude': 'auto'})
            
            if category_mapping:
                dataset[native_var].attrs['categorical_encoding'] = category_mapping
            
            zarr_var_path = f"{arco_dir}/{title}_{native_var}.zarr"
            dataset.to_zarr(zarr_var_path, mode='w')

            del dataset
            del datasets
            gc.collect()
            return zarr_var_path
    
    # for a given geodatabase layer and a variable, convert that variable from that layer into a zarr dataset 
    def vector_layer_to_zarr(self, layer):
        vector_file = self.gdb_path
        arco_dir = self.temp_zarr_path
        title = os.path.splitext(os.path.basename(vector_file))[0]
        self.zarr_vars_paths = []
        if layer is None:
            layer = 0

        resolution = 0.01

        if self.variables is None:
            with fiona.open(vector_file, 'r', layer=layer) as src:
                self.variables = list(src.schema['properties'].keys())

        self.zarr_vars_paths = []
        # Initialize category mappings for each variable
        self.category_mappings = {var: {} for var in self.variables}

        
        for native_var in self.variables:
            logger.info(f"Processing variable: {native_var}")
            zarr_path = self.single_vector_variable_to_zarr(vector_file, layer, native_var, self.temp_zarr_path, resolution)
            self.zarr_vars_paths.append(zarr_path)

        logger.info(f"Finished processing layer: {layer}")


# combine the zarr datasets from each variable into a single dataset
    def combine_and_rechunk(self):
        logger.info('combining final zarr dataset')
        combined_dataset = xr.Dataset()
        try:
            for path in self.zarr_vars_paths:
                dataset = xr.open_dataset(path, chunks={}, engine='zarr')
                combined_dataset = xr.merge([combined_dataset, dataset], compat='override', join='outer')
                
            logger.info(f"chunking lat and lon for {path}")
            combined_dataset = combined_dataset.chunk({'latitude': 'auto', 'longitude': 'auto'})
            self.combined_dataset = combined_dataset
        except Exception as e:
            logger.error(f"Error combining zarr files: {e}")
            return None

# update the final combined dataset from each gdb layer with necessary metadata and attributes
    def update_and_finalize(self, gdblayer):
        categorical_encodings_dict = {}
        for var in self.combined_dataset.variables:
            if 'categorical_encoding' in self.combined_dataset[var].attrs:
                categorical_encodings_dict[var] = self.combined_dataset[var].attrs['categorical_encoding']

        self.combined_dataset.attrs['categorical_encoding'] = categorical_encodings_dict

        try:
            final_dataset = self.combined_dataset.chunk({'latitude': 'auto', 'longitude': 'auto'})
            final_dataset = self.update_crs(final_dataset)
            zarr_path = f"{self.temp_zarr_path}/geodatabase_to_zarr_finalzarr/{gdblayer}_{self.title}.zarr"
            final_dataset = self.attributes_update(final_dataset)
            final_dataset.to_zarr(zarr_path, mode='w', consolidated=True)
        except Exception as e:
            print(f"Failed to save final zarr dataset: {e}")


# the run process from the ZarrConverter Object
    def run(self):
        self.download_and_extract()
        #Start memory logging in a separate thread
        # Start memory logging in a separate thread with the 'converter' logger
        memory_thread = threading.Thread(target=log_memory_usage, args=(logger,), daemon=True)
        memory_thread.start()
        for layer in self.gdblayers:
            layer = self.gdblayers[self.layer_index]
            self.vector_layer_to_zarr(layer)
            self.combine_and_rechunk()
            self.update_and_finalize(layer)
            break

def main():

    # Small test dataset
    zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_human_activities/energy/wind_farms_points/EMODnet_HA_Energy_WindFarms_20240508.zip'
    geodatabase = 'EMODnet_HA_Energy_WindFarms_20240508.gdb'
    layer_index = 1
    variables = ['COUNTRY', 'POWER_MW', 'STATUS', 'YEAR']

    # larger dataset
    # zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_geology/seabed_substrate/multiscale_folk_5/EMODnet_GEO_Seabed_Substrate_All_Res.zip'
    # geodatabase = 'EMODnet_Seabed_Substrate_1M.gdb'
    # layer_index = 0
    #variables = ['Folk_5cl', 'Folk_5cl_txt']

    # zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_seabed_habitats/12550/EUSeaMap_2023_MediterraneanSea.zip'
    # geodatabase = 'EUSeaMap_2023_MediterraneanSea.gdb'
    # layer_index = 0
    # variables = ['Biozone', 'ModelCode']

    ## so large it crashes
    # zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_seabed_habitats/12549/EUSeaMap_2023.zip'
    # geodatabase = 'EUSeaMap_2023.gdb'
    # layer_index = 0
    # variables = ['EUNIS2019C', 'Energy', 'Substrate', 'Biozone']

    # resolution ~1 km
    resolution = 0.01


    # establish a conversion object, and process each geodatabase layer into zarr datasets
    converter = ZarrConverter(zipurl, geodatabase, layer_index, resolution=resolution, variables=variables)
    converter.run()

if __name__ == '__main__':
    n_workers = 3
    
    with LocalCluster(n_workers=n_workers, threads_per_worker=2, memory_limit='6GiB') as cluster:
        client = Client(cluster)
        print(f"LocalCluster started with {n_workers} workers")
        main()
        client.close()