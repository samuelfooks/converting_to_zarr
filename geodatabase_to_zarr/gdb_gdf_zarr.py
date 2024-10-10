# uses spatial indexing to save geometries.  Has worked for smaller datasets, now testing on larger datasets

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
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = make_logger('converter', 'converter.log')

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
        self.temp_zarr_path = f"{self.data_dir}/converted_zarr_files_spat"
        self.extract_dir = f"{self.data_dir}/extracted_files_spat"
        for dir in [self.data_dir, self.temp_zarr_path, self.extract_dir]:
            os.makedirs(dir, exist_ok=True)
        # clean up the directories
        
        for dir in [self.temp_zarr_path, self.extract_dir]:
            self.clean_dir(dir)
        self.zarr_vars_paths = []
        self.gdb_path = None
        self.title = os.path.splitext(os.path.basename(geodatabase))[0]
        self.crs = None
        # Initialize the logger
        self.setup_logger()

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
    def process_gdb_layer(self, layer):
        vector_file = self.gdb_path
        arco_dir = self.temp_zarr_path
        title = os.path.splitext(os.path.basename(vector_file))[0]
        self.zarr_vars_paths = []
        if layer is None:
            layer = 0

        resolution = 0.01

        def rasterize_chunk(chunk, raster_shape, transform):
            raster_chunk = np.zeros(raster_shape, dtype=np.float32)

            valid_geoms = [geom for geom in chunk['geometries'] if geom.is_valid]
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

        def cleaner(data):
            if isinstance(data, str):
                if data == '0' or data.strip() == '' or pd.isna(data):
                    return 'None'
            return data

        def encode_categorical(data, category_mapping):
            if data.dtype == 'object':
                data = pd.Series(data).fillna('None')
                data.replace({' ': 'None', '': 'None', '0': 'None'}, inplace=True)
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

        if self.variables is None:
            with fiona.open(vector_file, 'r', layer=layer) as src:
                self.variables = list(src.schema['properties'].keys())

        # Initialize category mappings for each variable
        self.category_mappings = {var: {} for var in self.variables}

        # Build a spatial index for all geometries in the layer
        
        with fiona.open(vector_file, 'r', layer=layer) as src:
            logging.info(f"Building spatial index for layer: {layer} for {len(src)} features")
            spatial_index = rtree_index.Index()
            geometries = []
            with tqdm(total =len(src), desc='features') as pbar:
                for fid, feature in enumerate(src):
                    geom = shapely.geometry.shape(feature['geometry'])
                    geometries.append(geom)
                    spatial_index.insert(fid, geom.bounds)
                    pbar.update(1)


        for native_var in self.variables:
            logger.info(f"Processing variable: {native_var}")
            with fiona.open(vector_file, 'r', layer=layer) as src:
                self.crs = src.crs
                self.total_bounds = src.bounds
                lon_min, lat_min, lon_max, lat_max = self.total_bounds
                width = int(np.ceil((lon_max - lon_min) / resolution))
                height = int(np.ceil((lat_max - lat_min) / resolution))
                raster_transform = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

                # Define chunk sizes
                chunk_width = 1000
                chunk_height = 1000

                # Initialize Zarr store for the current variable
                zarr_var_path = f"{arco_dir}/{title}_{native_var}.zarr"
                logger.info(f"Initializing Zarr store at {zarr_var_path}")

                # Predefine the shape and chunks for the DataArray
                shape = (height, width)
                chunks = (chunk_height, chunk_width)

                # Initialize an empty DataArray with Dask chunks
                data_array = xr.DataArray(
                    da.zeros(shape, chunks=chunks, dtype=np.float32),
                    dims=['latitude', 'longitude'],
                    coords={
                        'latitude': np.linspace(lat_max, lat_min, height, dtype=float),
                        'longitude': np.linspace(lon_min, lon_max, width, dtype=float)
                    },
                    name=native_var
                )

                # Create the Dataset with the DataArray
                dataset = xr.Dataset({native_var: data_array})

                # Initialize the Zarr store
                dataset.to_zarr(zarr_var_path, mode='w', compute=False)

                # Iterate over chunks and write data incrementally
                for i in range(0, height, chunk_height):
                    window_height = min(chunk_height, height - i)
                    for j in range(0, width, chunk_width):
                        window_width = min(chunk_width, width - j)
                        logger.info(f"Processing window: rows {i}-{i+window_height}, cols {j}-{j+window_width}")

                        window = Window(j, i, window_width, window_height)
                        window_geom = box(
                            *rasterio.windows.bounds(window, transform=raster_transform)
                        )
                        chunk_geoms = []
                        chunk_data = []
                        # Find intersecting geometries using spatial index
                        intersecting_ids = list(spatial_index.intersection(window_geom.bounds))
                        chunk_geoms = [geometries[fid] for fid in intersecting_ids if geometries[fid].intersects(window_geom)]

                        if not chunk_geoms:
                            raster_chunk = np.zeros((window_height, window_width), dtype=np.float32)
                        else:
                            # Extract properties for the current variable
                            for fid in intersecting_ids:
                                feature = fiona.open(vector_file, 'r', layer=layer)[fid]
                                if feature is None:
                                    logger.info(f"Feature with fid {fid} is None")
                                    continue
                                if 'properties' not in feature:
                                    logger.info(f"Feature with fid {fid} has no 'properties'")
                                    continue
                                if native_var not in feature['properties']:
                                    logger.info(f"native_var '{native_var}' not found in feature {fid}")
                                    continue
                                chunk_data.append(feature['properties'][native_var])

                            if isinstance(chunk_data[0], str):
                                chunk_data = [cleaner(d) for d in chunk_data]

                            chunk_data = pd.Series(chunk_data).values
                            encoded_data, self.category_mappings[native_var] = encode_categorical(
                                chunk_data, self.category_mappings[native_var]
                            )

                            # Calculate the transform for the current window
                            chunk_x = raster_transform.c + window.col_off * raster_transform.a
                            chunk_y = raster_transform.f + window.row_off * raster_transform.e
                            chunk_transform = Affine(
                                raster_transform.a, raster_transform.b, chunk_x,
                                raster_transform.d, raster_transform.e, chunk_y
                            )

                            # Rasterize the chunk using Dask delayed
                            lazy_raster = dask.delayed(rasterize_chunk)(
                                {'geometries': chunk_geoms, 'encoded_data': encoded_data},
                                (window_height, window_width),
                                chunk_transform
                            )
                            # Compute the raster chunk
                            raster_chunk = da.from_delayed(
                                lazy_raster, shape=(window_height, window_width), dtype=np.float32
                            ).compute()

                        # Assign the raster_chunk to the appropriate window in the DataArray
                        dataset[native_var].data[i:i+window_height, j:j+window_width] = raster_chunk

                        logger.info(f"Finished processing window: rows {i}-{i+window_height}, cols {j}-{window_width}")

                # After processing all chunks, assign category mappings as attributes
                if self.category_mappings.get(native_var):
                    dataset[native_var].attrs['categorical_encoding'] = self.category_mappings[native_var]

                # Write the completed DataArray to Zarr
                dataset.to_zarr(zarr_var_path, mode='a', compute=True)

                self.zarr_vars_paths.append(zarr_var_path)
                logger.info(f"Finished processing variable: {native_var}")

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
            zarr_path = f"{self.temp_zarr_path}/geodatabase_to_zarr/{gdblayer}_{self.title}.zarr"
            final_dataset = self.attributes_update(final_dataset)
            final_dataset.to_zarr(zarr_path, mode='w', consolidated=True)
        except Exception as e:
            print(f"Failed to save final zarr dataset: {e}")

# the run process from the ZarrConverter Object
    def run(self):
        self.download_and_extract()
        for layer in self.gdblayers:
            layer = self.gdblayers[self.layer_index]

            self.process_gdb_layer(layer)
            self.combine_and_rechunk()
            self.update_and_finalize(layer)
            break
def main():
    # zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_geology/seabed_substrate/multiscale_folk_5/EMODnet_GEO_Seabed_Substrate_All_Res.zip'
    # geodatabase = 'EMODnet_Seabed_Substrate_1M.gdb'
    # layer_index = 0

    zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_human_activities/energy/wind_farms_points/EMODnet_HA_Energy_WindFarms_20240508.zip'
    geodatabase = 'EMODnet_HA_Energy_WindFarms_20240508.gdb'
    layer_index = 1
    variables = ['COUNTRY', 'POWER_MW', 'STATUS']

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