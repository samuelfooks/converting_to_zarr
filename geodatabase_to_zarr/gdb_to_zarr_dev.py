
# In development: Scanning all the features and loading all the variables into data arrays, still processing using windows, but instead
# of saving each variable as a separate zarr dataset, we are combining all the variables into a single dataset, and then saving the final
# still memory intensive and distributing the computation needs further work

import fiona
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
from rasterio.windows import Window
from rasterio.features import rasterize
from affine import Affine
import matplotlib.pyplot as plt
import shapely
import cartopy.crs as ccrs
from shapely.geometry import box
import zarr
import xarray as xr
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime, date
import urllib.request
import zipfile
import dask
from pyproj import CRS
from dask.distributed import LocalCluster, Client
import dask.array as da


class ZarrConverter:
    def __init__(self, zipurl, geodatabase, layer_index, resolution=0.01, variables=None):
        self.zipurl = zipurl
        self.geodatabase = geodatabase
        self.layer_index = layer_index
        self.resolution = resolution
        self.variables = variables
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.zip_file = os.path.basename(zipurl)
        self.temp_zarr_path = f"{self.script_dir}/converted_zarr_files"
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


# create an empty dataset, open each variable zarr dataset, chunk them by lat and lon, merge datasets
    def combine_and_rechunk(self):
        self.combined_dataset = xr.Dataset()
        with dask.config.set(scheduler='single-threaded'):
            for path in self.zarr_vars_paths:
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

    def plot_data_array(self, data_array, title):
        """Function to plot the data array for visualization."""
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax = plt.axes(projection=ccrs.PlateCarree())
        data_array.plot()
        ax.coastlines()
        ax.set_title(title)
        plt.show()



    def process_gdblayer(self, layer_name):
        """
        Process a specific layer from the geodatabase and convert it to Zarr format.
        
        Parameters:
            layer_name: The name of the layer to process.
        """
        vector_file = self.gdb_path
        arco_dir = self.temp_zarr_path  # Assuming this holds the path for the converted directory

        with fiona.open(vector_file, 'r', layer=layer_name) as src:
            self.crs = src.crs
            self.total_bounds = src.bounds
            lon_min, lat_min, lon_max, lat_max = self.total_bounds
            resolution = 0.01
            width = int(np.ceil((lon_max - lon_min) / resolution))
            height = int(np.ceil((lat_max - lat_min) / resolution))
            raster_transform = rasterio.transform.from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

            # Create an empty Dataset for this layer
            dataset = xr.Dataset()
            variable_names = src.schema['properties'].keys()  # Get all variable names
            # Define the latitude and longitude arrays
            latitudes = np.linspace(lat_max, lat_min, height)
            longitudes = np.linspace(lon_min, lon_max, width)
            # Pre-create empty DataArrays for each variable in the dataset
            for native_var in variable_names:
                dataset[native_var] = xr.DataArray(
                    da.zeros((height, width), chunks=(2000, 2000), dtype=np.float32),  # Lazy empty array
                    dims=['latitude', 'longitude'],
                    coords={'latitude': latitudes, 'longitude': longitudes}
                )

            # Define chunk sizes
            chunk_height = 2000
            chunk_width = 2000
            # Loop over window heights (rows)
            for i in range(0, height, chunk_height):
                remainder_y = height - i
                window_height = min(chunk_height, remainder_y)
                print(f"Processing window height {i}:{i + window_height}")

                # Loop over window widths (chunks along longitude)
                for j in range(0, width, chunk_width):
                    remainder_x = width - j
                    window_width = min(chunk_width, remainder_x)
                    print(f"  Processing window width {j}:{j + window_width}")
                    window = Window(j, i, window_width, window_height)

                    window_geom = box(*rasterio.windows.bounds(window, transform=raster_transform))
                    chunk_geoms = []
                    chunk_data = {}

                    with tqdm(total=len(src), desc="Processing features") as pbar:
                        for feature in src:
                            geom = feature['geometry']
                            if isinstance(geom, fiona.model.Geometry):
                                geom = shapely.geometry.shape(geom)

                            if geom.intersects(window_geom):
                                properties = feature['properties']
                                chunk_geoms.append(geom)

                                # Store properties in a dictionary
                                for var_name, value in properties.items():
                                    if var_name not in chunk_data:
                                        chunk_data[var_name] = []
                                    cleaned_value = self.clean_categorical_data(value)
                                    chunk_data[var_name].append(cleaned_value)

                            pbar.update(1)

                    if not chunk_geoms:
                        continue  # Skip empty chunks

                    # Process each variable and update the dataset
                    for native_var, data in chunk_data.items():
                        print(f"Rasterizing variable '{native_var}'")

                        # Clean and encode the data
                        encoded_data, category_mapping = self.encode_categorical_column(pd.Series(data))

                        # Rasterize the chunk for the current variable
                        lazy_raster = dask.delayed(self.rasterize_chunk)(
                            {'geometries': chunk_geoms, 'encoded_data': encoded_data},
                            (window_height, window_width),
                            raster_transform
                        )
                        lazy_raster = da.from_delayed(lazy_raster, shape=(window_height, window_width), dtype=np.float32)

                        # Replace the relevant window in the dataset
                        dataset[native_var][i:i + window_height, j:j + window_width] = lazy_raster
                
                var = list(dataset.data_vars)[0]
                self.plot_data_array(dataset[var], var)
        # Save the dataset to Zarr format
        zarr_var_path = os.path.join(arco_dir, f"{self.title}.zarr")
        dataset.to_zarr(zarr_var_path, mode='w')
        print(f"Zarr created at {zarr_var_path}")

    def rasterize_chunk(self, chunk, raster_shape, transform):
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



# the run process from the ZarrConverter Object
    def run(self):
        self.download_and_extract()
        layer = self.gdblayers[self.layer_index]
        
        self.process_gdblayer(layer)
        
        self.update_and_finalize(layer)

def main():
    # zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_geology/seabed_substrate/multiscale_folk_5/EMODnet_GEO_Seabed_Substrate_All_Res.zip'
    # geodatabase = 'EMODnet_Seabed_Substrate_1M.gdb'
    # layer_index = 0
    # Download the zip file


    zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_native/emodnet_human_activities/energy/wind_farms_points/EMODnet_HA_Energy_WindFarms_20240508.zip'
    geodatabase = 'EMODnet_HA_Energy_WindFarms_20240508.gdb'
    layer_index = 1
    # zipurl = 'https://s3.waw3-1.cloudferro.com/emodnet/emodnet_seabed_habitats/12549/EUSeaMap_2023.zip'
    # geodatabase = 'EUSeaMap_2023.gdb'
    # layer_index = 0
    # resolution ~1 km
    resolution = 0.1

    # for specific columns/variables
    variables = []

    # establish a conversion object, and process each geodatabase layer into zarr datasets
    converter = ZarrConverter(zipurl, geodatabase, layer_index, resolution=resolution, variables=variables)
    converter.run()

if __name__ == '__main__':
    n_workers = 4
    
    with LocalCluster(n_workers=n_workers, threads_per_worker=4, memory_limit='4GiB') as cluster:
        client = Client(cluster)
        print(f"LocalCluster started with {n_workers} workers")
        main()
        client.close()