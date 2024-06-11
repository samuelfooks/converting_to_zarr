# Geodatabase to Zarr Conversion Tutorial

This Python script is part of a tutorial series on converting vector data to Zarr format. The script is designed to download a zip file containing a Geodatabase, extract the Geodatabase, and convert a specified layer, and its columns(variables) it into a Zarr dataset.

## Dependencies
 pip install -r requirements.txt

## How it Works

The script 'gdb2zarr.py'  contains the following general components

1. `attributes_update(dataset, title, resolution, zip_url)`: This function updates the attributes of the dataset with latitude, longitude, and other metadata.

2. `download_and_extract_zip(zip_url)`: This function downloads a zip file from a given URL, extracts it, and returns the path to the shapefile.

3. `gdf2zarrconverter(gdb_file_path, variable, resolution, arco_asset_tmp_path, zip_url)`: This function converts the shapefile to a Zarr dataset. It first reads the shapefile, rasterizes the geometries, and then creates an xarray dataset from the raster. The dataset is then chunked and sorted, and the attributes are updated. Finally, the dataset is saved as a Zarr file.

4. In the main part of the script, it downloads and extracts a zip file containing a geodatabase. It then converts each layer and specific variables to a Zarr dataset using the `gdf2zarrconverter` function. The script processes each variable in the shapefile separately, creating a separate Zarr dataset for each one. These datasets are then rechunked and combined into a single Zarr dataset.  

## Usage

To use the script, you need to provide the URL of the zip file containing the Geodatabase, the resolution for the rasterization, and the temporary directory for storing the Zarr files. These parameters are currently hardcoded in the script, but you can modify the script to take them as command-line arguments or environment variables.

## Output

The script outputs a Zarr dataset containing the data from the geodatabase. The dataset includes the latitude and longitude coordinates, the values of each variable, and metadata such as the CRS and the date of conversion. The dataset is saved in the temporary directory specified in the script.

## Note

This script is part of a tutorial series and is intended for educational purposes. It may not be suitable for production use without modification.