# TIF to Zarr Conversion Tutorial

This Python script is part of a tutorial series on converting raster data (TIF format) to Zarr format. The script is designed to handle TIF files and convert them into a Zarr dataset.

## Dependencies

The script uses several Python libraries, including:

- xarray
- numpy
- rasterio
- dask
- pandas

## How it Works

The script contains two main functions:

1. [`attributes_update(dataset, title, resolution, zip_url)`](command:_github.copilot.openSymbolFromReferences?%5B%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fsamwork%2FDocuments%2Fcoding%2Fzarr_conversion%2Ftif2zarr%2Ftif_to_zarr.py%22%2C%22scheme%22%3A%22file%22%7D%2C%7B%22line%22%3A21%2C%22character%22%3A4%7D%5D "tif2zarr/tif_to_zarr.py"): This function updates the attributes of the dataset with latitude, longitude, and other metadata. It also sets the Coordinate Reference System (CRS) as an attribute and updates the dataset's attributes with the minimum and maximum latitude and longitude values. Finally, it adds a history attribute indicating the conversion date, a title attribute, and a comment attribute indicating the source of the data and the conversion details.

2. [`tif2zarr(file_path, band_variable, arco_asset_temp_dir, zip_url)`](command:_github.copilot.openSymbolFromReferences?%5B%7B%22%24mid%22%3A1%2C%22path%22%3A%22%2Fhome%2Fsamwork%2FDocuments%2Fcoding%2Fzarr_conversion%2Ftif2zarr%2Ftif_to_zarr.py%22%2C%22scheme%22%3A%22file%22%7D%2C%7B%22line%22%3A57%2C%22character%22%3A4%7D%5D "tif2zarr/tif_to_zarr.py"): This function is intended to convert a TIF file to a Zarr dataset. The function starts by opening the TIF file with rasterio, and it gets the band count from the source file. However, the function is incomplete and does not yet perform the conversion.

## Usage

To use the script, you need to provide the path of the TIF file, the band variable, the temporary directory for storing the Zarr files, and the URL of the zip file containing the TIF file. These parameters are currently hardcoded in the script, but you can modify the script to take them as command-line arguments or environment variables.

## Output

The script is intended to output a Zarr dataset containing the data from the TIF file. The dataset includes the latitude and longitude coordinates, the values of each variable, and metadata such as the CRS and the date of conversion. The dataset is saved in the temporary directory specified in the script.

## Note

This script is part of a tutorial series and is intended for educational purposes. It may not be suitable for production use without modification.

## Docker Deployment

For those who prefer a containerized deployment, we provide a Dockerfile. This Dockerfile creates a Docker image with Python 3.9 as the base image. It sets up the necessary environment, including the installation of required packages, and prepares the application for execution.

The Dockerfile performs the following steps:

1. Uses Python 3.9 as the base image.
2. Sets the working directory in the container to `/app`.
3. Creates a directory `/tiffiles` for storing TIF files.
4. Adds permissions for accessing the `/tiffiles` directory.
5. Installs `wget` using `apt-get`.
6. Copies the contents of the current directory into the `/app` directory in the container.
7. Installs Python packages specified in [`requirements.txt`] using `pip`.
8. Exposes port 80 for outside access.
9. Sets an environment variable `ARCO_ASSET_TEMP_DIR` to `/output-data`.
10. Sets the command to run [`tif_to_zarr.py`] when the container launches.
11. You need to supply a zip url and the name for the data variable to be used for the zarr dataset.

To build the Docker image, you need to have Docker installed on your machine. Navigate to the directory containing the Dockerfile and run the following command:

```bash
docker build -t tiftozarr-tut .

docker run -it tiftozarr-tut
```

**Please note that the `tif2zarr` function is incomplete and does not yet perform the conversion from TIF to Zarr.**