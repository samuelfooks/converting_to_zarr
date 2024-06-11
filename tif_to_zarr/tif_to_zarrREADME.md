# TIF to Zarr Conversion Tutorial

This Python script is part of a tutorial series on converting raster data (TIF format) to Zarr format. The script is designed to handle TIF files and convert them into a Zarr dataset.

## Dependencies

    pip install -r requirements.txt

## How it Works

The script contains two main functions:

1. `attributes_update(dataset, title, resolution, zip_url)` : This function updates the attributes of the dataset with latitude, longitude, and other metadata. It also sets the Coordinate Reference System (CRS) as an attribute and updates the dataset's attributes with the minimum and maximum latitude and longitude values. Finally, it adds a history attribute indicating the conversion date, a title attribute, and a comment attribute indicating the source of the data and the conversion details.

2. 'tif_to_zarr.py' : This function is intended to convert a TIF file to a Zarr dataset. The function starts by opening the TIF file with rasterio, and it gets the band count from the source file. However, the function is incomplete and does not yet perform the conversion.

## Usage

To use the script, you need to provide the path of the TIF file, the dataset variable, the temporary directory for storing the Zarr files, and the URL of the zip file containing the TIF file. These parameters are currently hardcoded in the script, but you can modify the script to take them as command-line arguments or environment variables.  

## Output

The script is intended to output a Zarr dataset containing the data from the TIF file. The dataset includes the latitude and longitude coordinates, the values of each variable, and metadata such as the CRS and the date of conversion. The dataset is saved in the temporary directory specified in the script.

## Note

This script is part of a tutorial series and is intended for educational purposes. It may not be suitable for production use without modification.

## Docker Deployment

For those who prefer a containerized deployment, we provide a Dockerfile. This Dockerfile creates a Docker image with Python 3.9 as the base image. It sets up the necessary environment, including the installation of required packages, and prepares the application for execution.


You need to supply a zip url and the name for the data variable to be used for the zarr dataset variable.
In the provided

To build the Docker image, you need to have Docker installed on your machine. Navigate to the directory containing the Dockerfile and run the following command:

```bash
docker build -t tiftozarr-tut .

docker run -it tiftozarr-tut python tif_to_zarr.py <your zip url> <name of the data variable>
```