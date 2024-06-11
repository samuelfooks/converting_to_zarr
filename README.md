# Convert common geospatial data structures into Zarr

This is a tutorial divided into a few parts demonstrating how to convert common geospatial data structures into one Analyis Ready Cloud Optimized format called Zarr. 

Read more about Zarr:
https://zarr.readthedocs.io/en/stable/getting_started.html

## Tutorial Outline

### Install required dependencies
    pip install -r requirements.txt

### geodatabase_to_zarr

* Converting Geodatabases to Zarr using Fiona, Geopandas, Rasterio and Zarr
* Comparing the converted dataset to the original

### shape_to_zarr

* Converting Shape files to Zarr using Fiona, Geopandas, Rasterio and Zarr
* There is a Dockerfile included with the script, if you clone this repository locally and not on edito you can run the docker scripts and create your own zarr files or deploy it as a conversion process yourself

### tif_to_zarr

* Converting Geotiff files to Zarr using Rasterio, Zarr
* Comparing the converted dataset to the original

* There is a Dockerfile included with the script, if you clone this repository locally and not on edito you can run the docker scripts and create your own zarr files or deploy it as a conversion process yourself


## Getting Started

Each part of the tutorial has its own directory with a README.md file that provides an outline of what each conversion process does.

Happy converting!

Licence: CC BY-4.0
