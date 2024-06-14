# Convert common geospatial data structures into Zarr

This is a tutorial divided into a few parts demonstrating how to convert common geospatial data structures into one Analyis Ready Cloud Optimized format called Zarr. 

Read more about Zarr:
https://zarr.readthedocs.io/en/stable/getting_started.html

## Tutorial Outline

### Install required dependencies
    mamba install --file requirements.txt
    mamba env update --file environment.yml --name convertzarrenv

### geodatabase_to_zarr

* Converting Geodatabases to Zarr using Fiona, Geopandas, Rasterio and Zarr
* Comparing the converted dataset to the original

**If you choose a resolution finer than 0.01 degrees expect the resource consumption to be heavier than a laptop with 16GB of RAM and a 12 Core Intel i7 processor
### shape_to_zarr

* Converting Shape files to Zarr using Fiona, Geopandas, Rasterio and Zarr

**If you choose a resolution finer than 0.01 degrees expect the resource consumption to be heavier than a laptop with 16GB of RAM and a 12 Core Intel i7 processor

### tif_to_zarr

* Converting Geotiff files to Zarr using Rasterio, Zarr
* Comparing the converted dataset to the original


## Getting Started

Each part of the tutorial has its own directory with a README.md file that provides an outline of what each conversion process does.

Happy converting!

Licence: CC BY-4.0
