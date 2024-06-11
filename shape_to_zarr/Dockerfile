# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Create a directory for the zip file
RUN mkdir /zipfiles

# Add permissions for accessing /zipfiles directory
RUN chmod 777 /zipfiles

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV ARCO_ASSET_TEMP_DIR /output-data

CMD ["python", "shp_to_zarr.py"]


