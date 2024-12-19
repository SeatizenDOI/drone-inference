# Use an official Python runtime as a parent image
FROM python:3.12-slim-bullseye

# Install lib, create user.
RUN apt-get update && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd -ms /bin/bash seatizen && \
    pip install --no-cache-dir \
    pandas==2.2.3 \
    tqdm==4.67.1 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    huggingface_hub==0.26.5 \
    transformers==4.47.0 \
    matplotlib==3.10.0 \
    scipy==1.14.1 \
    geopandas==1.0.1 \
    geocube==0.7.0 \
    pyproj==3.7.0 \
    shapely==2.0.6 \
    rasterio==1.4.3

# Add local directory and change permission.
ADD --chown=seatizen ../. /home/seatizen/app/

# Setup workdir in directory.
WORKDIR /home/seatizen/app

# Change with our user.
USER seatizen

# Define the entrypoint script to be executed.
ENTRYPOINT ["python", "inference.py"]