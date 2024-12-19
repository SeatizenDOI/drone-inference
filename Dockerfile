FROM nvcr.io/nvidia/tensorrt:24.11-py3

# Refresh apt && Create user && Setup python with tensorrt and install all dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd -ms /bin/bash seatizen && \ 
    /opt/tensorrt/python/python_setup.sh && \ 
    pip install --no-cache-dir numpy natsort transformers pandas pillow tqdm zenodo_get \
    rasterio scipy matplotlib geocube huggingface_hub geopandas

# Add local directory.
ADD --chown=seatizen ../. /home/seatizen/app/

# Setup workdir in directory.
WORKDIR /home/seatizen/app

# Change with our user.
# USER seatizen

# Define the entrypoint script to be executed.
ENTRYPOINT ["python", "inference.py"]