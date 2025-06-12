import json
import numpy as np
from pathlib import Path
from argparse import Namespace
from pyproj import Transformer
from shapely.geometry import box

import rasterio
from rasterio.windows import Window

from .pipeline import Pipeline

class CaptureImages(Pipeline):
    """Pipeline task to extract image from source"""

    def __init__(self, args: Namespace):
        super(CaptureImages).__init__()

        self.args = args
        self.batch_size = int(self.args.batch_size) if self.args.batch_size.isnumeric() else 1

    # All path variable not defined in constructor are defined in setup and nowhere else.
    def setup(self, session: Path) -> None:
        """ Reset image loaded """
        # Session path
        self.session = session

        # Orthophoto path.
        self.orthophoto_filepath = Path(self.session, "PROCESSED_DATA", "PHOTOGRAMMETRY", "odm_orthophoto", "odm_orthophoto.tif")
        if not self.orthophoto_filepath.exists() or not self.orthophoto_filepath.is_file():
            raise NameError(f"Orthophoto not found at path: {self.orthophoto_filepath}")
        
        # Stats path.
        self.stats_filepath = Path(self.session, "PROCESSED_DATA", "PHOTOGRAMMETRY", "odm_report", "stats.json")
        if not self.stats_filepath.exists() or not self.stats_filepath.is_file():
            raise NameError(f"Stats not found at path: {self.stats_filepath}")
        
        # Get GSD mean.
        with open(self.stats_filepath, "r") as stat_file:
            stats = json.load(stat_file)
            try:
                self.GSD_mean = round(stats["odm_processing_statistics"]["average_gsd"], 2)
            except:
                raise NameError("Cannot get GSD value")
        
        # Check if orthophoto is in the correct crs.
        with rasterio.open(self.orthophoto_filepath) as ortho:
            if ortho.crs != rasterio.crs.CRS.from_epsg(self.args.matching_crs):
                raise NameError(f"Orthophoto crs doesn't match with desired args {self.args.matching_crs}")

        self.transformer = Transformer.from_crs(self.args.matching_crs, "EPSG:4326", always_xy=True)

    def generator(self):

        self.tile_size = int(self.args.tiles_size_meters // (self.GSD_mean / 100))
        self.size_inline_tile = self.tile_size**2
        self.x_overlap = int(self.tile_size * (1 - self.args.h_shift))
        self.y_overlap = int(self.tile_size * (1 - self.args.v_shift))
        
        counter, tiles, tiles_name, tiles_position = 0, [], [], []
        with rasterio.open(self.orthophoto_filepath) as src:
            for i in range(0, src.height - self.tile_size + 1, self.y_overlap):
                for j in range(0, src.width - self.tile_size + 1, self.x_overlap):
                    window = Window(j, i, self.tile_size, self.tile_size)
                    transform_window = src.window_transform(window)

                    tile = src.read(window=window, indexes=[1, 2, 3])

                    # Apply threshold to avoid keep useless image.                    
                    greyscale_tile = np.sum(tile, axis=0) / 3
                    
                    # Black threshold.
                    percentage_black_pixel = np.sum(greyscale_tile == 0) * 100 / self.size_inline_tile
                    if percentage_black_pixel > self.args.black_pixels_threshold_percentage:
                        continue

                    # White threshold.
                    percentage_white_pixel = np.sum(greyscale_tile == 255) * 100 / self.size_inline_tile
                    if percentage_white_pixel > self.args.white_pixels_threshold_percentage:
                        continue
                    
                    # Get bounds.
                    tile_box = None
                    with rasterio.open(
                        f"/tmp/tmp_tile_uav_{self.session.name}.tif", "w",
                        driver="GTiff",
                        height=tile.shape[1],
                        width=tile.shape[2],
                        count=3,
                        dtype=tile.dtype,
                        crs=src.crs,
                        transform=transform_window
                    ) as dst:
                        tile_box = box(*dst.bounds)

                    # Transpose tile from (3, n, n) to (n, n, 3) and save new path
                    tile = np.transpose(tile, (1, 2, 0))
                    tile_filename = f"{self.session.name}_{int(tile_box.centroid.x)}_{int(tile_box.centroid.y)}.png"
                    lon, lat = self.transformer.transform(tile_box.centroid.x,tile_box.centroid.y)
                    
                    tiles.append(tile)
                    tiles_name.append(tile_filename)
                    tiles_position.append((lon, lat))
                    counter += 1

                    # If enough images, yield 
                    if counter % self.batch_size != 0: continue
                    try:
                        data = {
                            "frames": tiles,
                            "frame_paths": tiles_name,
                            "frames_position": tiles_position
                        }

                        if self.filter(data):
                            counter, tiles, tiles_name, tiles_position = 0, [], [], []

                            yield self.map(data)

                    except StopIteration:
                        return
        yield None
        
    
    def cleanup(self):
        """ nothing to release """
        pass