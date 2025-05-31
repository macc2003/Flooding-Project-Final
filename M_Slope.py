import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

def get_slope(run_title):
    slope = compute_and_plot_slope(f"{run_title}/input/dem_{run_title}.tif")
    output_directory = f"{run_title}/produced"
    output_filename = f"slope_{run_title}.tif"  

    save_raster(output_directory, output_filename, slope)

def compute_and_plot_slope(input_dem_path):
    with rasterio.open(input_dem_path) as src:
        dem = src.read(1, masked=True)
        transform = src.transform
        pixel_size_x = transform.a
        pixel_size_y = -transform.e  

        # Compute gradient
        dz_dy, dz_dx = np.gradient(dem, pixel_size_y, pixel_size_x)

        # Compute slope magnitude (in metres per metre)
        slope = np.sqrt(dz_dx**2 + dz_dy**2)



    return slope

def save_raster(output_dir, output_filename, array):

    output_path = os.path.join(output_dir, output_filename)

    print(f"Saving to: {output_path}")

    height, width = array.shape 
    transform = rasterio.Affine(1, 0, 0, 0, -1, 0)  
    crs = 'EPSG:4326'

    try:
        with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype='float32',
                           width=width, height=height, crs=crs, transform=transform) as dst:
            dst.write(array, 1) 
            print(f"Successfully saved {output_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

