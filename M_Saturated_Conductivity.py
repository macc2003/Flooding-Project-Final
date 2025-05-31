import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

def get_saturated_conductivity(run_title):
    sand_path = f"{run_title}/resampled/sand_rpj_{run_title}_resampled.tif"
    clay_path = f"{run_title}/resampled/clay_rpj_{run_title}_resampled.tif"

    sand = read_raster(sand_path)[0]
    clay = read_raster(clay_path)[0]

    conductivity_data = get_conductivity_data()

    # Use the get_drainage function from above to extract the drainage value from the data.
    # This will return None if no value exists for the given sand and clay

    conductivity_array = np.zeros((len(sand), len(sand[0])))

    for i in range(len(sand)):
        for j in range(len(sand[0])):
            if sand[i, j] % 2 == 0:
                sand_content = sand[i, j]
            else:
                sand_content = sand[i, j] - 1

            if clay[i, j] % 2 == 0:
                clay_content = clay[i, j]
            else:
                clay_content = clay[i, j] + 1

            conductivity_array[i, j] = get_conductivity(sand_content, clay_content, conductivity_data)

    # convert from cm/day to mm/s
    conductivity_array = conductivity_array * 10 / (24 * 60 ** 2)
  
    save_raster(f"{run_title}/produced", f"saturated_conductivity_{run_title}.tif", conductivity_array)


def read_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1)  # Read first band
        profile = src.profile
    return array, profile

def get_conductivity_data():
    """ This function opens the csv file specified inside the function and returns the populated dictionary"""

    import csv

    K_s_data = {}

    with open('Simple Conductivities Import.csv', mode ='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        print("CSV Headers:", reader.fieldnames)  
        for row in reader:
            # Get data from each row of csv file
            sand_no = int(row["sand"])
            clay_pct = int(row["clay"])
            K_s_val = float(row["conductivity"])

            # Adding the dr_key (sand + clay/100) to the dictionary as the key with the drainage as the value
            K_s_key = sand_no + (clay_pct/100)
            K_s_data[K_s_key] = K_s_val

    return K_s_data


def get_conductivity(sand, clay, conductivities):
    """ This function returns the drainage value for the give sand and clay values from the given drainage
    data dictionary which has been previously created with the get_drainage_rates_data function"""

    conductivity_key = sand + clay/100
    conductivity_value = conductivities.get(conductivity_key)
    if conductivity_value == None:
        conductivity_value = 0
    return conductivity_value

def save_raster(output_dir, output_filename, array):
    
    output_path = os.path.join(output_dir, output_filename)

    print(f"Saving to: {output_path}")

    height, width = array.shape  
    transform = rasterio.Affine(1, 0, 0, 0, -1, 0)  
    crs = 'EPSG:4326'

    try:
        with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype='float32',
                           width=width, height=height, crs=crs, transform=transform) as dst:
            dst.write(array, 1)  # Write the array to the first band
            print(f"Successfully saved {output_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

