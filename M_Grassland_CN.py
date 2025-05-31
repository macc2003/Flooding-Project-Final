import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio


def get_grassland_cn(run_title):
    hsg_path = f"{run_title}/produced/hsg_classification_{run_title}.tif"
    hsg_array = read_raster(hsg_path)[0]

    print(hsg_array)

    # Populate the land_cover_data dictionary by calling the get lct_data function
    land_cover_data = get_lct_data()

    CN_array = np.zeros((len(hsg_array), len(hsg_array[0])))

    for i in range(len(hsg_array)):
        for j in range(len(hsg_array[i])):
            CN = land_cover_data[30][int(hsg_array[i][j])]
            CN_array[i][j] = CN

    # plt.figure(figsize=(10, 6))
    # plt.imshow(CN_array, cmap='viridis')
    # plt.colorbar(label="Equivalent Flat Grassland Curve Number")
    # plt.title("Equivalent Flat Grassland Curve Number")
    # plt.axis('off')
    # plt.show()

    output_directory = f"{run_title}/produced"  # Ensure this folder exists or create it
    output_filename = f"grassland_cn_{run_title}.tif"  # Desired output file name

    # save_raster(output_directory, wp_output_filename, wilting_point_array)
    save_raster(output_directory, output_filename, CN_array)


def read_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1)  # Read first band
        profile = src.profile
    return array, profile


def get_lct_data():
    import csv

    lct_data = {}

    with open('Land Cover Types Import.csv', mode ='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Get data from each row of csv file
            land_id = int(row["Value"])
            land_type = row["Land Cover Type"]
            soil_a = float(row["Soil Group A"])
            soil_b = float(row["Soil Group B"])
            soil_c = float(row["Soil Group C"])
            soil_d = float(row["Soil Group D"])

            # if land_id not in lct_data:
            #     lct_data[land] = {}

            # Add the CN value to the dictionary
            lct_data[land_id] = [land_type, soil_a, soil_b, soil_c, soil_d]

        return lct_data

def save_raster(output_dir, output_filename, array):
    output_path = os.path.join(output_dir, output_filename)
    print(f"Saving to: {output_path}")

    height, width = array.shape  # Shape of the array (height and width)
    transform = rasterio.Affine(1, 0, 0, 0, -1, 0)  # Set a dummy affine transform (adjust if needed)
    crs = 'EPSG:4326'  # Example CRS, change if needed (e.g., 'EPSG:4326' for WGS 84)

    try:
        with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype='float32',
                           width=width, height=height, crs=crs, transform=transform) as dst:
            dst.write(array, 1)  # Write the array to the first band
            print(f"Successfully saved {output_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")
