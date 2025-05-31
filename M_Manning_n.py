import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

def get_manning_n(run_title):
    landcover_path = f"{run_title}/resampled/landcover_rpj_{run_title}_resampled.tif"
    built_path = f"{run_title}/resampled/built_char_rpj_{run_title}_resampled.tif"

    landcover_array = read_raster(landcover_path)
    built_array = read_raster(built_path)

    # plt.imshow(built_array[0],vmin = 0, vmax = 1)
    # # plt.imshow(landcover_array[0], vmin = 40, vmax = 60)
    # plt.colorbar()
    # plt.show()
    # print(landcover_array)

    land_cover_manning_data = get_lct_manning_data()

    urban_manning_data = get_urban_manning_data()

    manning_n_array = np.zeros((len(landcover_array[0]), len(landcover_array[0][0])))

    for i in range(len(landcover_array[0])):
        for j in range(len(landcover_array[0][i])):

            value = land_cover_manning_data[landcover_array[0][i][j]][1]
            manning_n = value

            if value == 0:
                urban_class = built_array[0][i][j]
                if urban_class != 0:
                    urban_value = urban_manning_data[built_array[0][i][j]][1]
                else:
                    urban_value = 0.1  # These are the points that are classed as urban in the landcover map built not classified in the built characteristics map. Set to 0.1 as this is the average of the urban manning values
                manning_n = urban_value

            manning_n_array[i][j] = manning_n
    #
    # plt.imshow(manning_n_array, vmin=0, vmax=0.2)
    # plt.colorbar()
    # plt.show()

    output_directory = f"{run_title}/produced"
    output_filename = f"manning_n_{run_title}.tif"

    save_raster(output_directory, output_filename, manning_n_array)


def read_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1)  # Read first band
        profile = src.profile
    return array, profile

def get_lct_manning_data():
    import csv

    lct_manning_data = {}

    with open('Land Cover Manning Import.csv', mode ='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Getting data from each row of csv file
            land_id = int(row["Value"])
            land_type = row["Land Cover Type"]
            manning_n_value= float(row["Manning N"])

            # Add the land value to the dictionary with Land Cover Type and Manning N
            lct_manning_data[land_id] = [land_type, manning_n_value]

    return lct_manning_data

def get_urban_manning_data():
    import csv

    urb_manning_data = {}

    with open('Urban Manning Import.csv', mode ='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            urban_id = int(row["Value"])
            urban_type = row["Description"]
            manning_n_value= float(row["Manning N"])
            
            # Add the land value to the dictionary with Land Cover Type and Manning N
            urb_manning_data[urban_id] = [urban_type, manning_n_value]

    return urb_manning_data


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


