import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

def get_cn(run_title):
    landcover_path = f"{run_title}/resampled/landcover_rpj_{run_title}_resampled.tif"
    soil_path = f"{run_title}/produced/hsg_classification_{run_title}.tif"
    slope_path = f"{run_title}/produced/slope_{run_title}.tif"

    # Populating the land_cover_data dictionary by calling the get lct_data function
    land_cover_data = get_lct_data()

    cn_data = get_cn_data()

    array1, profile1 = read_raster(landcover_path)
    array2, profile2 = read_raster(soil_path)
    slope_array, profile = read_raster(slope_path)

    if array1.shape != array2.shape:
        raise ValueError("Raster dimensions do not match!")

    CN_array = np.zeros((len(array1), len(array1[0])))
    slope_adjusted_CN_array = np.zeros((len(array1), len(array1[0])))

    for i in range(len(array1)):
        for j in range(len(array1[i])):
            CN = land_cover_data[int(array1[i][j])][int(array2[i][j])]
            slope = slope_array[i][j]
            slope_adj_CN = slope_adjusted_CN(int(CN), slope)
            CN_array[i][j] = CN
            slope_adjusted_CN_array[i][j] = int(slope_adj_CN)

    # # 5. Plot the result
    # plt.figure(figsize=(10, 6))
    # plt.imshow(slope_adjusted_CN_array, cmap='viridis')
    # plt.colorbar(label="Slope Adjusted Curve Number")
    # plt.title("Slope Adjusted Curve Number")
    # plt.axis('off')
    # plt.show()

    output_directory = f"{run_title}/produced"
    output_filename = f"slope_adjusted_cn_{run_title}.tif"

    save_raster(output_directory, output_filename, slope_adjusted_CN_array, profile1)

def read_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1)  # Read first band
        profile = src.profile
    return array, profile

# Function for retrieving the CN from landcover and soil classification
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



# Function for converting AMCII CN to AMC I/III
def get_cn_data():
    # Reading data in from a CSV file with headers and put data into a list
    import csv

    cn_list = []
    amc_check = 0
    with open('Curve Numbers Import.csv', mode ='r') as file:
        reader = csv.DictReader(file)

        # Get data from each row of csv file
        for row in reader:
            amc2 = int(row["CN (AMCII)"])

            # Check that the next row of data has increased AMCII number by 1 before getting other data values
            if amc2 == amc_check:
                amc1 = int(row["CN (AMCI)"])
                amc3 = int(row["CN (AMCIII)"])

                # Append the values to the cn_list as a list of 3 valuses (AMCI, AMCII, AMCIII)
                cn_list.append([amc1, amc2, amc3])
                amc_check += 1
            else:
                print("Error in CSV Data - AMC II number out of sequence!")
                return []

        return cn_list

def read_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1)  # Read first band
        profile = src.profile
    return array, profile

def slope_adjusted_CN(CN,slope):
    adj_CN = (CN*(50-0.5*CN)/(CN+75.43))*(1-np.exp(-7.125*(slope-0.05)))+CN
    return adj_CN

def save_raster(output_dir, output_filename, array, profile):
    output_path = os.path.join(output_dir, output_filename)

    print(f"Saving to: {output_path}")
    profile.update(dtype=rasterio.float32, count=1)

    try:
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(array, 1)  # Write the array to the first band
            print(f"Successfully saved {output_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

