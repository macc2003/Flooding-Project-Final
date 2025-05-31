import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

def get_soil_data(run_title):
    sand_path = f"{run_title}/resampled/sand_rpj_{run_title}_resampled.tif"
    clay_path = f"{run_title}/resampled/clay_rpj_{run_title}_resampled.tif"

    with rasterio.open(clay_path) as clay_src:
        clay = clay_src.read(1)
        profile = clay_src.profile 

    with rasterio.open(sand_path) as sand_src:
        sand = sand_src.read(1)

    usda = classify_soil_usda_array(sand, clay)

    usda_classes = [
        "Sand", "Loamy Sand", "Sandy Loam", "Sandy Clay Loam", "Clay Loam",
        "Silty Clay Loam", "Loam", "Silt Loam", "Silt", "Sandy Clay",
        "Silty Clay", "Clay", "Unknown"]

    usda_classes = sorted(set(usda_classes))

    usda_int = np.full(usda.shape, -1)

    for i, name in enumerate(usda_classes):
        usda_int[usda == name] = i

    wilting_point_array = calculate_wilting_point_array(sand, clay)

    saturation_point_array = calculate_saturation_point_array(sand, clay)

    hsg = classify_soil_hsg_array(sand, clay)

    # Map group letters to integers
    hsg_classes = ["A", "B", "C", "D", "Unknown"]
    hsg_int = np.full(hsg.shape, -1)  # -1 for Unknown

    for i, group in enumerate(hsg_classes):
        hsg_int[hsg == group] = i

    hsg_to_int = {"A": 1, "B": 2, "C": 3, "D": 4, "Unknown": 5}
    hsg_int_export = np.vectorize(hsg_to_int.get)(hsg)

    export_profile = profile.copy()
    export_profile.update({
        'dtype': 'uint8',
        'count': 1,
        'compress': 'lzw'
    })

    output_directory = f"{run_title}/produced" 
    wp_output_filename = f"wilting_point_{run_title}.tif" 
    sp_output_filename = f'saturation_point_{run_title}.tif'
    usda_ouput_filename = f'usda_classification_{run_title}.tif'
    hsg_ouput_filename = f'hsg_classification_{run_title}.tif'

    save_raster(output_directory, wp_output_filename, wilting_point_array)
    save_raster(output_directory, sp_output_filename, wilting_point_array)
    save_raster(output_directory, usda_ouput_filename, usda_int)
    save_raster(output_directory, hsg_ouput_filename, hsg_int)

def classify_soil_usda_array(sand, clay):
    silt = 100 - sand - clay

    result = np.full(sand.shape, 'Unknown', dtype=object)

    result[(clay >= 40) & (silt >= 40)] = 'Silty Clay'
    result[(clay >= 40) & (sand <= 45)] = 'Clay'
    result[(clay >= 35) & (sand > 45)] = 'Sandy Clay'
    result[(clay >= 27) & (clay < 40) & (sand >= 20) & (sand <= 45)] = 'Clay Loam'
    result[(clay >= 20) & (clay < 35) & (sand > 45) & (silt <= 27)] = 'Sandy Clay Loam'
    result[(clay >= 7) & (clay < 27) & (sand <= 52) & (silt >= 27) & (silt <= 50)] = 'Loam'
    result[(sand >= 85) & (silt + 1.5 * clay < 15)] = 'Sand'
    result[(sand >= 70) & (silt + 1.5 * clay >= 15) & (silt + 2 * clay < 30)] = 'Loamy Sand'
    result[(silt >= 50) & (clay >= 27)] = 'Silty Clay Loam'
    result[(clay >= 27) & (clay < 40) & (sand <= 20)] = 'Loam'
    result[(clay <= 7) & (silt < 50) & (silt + 2 * clay >= 30)] = 'Sandy Loam'
    result[(clay > 7) & (clay < 20) & (sand > 52) & (silt + 2 * clay >= 30)] = 'Sandy Loam'
    result[(clay > 14) & (clay < 27) & (silt > 50)] = 'Silt Loam'
    result[(clay <= 14) & (silt > 50) & (silt < 80)] = 'Silt Loam'
    result[(clay <= 14) & (silt >= 80)] = 'Silt'

    return result

def classify_soil_hsg_array(sand, clay):
    usda = classify_soil_usda_array(sand, clay)

    hsg = np.full(usda.shape, 'Unknown', dtype=object)

    # Define USDA groupings
    A = ["Sand"]
    B = ["Loamy Sand", "Sandy Loam"]
    C = ["Sandy Clay Loam", "Clay Loam", "Silty Clay Loam", "Loam", "Silt Loam", "Silt"]
    D = ["Sandy Clay", "Silty Clay", "Clay"]

    hsg[np.isin(usda, A)] = "A"
    hsg[np.isin(usda, B)] = "B"
    hsg[np.isin(usda, C)] = "C"
    hsg[np.isin(usda, D)] = "D"

    return hsg

def calculate_wilting_point_array(sand, clay):
    usda = classify_soil_usda_array(sand, clay)

    usda_classes = np.array([
        "Sand", "Loamy Sand", "Sandy Loam", "Sandy Clay Loam", "Clay Loam",
        "Silty Clay Loam", "Loam", "Silt Loam", "Silt", "Sandy Clay",
        "Silty Clay", "Clay", "Unknown"
    ])

    wilting_points = np.array([
        0.04, 0.06, 0.08, 0.13, 0.25, 0.22, 0.10, 0.13, 0.13,
        0.13, 0.27, 0.28, 0.14  
    ])

    wilting_point_array = np.full(usda.shape, 0.0)

    for i, soil_class in enumerate(usda_classes):
        wilting_point_array[usda == soil_class] = wilting_points[i]

    return wilting_point_array

def calculate_saturation_point_array(sand, clay):
    usda = classify_soil_usda_array(sand, clay)

    usda_classes = np.array([
        "Sand", "Loamy Sand", "Sandy Loam", "Sandy Clay Loam", "Clay Loam",
        "Silty Clay Loam", "Loam", "Silt Loam", "Silt", "Sandy Clay",
        "Silty Clay", "Clay", "Unknown"
    ])

    saturation_points = np.array([
       0.375, 0.39 , 0.387, 0.384, 0.442, 0.482, 0.399, 0.439, 0.489,
       0.385, 0.481, 0.459, 0.426  # Average for unknown
    ])

    saturation_point_array = np.full(usda.shape, 0.0)

    for i, soil_class in enumerate(usda_classes):
       saturation_point_array[usda == soil_class] = saturation_points[i]

    return saturation_point_array


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


