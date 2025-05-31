import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

def get_et_water_loss_rate(run_title):

    ndvi_path = f"{run_title}/resampled/ndvi_rpj_{run_title}_resampled.tif"
    temperature_path = f"{run_title}/resampled/temp_rpj_{run_title}_resampled.tif"
    lw_ab_path = f"{run_title}/resampled/ab_longwave_rpj_{run_title}_resampled.tif"
    lw_em_path = f"{run_title}/resampled/em_longwave_rpj_{run_title}_resampled.tif"
    sw_ab_path = f"{run_title}/resampled/ab_shortwave_rpj_{run_title}_resampled.tif"

    ndvi = read_raster(ndvi_path)
    temperature = read_raster(temperature_path)
    lw_em = read_raster(lw_em_path)
    lw_ab = read_raster(lw_ab_path)
    sw_ab = read_raster(sw_ab_path)

    net = sw_ab[0] + lw_ab[0] - lw_em[0]

    # # For EVI
    # a_0 = 0.137
    # a_1 = 0.759
    # a_2 = 0.004

    # For NDVI
    a_0 = 0.1505
    a_1 = 0.45
    a_2 = 0.004

    ET_array = np.zeros((len(net), len(net[0])))
    R_n_array = np.zeros((len(net), len(net[0])))

    # plt.imshow(ET_array)
    # # plt.imshow(water_loss_rate)
    # # plt.title('Water loss rate')
    # plt.title('Evapotranspiration (W/m^2)')
    # plt.colorbar()
    # plt.show()
    #
    # print(ndvi[0] * 0.0001)
    # print(temperature[0] * 0.02 - 273)

    for i in range(len(net)):
        for j in range(len(net[0])):
            # R_n = net_radiation(lw_ab[0][i][j], lw_em[0][i][j], sw_ab[0][i][j])
            R_n = net[i][j]
            R_n_array[i][j] = R_n
            e_t = ET(R_n, ndvi[0][i][j], temperature[0][i][j], a_0, a_1, a_2)
            ET_array[i][j] = e_t

    water_loss_rate = ET_array / (2.45 * 10 ** 6 * 1000) * 1000


    output_directory = f"{run_title}/produced"  # Ensure this folder exists or create it
    output_filename = f"et_water_loss_rate_{run_title}.tif"  # Desired output file name

    save_raster(output_directory, output_filename, water_loss_rate)


def read_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1)  # Read first band
        profile = src.profile
    return array, profile

def net_radiation(lw_ab, lw_em, sw_ab):
    return lw_ab + sw_ab - lw_em

# Temperature data needs scaling by factor of 0.02 and VI by 0.0001
def ET(radiation, VI, temperature,a_0,a_1,a_2):
    return radiation*(a_0 + a_1*VI*0.0001 + a_2*(temperature*0.02-273))

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
