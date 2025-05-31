import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
import matplotlib.pyplot as plt
import random
import rasterio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import re
import sys
import csv
from pathlib import Path

# Importing the functions which produce the required rasters 

from M_Resampling import resampling_continuous, resampling_discrete
from M_Slope import get_slope
from M_Grassland_CN import get_grassland_cn
from M_Soil_Classification import get_soil_data
from M_ET import get_et_water_loss_rate
from M_Saturated_Conductivity import get_saturated_conductivity
from M_Manning_n import get_manning_n
from M_Drainage import get_drainage_rate
from M_CN import get_cn

r_title = "Huaraz" #Set this to match the name of the directory to ensure that file paths are correct

raster_paths = [
    f'{r_title}/input/dem_{r_title}.tif',
    f'{r_title}/produced/slope_adjusted_CN_{r_title}.tif',
    f'{r_title}/produced/et_water_loss_rate_{r_title}.tif',
    f'{r_title}/resampled/snowmelt_rpj_{r_title}_resampled.tif',
    f'{r_title}/produced/usda_classification_{r_title}.tif',
    f'{r_title}/produced/saturation_point_{r_title}.tif',
    f'{r_title}/resampled/fc_rpj_{r_title}_resampled.tif',
    f'{r_title}/produced/wilting_point_{r_title}.tif',
    f'{r_title}/input/moisture_rpj_{r_title}.tif',
    f'{r_title}/produced/grassland_cn_{r_title}.tif',
    f'{r_title}/produced/saturated_conductivity_{r_title}.tif',
    f'{r_title}/produced/drainage_rate_{r_title}.tif',
    f'{r_title}/produced/manning_n_{r_title}.tif'
]

if not all(Path(p).is_file() for p in raster_paths):
    resampling_discrete(r_title)
    resampling_continuous(r_title)
    get_soil_data(r_title)
    get_slope(r_title)
    get_cn(r_title)
    get_et_water_loss_rate(r_title)
    get_saturated_conductivity(r_title)
    get_manning_n(r_title)
    get_drainage_rate(r_title)
    get_grassland_cn(r_title)


with rasterio.open(f'{r_title}/input/dem_{r_title}.tif') as src:
    # Read the image as a NumPy array (first band by default)
    ref = src.read(1)  # '1' reads the first band (in case of multi-band TIFFs)

size = [len(ref),len(ref[0])]

start_time = time.time()

# Define directions for MFD
DIRECTIONS = [
    (-1, 0), (-1, 1), (0, 1),
    (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)
]

def md8_flow_direction_torch(
    dem, head, slope_adj_CN, CN_I, CN_II, CN_III,
    accumulated_infiltration, ET_rate, snowmelt_array,
    usda_array, grassland_infiltration_data, saturation_point_array,
    field_capacity_array, wilting_point_array, equiv_flat_grassland_CN_array,
    saturated_conductivity, drainage_array, manning_n_array,
    resolution, dt, iteration, precipitation
):
    device = dem.device
    runoff = torch.zeros_like(dem)


    WP = wilting_point_array * 1000
    FC = field_capacity_array * 1000
    CN = torch.where(accumulated_infiltration < WP, CN_I,
          torch.where(accumulated_infiltration > FC, CN_III, CN_II))
    S = 25400.0 / (CN + 1e-6) - 254.0
    Sg = 25400.0 / (equiv_flat_grassland_CN_array + 1e-6) - 254.0

    
    head += precipitation*dt
    head += snowmelt_array * (1000 * dt) / (24 * 60**2)

    Q = (head ** 2) / (head + S + 1e-6)
    Qg = (head ** 2) / (head + Sg + 1e-6)
    infiltration = (head - Q) / (head - Qg + 1e-6) * grassland_infiltration_data[usda_array.long()] * dt
    infiltration = torch.minimum(infiltration, head)

    runoff = head - infiltration
    head -= infiltration

    head = torch.clamp(head,0)

    accumulated_infiltration += infiltration
    ET_rate = torch.nan_to_num(ET_rate, nan=0.0)
    accumulated_infiltration -= ET_rate * dt

    accumulated_infiltration[accumulated_infiltration > FC] -= drainage_array[accumulated_infiltration > FC] * dt
    accumulated_infiltration = torch.clamp(accumulated_infiltration, min=0.0)

    dem = dem*1000 # Convert DEM to mm to match the surface water depths

    elevation = dem + head
    # print(f"head = {head}")
    grad_list = []
    total_downhill = torch.zeros_like(elevation)

    for (di, dj) in DIRECTIONS:
        shifted = torch.roll(elevation, shifts=(-di, -dj), dims=(0, 1))
        dist = resolution * ((di**2 + dj**2)**0.5) if (di != 0 or dj != 0) else 1.0
        grad = (shifted-elevation) / dist
        grad_list.append(grad)
        total_downhill += torch.where(grad < 0, grad, 0.0)

    head_update = head
    total_outflow = torch.zeros_like(dem)
    max_speed = torch.zeros_like(elevation)

    total_weight = 0
    for k, (di, dj) in enumerate(DIRECTIONS):
        grad = grad_list[k]
        slope = torch.where(grad < 0, grad, 0.0)
        weight = torch.where(total_downhill < 0, slope / (9*total_downhill), 0.0)
        speed = torch.where(grad < 0,
                            1.0 / manning_n_array * runoff ** (2/3) * torch.sqrt(-slope + 1e-6),
                            0.0)
        max_speed = torch.maximum(max_speed, speed)

    region_max_speed = max_speed.max()
    flow_shifted = torch.zeros_like(dem)

    for k, (di, dj) in enumerate(DIRECTIONS):
        grad = grad_list[k]
        slope = torch.where(grad < 0, grad, 0.0)
        weight = torch.where(total_downhill < 0, slope / (total_downhill), 0.0)
        # print(f"mean slope = {slope.mean()}")
        # print(f"total downhill = {total_downhill.mean()}")
        speed = torch.where(grad < 0,
                            1.0 / manning_n_array * runoff ** (2/3) * torch.sqrt(-slope + 1e-6),
                            0.0)
        flow = weight * head * speed / region_max_speed
        # print(f"speed {k} = {speed}")
        directional_flow_shifted = torch.roll(flow, shifts=(di, dj), dims=(0, 1))
        flow_shifted += directional_flow_shifted
        total_outflow += flow
    head_update += flow_shifted
    head_update -= total_outflow

    weight_list = []

    for k, (di, dj) in enumerate(DIRECTIONS):
        grad = grad_list[k]

        slope = torch.where(grad < 0, grad, 0.0)  # slope is negative
        weight = torch.where(total_downhill < 0, slope / (total_downhill), 0.0)
        weight_list.append(weight)

    dt_new = resolution / (max_speed.max() + 1e-6)

    if head.max() == 0:
        dt_new = 1

    head = head_update
    head = torch.clamp(head,0)

        # Zero the border pixels (4 edges)
    head[0, :] = 0          # top row
    head[-1, :] = 0         # bottom row
    head[:, 0] = 0          # left column
    head[:, -1] = 0         # right column

    return head, accumulated_infiltration, dt_new

def run_model_torch(
    iterations, dem, head, slope_adj_CN, CN_I, CN_II, CN_III,
    accumulated_infiltration, ET_rate, snowmelt_array,
    usda_array, grassland_infiltration_data, saturation_point_array,
    field_capacity_array, wilting_point_array, equiv_flat_grassland_CN_array,
    saturated_conductivity, drainage_array, manning_n_array,
    resolution, precipitation, coords):

    # Initialise the time step
    
    dt = 1
    T = 0
    hydrograph = []

    for it in range(iterations):
        head, accumulated_infiltration, dt = md8_flow_direction_torch(
            dem, head, slope_adj_CN, CN_I, CN_II, CN_III,
            accumulated_infiltration, ET_rate, snowmelt_array,
            usda_array, grassland_infiltration_data, saturation_point_array,
            field_capacity_array, wilting_point_array, equiv_flat_grassland_CN_array,
            saturated_conductivity, drainage_array, manning_n_array,
            resolution, dt, it, precipitation
        )
        T += dt
        hydrograph.append(head[coords[1], coords[0]].item())
        if it % 1000 == 0:
            print(f"Iteration {it}, Runtime = {time.time()-start_time}, dt = {dt:.4f}, T = {T}, head max = {head.max().item():.2f}, head mean = {head.mean()} Infiltration = {accumulated_infiltration.max().item():.2f}")
            # plot_raster(
            #     array=(np.log1p(head)).detach().cpu().numpy(),
            #     title=f"Head at Iteration {it}",
            #     coords=coords)
            # plt.pause(5)
            # plt.close()


    return hydrograph, head.cpu().numpy()

def plot_hydrograph(hydrograph):
    plt.plot(hydrograph)
    plt.xlabel("Iteration")
    plt.ylabel("Water Level at Point")
    plt.title("Hydrograph")
    plt.grid(True)
    plt.show()

def tensorize_input(array, dtype=torch.float32, device=device):
    return torch.tensor(array, dtype=dtype, device=device)

def tensorize_long(array, device=device):
    return torch.tensor(array, dtype=torch.long, device=device)

def prepare_inputs_numpy_to_torch(data_dict, device=device):
    return {
        key: tensorize_input(val, device=device) if key != 'usda_array' and key != 'grassland_infiltration_data'
             else (tensorize_input(val, device=device) if key == 'grassland_infiltration_data'
                   else tensorize_long(val, device=device))
        for key, val in data_dict.items()
    }

def load_raster(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)

def save_raster(array, out_path, crs='EPSG:4326'):
    height, width = array.shape
    transform = rasterio.Affine(1, 0, 0, 0, -1, 0)
    with rasterio.open(out_path, 'w', driver='GTiff',
                       height=height, width=width,
                       count=1, dtype=array.dtype,
                       crs=crs, transform=transform) as dst:
        dst.write(array, 1)

def plot_raster(array, title, coords=None):
    plt.imshow(array, cmap='Blues', origin='upper')
    plt.colorbar(label='Value')
    plt.title(title)
    if coords:
        plt.plot(coords[0], coords[1], 'ro')
    plt.show()

def run_simulation_full(coords, iterations=1000, resolution=30000):
    inputs = {
        'dem': load_raster(f'{r_title}/input/dem_{r_title}.tif'),
        'slope_adj_CN': load_raster(f'{r_title}/produced/slope_adjusted_cn_{r_title}.tif'),
        'ET_rate': load_raster(f'{r_title}/produced/et_water_loss_rate_{r_title}.tif'),
        'snowmelt_array': load_raster(f'{r_title}/resampled/snowmelt_rpj_{r_title}_resampled.tif'),
        'usda_array': load_raster(f'{r_title}/produced/usda_classification_{r_title}.tif'),
        'saturation_point_array': load_raster(f'{r_title}/produced/saturation_point_{r_title}.tif'),
        'field_capacity_array': load_raster(f'{r_title}/resampled/fc_rpj_{r_title}_resampled.tif') / 100,
        'wilting_point_array': load_raster(f'{r_title}/produced/wilting_point_{r_title}.tif'),
        'moisture_content_array': load_raster(f'{r_title}/resampled/moisture_rpj_{r_title}_resampled.tif')*10,
        'equiv_flat_grassland_CN_array': load_raster(f'{r_title}/produced/grassland_cn_{r_title}.tif'),
        'saturated_conductivity': load_raster(f'{r_title}/produced/saturated_conductivity_{r_title}.tif'),
        'drainage_array': load_raster(f'{r_title}/produced/drainage_rate_{r_title}.tif'),
        'manning_n_array': load_raster(f'{r_title}/produced/manning_n_{r_title}.tif'),
        'precipitation': np.zeros((size[0], size[1])),
        'CN_I_array': np.zeros((size[0], size[1])),
        'CN_II_array': np.zeros((size[0], size[1])),
        'CN_III_array': np.zeros((size[0], size[1])),
    }

    grassland_infiltration_data = []
    with open('Infiltration Rates Import.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rate = float(row['Grassland (mm/hr)']) * 10 / 3600
            grassland_infiltration_data.append(rate)
    inputs['grassland_infiltration_data'] = np.array(grassland_infiltration_data)

    cn_data = []
    with open('Curve Numbers Import.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cn_data.append([int(row['CN (AMCI)']), int(row['CN (AMCII)']), int(row['CN (AMCIII)'])])

    slope = inputs['slope_adj_CN']
    for i in range(slope.shape[0]):
        for j in range(slope.shape[1]):
            idx = int(slope[i, j])
            if 0 <= idx < len(cn_data):
                inputs['CN_I_array'][i, j] = cn_data[idx][0]
                inputs['CN_II_array'][i, j] = cn_data[idx][1]
                inputs['CN_III_array'][i, j] = cn_data[idx][2]

    # Create precipitation array using the shape of the loaded DEM
    dem_shape = inputs['dem'].shape
    rows, cols = dem_shape

    precip = np.zeros((rows, cols), dtype=np.float32)

    # Define the precipitation rate and its spatial distribution
    
    # Single point
    # precip[104, 376] = 10.0  # 10 mm/timestep at one grid cell

    # Circular area
    # cx, cy = 693, 1172  # center of rainfall in (x, y) = (col, row)
    cx, cy = 1200, 500  # center of rainfall in (x, y) = (col, row)
    outer_radius = 1000
    inner_radius = 500
    y_idx, x_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    distance = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    precip[distance <= outer_radius] = 0
    precip[distance <= inner_radius] = 10  # mm

    inputs['precipitation'] = precip

    torch_inputs = prepare_inputs_numpy_to_torch(inputs)

    hydrograph, final_head = run_model_torch(
        iterations=iterations,
        dem=torch_inputs['dem'],
        head=torch.zeros_like(torch_inputs['dem']),
        slope_adj_CN=torch_inputs['slope_adj_CN'],
        CN_I=torch_inputs['CN_I_array'],
        CN_II=torch_inputs['CN_II_array'],
        CN_III=torch_inputs['CN_III_array'],
        accumulated_infiltration=torch_inputs['moisture_content_array'],
        ET_rate=torch_inputs['ET_rate'],
        snowmelt_array=torch_inputs['snowmelt_array'],
        usda_array=torch_inputs['usda_array'],
        grassland_infiltration_data=torch_inputs['grassland_infiltration_data'],
        saturation_point_array=torch_inputs['saturation_point_array'],
        field_capacity_array=torch_inputs['field_capacity_array'],
        wilting_point_array=torch_inputs['wilting_point_array'],
        equiv_flat_grassland_CN_array=torch_inputs['equiv_flat_grassland_CN_array'],
        saturated_conductivity=torch_inputs['saturated_conductivity'],
        drainage_array=torch_inputs['drainage_array'],
        manning_n_array=torch_inputs['manning_n_array'],
        resolution=resolution,
        precipitation=torch_inputs['precipitation'],
        coords=coords
    )

    plot_hydrograph(hydrograph)
    plot_raster(np.log1p(final_head), 'Final Water Head (log-scaled)', coords)
    save_raster(final_head.astype(np.float32), f'final_head_output_{r_title}.tif')

run_simulation_full(5000)






