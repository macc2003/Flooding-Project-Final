import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

def resampling_continuous(run_title):
    ref_path = os.path.join(f"{run_title}/input", f"dem_{run_title}.tif")
    with rasterio.open(ref_path) as ref:
        ref_array = ref.read(1)
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds
        ref_dtype = ref.dtypes[0]

        # Paths to process

    input_paths = {
        "temp": f"{run_title}/input/temp_rpj_{run_title}.tif",
        "sand": f"{run_title}/input/sand_rpj_{run_title}.tif",
        "clay": f"{run_title}/input/clay_rpj_{run_title}.tif",
        "ndvi": f"{run_title}/input/ndvi_rpj_{run_title}.tif",
        "fc": f"{run_title}/input/fc_rpj_{run_title}.tif",
        "em_lw": f"{run_title}/input/em_longwave_rpj_{run_title}.tif",
        "ab_lw": f"{run_title}/input/ab_longwave_rpj_{run_title}.tif",
        "ab_sw":    f"{run_title}/input/ab_shortwave_rpj_{run_title}.tif",
        "snowmelt": f"{run_title}/input/snowmelt_rpj_{run_title}.tif",
        "moisture": f"{run_title}/input/moisture_rpj_{run_title}.tif"

    }

    output_folder = f"{run_title}/resampled"

    resampled_rasters = {"Reference": ref_array}

    for key, path in input_paths.items():
        data, dtype, original_filename = reproject_match_bilinear(path, ref_crs, ref_transform, ref_width, ref_height)
        new_filename = add_suffix(original_filename, "_resampled")
        save_raster(output_folder, new_filename, data, ref_transform, ref_crs, dtype)
        resampled_rasters[key] = data
        

def resampling_discrete(run_title):
    '''Input file directory'''
    # -- Load reference raster --
    ref_path = os.path.join(f"{run_title}/input", f"dem_{run_title}.tif")
    with rasterio.open(ref_path) as ref:
        ref_array = ref.read(1)
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds
        ref_dtype = ref.dtypes[0]

        # Paths to process
    input_paths = {
        "landcover":    f"{run_title}/input/landcover_rpj_{run_title}.tif",
        "built": f"{run_title}/input/built_char_rpj_{run_title}.tif",
        "rivers": f"{run_title}/input/rivers_{run_title}.tif"
    }

    output_folder = f"{run_title}/resampled"

    resampled_rasters = {"Reference": ref_array}

    for key, path in input_paths.items():
        data, dtype, original_filename = reproject_match_nearest(path, ref_crs, ref_transform, ref_width, ref_height)
        new_filename = add_suffix(original_filename, "_resampled")
        save_raster(output_folder, new_filename, data, ref_transform, ref_crs, dtype)
        resampled_rasters[key] = data

   
def reproject_match_bilinear(source_path, dst_crs, dst_transform, dst_width, dst_height):
    with rasterio.open(source_path) as src:
        dst_array = np.empty((1, dst_height, dst_width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        return dst_array[0], src.dtypes[0], os.path.basename(source_path)

def reproject_match_nearest(source_path, dst_crs, dst_transform, dst_width, dst_height):
    with rasterio.open(source_path) as src:
        dst_array = np.empty((1, dst_height, dst_width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
        return dst_array[0], src.dtypes[0], os.path.basename(source_path)

def add_suffix(filename, suffix="_resampled"):
    name, ext = os.path.splitext(filename)
    return f"{name}{suffix}{ext}"

def save_raster(output_dir, filename, data, transform, crs, dtype):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data, 1)
    return output_path
