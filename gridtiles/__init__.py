# file: build_octree.py
import xarray as xr
import numpy as np
import msgpack
import msgpack_numpy as m
from pathlib import Path
from scipy.ndimage import zoom
import pyproj

m.patch()

def project_coords(ds, target_crs="EPSG:3857", z_crs="EPSG:4979"):
    transformer = pyproj.Transformer.from_crs(ds.rio.crs, target_crs, always_xy=True)
    xs, ys = transformer.transform(ds.lon.values, ds.lat.values)
    ds = ds.assign_coords(x=("lon", xs), y=("lat", ys))
    return ds, target_crs, z_crs

def downsample(array, factor):
    return zoom(array, [1/factor]*array.ndim, order=1)

def write_tile(path, variables, bounds, shape, dtype, level, resolution, crs, z_crs, children=None):
    tile = {
        "crs": crs,
        "z_crs": z_crs,
        "bounds": bounds,
        "shape": shape,
        "dtype": dtype,
        "resolution": resolution,
        "level": level,
        "variables": variables,
    }
    if children:
        tile["children"] = children
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        msgpack.pack(tile, f)

def build_octree(ds, tile_path, level=0, max_leaf=32):
    nx, ny, nz = len(ds.x), len(ds.y), len(ds.z)
    bounds = [ds.x.min(), ds.y.min(), ds.z.min(), ds.x.max(), ds.y.max(), ds.z.max()]
    resolution = [(ds.x.max()-ds.x.min())/nx, (ds.y.max()-ds.y.min())/ny, (ds.z.max()-ds.z.min())/nz]

    if max(nx, ny, nz) <= max_leaf:
        # Leaf node
        variables = {v: ds[v].values.astype("float32") for v in ds.data_vars}
        write_tile(tile_path, variables, bounds, (nx, ny, nz), "float32", level, resolution, ds.rio.crs, "EPSG:4979")
        return

    # Internal node downsample
    variables = {v: downsample(ds[v].values, 2).astype("float32") for v in ds.data_vars}
    write_tile(tile_path, variables, bounds, (nx//2, ny//2, nz//2), "float32", level, resolution, ds.rio.crs, "EPSG:4979")

    # Split into 8 children
    x_mid, y_mid, z_mid = nx//2, ny//2, nz//2
    slices = [
        (slice(0,x_mid), slice(0,y_mid), slice(0,z_mid)),
        (slice(0,x_mid), slice(0,y_mid), slice(z_mid,nz)),
        (slice(0,x_mid), slice(y_mid,ny), slice(0,z_mid)),
        (slice(0,x_mid), slice(y_mid,ny), slice(z_mid,nz)),
        (slice(x_mid,nx), slice(0,y_mid), slice(0,z_mid)),
        (slice(x_mid,nx), slice(0,y_mid), slice(z_mid,nz)),
        (slice(x_mid,nx), slice(y_mid,ny), slice(0,z_mid)),
        (slice(x_mid,nx), slice(y_mid,ny), slice(z_mid,nz)),
    ]
    children = []
    for i, s in enumerate(slices):
        child_path = tile_path.parent / f"{tile_path.stem}_{i}.msgpack"
        children.append(child_path.name)
        child_ds = ds.isel(x=s[0], y=s[1], z=s[2])
        build_octree(child_ds, child_path, level+1, max_leaf)
