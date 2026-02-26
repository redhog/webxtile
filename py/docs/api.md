# Gridtiles API Reference

## `write_gridtiles`

```python
gridtiles.write_gridtiles(
    ds: xr.Dataset,
    path: str | Path,
    *,
    spatial_dims: list[str] | None = None,
    max_leaf: int = 32,
    crs: str | None = None,
    z_crs: str | None = None,
) -> None
```

Write an xarray Dataset to a gridtiles directory.  The tree structure is chosen automatically from the number of spatial dimensions:

- **2 spatial dims** → **quadtree** (4 children per node, child indices 0–3)
- **3 spatial dims** → **octree** (8 children per node, child indices 0–7)

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ds` | `xr.Dataset` | Dataset to write. Must have 2 or 3 spatial dimensions. All data variables and coordinates are preserved. |
| `path` | `str \| Path` | Output directory. Created (including parents) if it does not exist. |
| `spatial_dims` | `list[str] \| None` | Names of the 2 or 3 dimensions to use for spatial decomposition, e.g. `["x", "y", "z"]` (octree) or `["x", "y"]` (quadtree). The length of this list determines the tree type. Auto-detected from CF `axis` / `standard_name` attributes when `None`. See [CF detection rules](#cf-dimension-auto-detection). |
| `max_leaf` | `int` | Maximum grid points per tile along any single spatial axis. A tile whose largest spatial axis has ≤ `max_leaf` points becomes a leaf (full-resolution) node. Default `32`. |
| `crs` | `str \| None` | Horizontal CRS identifier (e.g. `"EPSG:3857"`). Stored in `metadata.msgpack` for downstream consumers; not used by the library itself. |
| `z_crs` | `str \| None` | Vertical CRS identifier (e.g. `"EPSG:4979"`). Stored in metadata only. Only meaningful for 3-D (octree) datasets; pass `None` for 2-D datasets. |

**Raises**

- `ValueError` – if `spatial_dims` has fewer than 2 or more than 3 elements.
- `ValueError` – if auto-detection fails because no suitable dimensions are found.

---

## `read_gridtiles`

```python
gridtiles.read_gridtiles(
    path: str | Path,
    *,
    level: int | None = None,
    bbox: list[float] | None = None,
) -> xr.Dataset
```

Read a gridtiles directory into an xarray Dataset.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Directory produced by `write_gridtiles`. |
| `level` | `int \| None` | Tree depth to read. `0` = root tile only (coarsest). `None` (default) = all leaf tiles (full resolution). When a branch reaches a leaf before `level` is hit, that leaf is returned instead — so coverage is always complete. Applies to both quadtree (2-D) and octree (3-D) datasets. |
| `bbox` | `list[float] \| None` | Spatial bounding-box filter. Length must be `2 × len(spatial_dims)`: `[x_min, y_min, x_max, y_max]` (4 elements) for 2-D quadtree data, or `[x_min, y_min, z_min, x_max, y_max, z_max]` (6 elements) for 3-D octree data. Tiles that do not intersect the box are skipped entirely. |

**Returns** `xr.Dataset` with:
- Data variables at the requested resolution with original `dtype` and `attrs` restored.
- Spatial coordinates rebuilt from tile data; other coordinates (e.g. time) restored from metadata.
- Global dataset attributes restored.

**Raises**

- `ValueError` – if no tiles are found (e.g. bbox excludes everything).

---

## xarray engine

After `pip install` the package registers itself as an xarray backend engine named `gridtiles`:

```python
# 2-D quadtree dataset
ds = xr.open_dataset("tiles/", engine="gridtiles")
ds = xr.open_dataset("tiles/", engine="gridtiles", level=2)
ds = xr.open_dataset("tiles/", engine="gridtiles", bbox=[0, 0, 100, 100])          # [xmin, ymin, xmax, ymax]

# 3-D octree dataset
ds = xr.open_dataset("tiles/", engine="gridtiles")
ds = xr.open_dataset("tiles/", engine="gridtiles", level=2)
ds = xr.open_dataset("tiles/", engine="gridtiles", bbox=[0, 0, 0, 100, 100, 50])  # [xmin, ymin, zmin, xmax, ymax, zmax]
```

Standard xarray keyword arguments (`drop_variables`, `mask_and_scale`, etc.) are accepted and forwarded. The gridtiles-specific keywords `level` and `bbox` are also accepted; the correct `bbox` length (4 or 6) is determined by the dataset's `spatial_dims`.

Registration happens via the `xarray.backends` entry-point declared in `pyproject.toml`.

---

## xarray accessor

```python
ds.gridtiles.to_gridtiles(path, **kwargs)
```

Convenience accessor equivalent to calling `write_gridtiles(ds, path, **kwargs)`. All keyword arguments of `write_gridtiles` are accepted.

---

## CF dimension auto-detection

When `spatial_dims` is not provided, `write_gridtiles` inspects every dimension name in the dataset and classifies it as X, Y, or Z using two rules applied in order:

1. **CF attributes** on the coordinate variable associated with the dimension:
   - `axis = "X"` → X; `axis = "Y"` → Y; `axis = "Z"` → Z
   - `standard_name` in a recognised set (see table below) → X / Y / Z

2. **Name heuristics** (case-insensitive match against a fixed list):

| Axis | Recognised standard\_names | Recognised dimension names |
|------|---------------------------|---------------------------|
| X | `projection_x_coordinate`, `longitude`, `grid_longitude` | `x`, `lon`, `longitude`, `easting` |
| Y | `projection_y_coordinate`, `latitude`, `grid_latitude` | `y`, `lat`, `latitude`, `northing` |
| Z | `depth`, `altitude`, `height`, `air_pressure`, `geopotential_height`, … | `z`, `depth`, `altitude`, `height`, `elevation`, `level`, `lev`, `plev` |

**Outcome:**

| Detected axes | Tree type | Children per node |
|---|---|:---:|
| X + Y | **Quadtree** (2-D) | up to 4 |
| X + Y + Z | **Octree** (3-D) | up to 8 |

If no Z dimension is found, a 2-D quadtree is written.  The detected `spatial_dims` list is stored in `metadata.msgpack` and governs all subsequent reads — callers of `read_gridtiles` do not need to specify the tree type themselves.

If X or Y cannot be determined, a `ValueError` is raised with a message listing the available dimensions.
