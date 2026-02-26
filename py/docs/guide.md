# Gridtiles User Guide

## Installation

```bash
pip install gridtiles           # from PyPI (when published)
pip install -e /path/to/gridtiles  # editable install from source
```

Dependencies: `xarray`, `numpy`, `msgpack`, `msgpack-numpy`, `scipy`.
Optional: `pyproj` (for coordinate projection helpers).

---

## Quick start

```python
import gridtiles
import xarray as xr

# ── Write ─────────────────────────────────────────────────────────────────────
gridtiles.write_gridtiles(ds, "tiles/")

# ── Read (full resolution) ────────────────────────────────────────────────────
ds2 = gridtiles.read_gridtiles("tiles/")

# ── Read via xarray engine ────────────────────────────────────────────────────
ds2 = xr.open_dataset("tiles/", engine="gridtiles")
```

---

## Writing a dataset

### 2-D vs 3-D: quadtree and octree

The number of spatial dimensions determines the tree structure used:

- **2 spatial dims** → **quadtree** (up to 4 children per node, child indices 0–3)
- **3 spatial dims** → **octree** (up to 8 children per node, child indices 0–7)

This is decided automatically from the dataset when `spatial_dims` is not provided, or is set explicitly by the length of `spatial_dims`.  Everything else — tile naming, `bounds`, `shape`, `spatial_coords` — follows from this choice.

### Automatic dimension detection

The library reads CF `axis` and `standard_name` attributes to determine which dimensions are spatial.  If only X and Y are found, a quadtree is written; if X, Y, and Z are all found, an octree is written.

**3-D dataset (octree):**

```python
ds = xr.Dataset(
    {"resistivity": (["x", "y", "z"], data)},
    coords={
        "x": ("x", x_arr, {"axis": "X", "units": "m"}),
        "y": ("y", y_arr, {"axis": "Y", "units": "m"}),
        "z": ("z", z_arr, {"axis": "Z", "units": "m", "positive": "down"}),
    },
    attrs={"Conventions": "CF-1.8"},
)

gridtiles.write_gridtiles(ds, "tiles/")   # octree, 8 children per node
```

**2-D dataset (quadtree):**

```python
ds = xr.Dataset(
    {"mag": (["lat", "lon"], data)},
    coords={
        "lat": ("lat", lat_arr, {"axis": "Y", "units": "degrees_north"}),
        "lon": ("lon", lon_arr, {"axis": "X", "units": "degrees_east"}),
    },
    attrs={"Conventions": "CF-1.8"},
)

gridtiles.write_gridtiles(ds, "tiles/")   # quadtree, 4 children per node
```

### Explicit spatial dimensions

When dimension names or attributes are ambiguous, pass `spatial_dims` explicitly.  Its length controls the tree type:

```python
# 3-D → octree
gridtiles.write_gridtiles(ds, "tiles/", spatial_dims=["easting", "northing", "depth"])

# 2-D → quadtree
gridtiles.write_gridtiles(ds, "tiles/", spatial_dims=["easting", "northing"])
```

### Tuning tile size

`max_leaf` controls the granularity of the spatial tree.  Smaller values create more tiles (better spatial filtering, more files); larger values create fewer, bigger tiles.

```python
# finer tiles – good for large datasets with selective queries
gridtiles.write_gridtiles(ds, "tiles/", max_leaf=16)

# coarser tiles – fewer files, faster for whole-dataset reads
gridtiles.write_gridtiles(ds, "tiles/", max_leaf=64)
```

Rule of thumb: choose `max_leaf` so that each leaf tile fits comfortably in memory for a single frontend render call (e.g. 32–128 points per axis).  The same guidance applies for both quadtree and octree datasets.

### Attaching CRS information

The `crs` and `z_crs` parameters are stored in `metadata.msgpack` for consumers that need to reproject coordinates.  They do not affect how data is stored.

```python
gridtiles.write_gridtiles(
    ds, "tiles/",
    crs="EPSG:32632",    # UTM zone 32N
    z_crs="EPSG:5773",   # EGM96 geoid
)
```

### xarray accessor

```python
ds.gridtiles.to_gridtiles("tiles/", max_leaf=32, crs="EPSG:3857")
```

---

## Reading a dataset

### Full resolution (default)

```python
ds = gridtiles.read_gridtiles("tiles/")
# or
ds = xr.open_dataset("tiles/", engine="gridtiles")
```

All leaf tiles are loaded and merged.  The returned dataset is identical to the one that was written (same coordinates, dtypes, and attributes).

### Level-of-detail reads

```python
ds_lo = gridtiles.read_gridtiles("tiles/", level=0)   # root tile only
ds_l1 = gridtiles.read_gridtiles("tiles/", level=1)   # one level below root
ds_l2 = gridtiles.read_gridtiles("tiles/", level=2)   # two levels below root
```

Each level halves the number of grid points along each spatial axis relative to the level above.  The same mechanic applies to both quadtree and octree datasets; only the number of axes that are halved differs.

**3-D octree** — spatial size 256×256×128 at full resolution:

| level | approximate shape |
|-------|------------------|
| None (leaves) | 256 × 256 × 128 |
| 3 | 32 × 32 × 16 |
| 2 | 16 × 16 × 8 |
| 1 | 8 × 8 × 4 |
| 0 | 4 × 4 × 2 |

**2-D quadtree** — spatial size 1024×512 at full resolution:

| level | approximate shape |
|-------|------------------|
| None (leaves) | 1024 × 512 |
| 4 | 64 × 32 |
| 3 | 32 × 16 |
| 2 | 16 × 8 |
| 1 | 8 × 4 |
| 0 | 4 × 2 |

When a branch of the tree terminates (becomes a leaf) before the requested level is reached, that leaf tile is returned as-is for that spatial region.  Coverage is always complete.

### Bounding-box filter

The bbox format mirrors the number of spatial dimensions:

- **2-D** (quadtree): `[x_min, y_min, x_max, y_max]`
- **3-D** (octree): `[x_min, y_min, z_min, x_max, y_max, z_max]`

```python
# 2-D bbox
ds_sub = gridtiles.read_gridtiles(
    "tiles/",
    bbox=[500_000, 6_200_000, 520_000, 6_220_000],
)

# 3-D bbox
ds_sub = gridtiles.read_gridtiles(
    "tiles/",
    bbox=[500_000, 6_200_000, -500, 520_000, 6_220_000, 0],
)

# Combined with level-of-detail (works the same for 2-D and 3-D)
ds_sub_lo = gridtiles.read_gridtiles(
    "tiles/",
    level=2,
    bbox=[500_000, 6_200_000, -500, 520_000, 6_220_000, 0],
)
```

Tiles that do not intersect the bbox are skipped during traversal.  If all tiles for a region are filtered out, a coarser ancestor tile is returned instead so that coverage within the bbox is never empty.

---

## Non-spatial dimensions

Dimensions that are not part of the spatial decomposition (e.g. time, ensemble member) are handled transparently by both quadtree and octree datasets:

- Their coordinate values are stored once in `metadata.msgpack`.
- Each tile stores the **full extent** of these dimensions for its spatial chunk.
- On read they are restored exactly, including dtype and attributes.

```python
# 3-D octree with a time dimension
ds = xr.Dataset(
    {"temperature": (["time", "x", "y", "z"], data)},  # time is non-spatial
    coords={
        "time": ("time", times, {"units": "days since 2000-01-01", "axis": "T"}),
        "x": ..., "y": ..., "z": ...,
    },
)
gridtiles.write_gridtiles(ds, "tiles/", spatial_dims=["x", "y", "z"])

# 2-D quadtree with a time dimension
ds = xr.Dataset(
    {"mag": (["time", "lat", "lon"], data)},            # time is non-spatial
    coords={
        "time": ("time", times, {"units": "days since 2000-01-01", "axis": "T"}),
        "lat": ..., "lon": ...,
    },
)
gridtiles.write_gridtiles(ds, "tiles/", spatial_dims=["lat", "lon"])

ds2 = gridtiles.read_gridtiles("tiles/")
# ds2.time values, dtype, and attrs are identical to the original
```

---

## Roundtrip fidelity

| Property | Preserved? |
|----------|-----------|
| Spatial coordinate values | Exact (leaf tiles store the unmodified arrays) |
| Non-spatial coordinate values | Exact (stored in metadata) |
| Data variable values | Exact at full resolution (stored as `float32`) |
| Data variable dtype | Exact (cast back from `float32` on read) |
| Variable attributes | Exact |
| Coordinate attributes | Exact |
| Global dataset attributes | Exact |
| Internal node data values | Approximate — bilinear average of children (see [format spec](format.md#internal-node-data-values)) |

The only loss is that internal (non-leaf) tile data is a downsampled approximation; this is intentional and is what enables level-of-detail reads.
