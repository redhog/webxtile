# WebXTile User Guide

## Installation

```bash
pip install webxtile           # from PyPI (when published)
pip install -e /path/to/webxtile  # editable install from source
```

Dependencies: `xarray`, `numpy`, `msgpack`, `msgpack-numpy`, `scipy`.
Optional: `pyproj` (for coordinate projection helpers).

---

## Quick start

```python
import webxtile
import xarray as xr

# ── Write ─────────────────────────────────────────────────────────────────────
webxtile.write_webxtile(ds, "tiles/")

# ── Read (full resolution) ────────────────────────────────────────────────────
ds2 = webxtile.read_webxtile("tiles/")

# ── Read via xarray engine ────────────────────────────────────────────────────
ds2 = xr.open_dataset("tiles/", engine="webxtile")
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

webxtile.write_webxtile(ds, "tiles/")   # octree, 8 children per node
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

webxtile.write_webxtile(ds, "tiles/")   # quadtree, 4 children per node
```

### Explicit spatial dimensions

When dimension names or attributes are ambiguous, pass `spatial_dims` explicitly.  Its length controls the tree type:

```python
# 3-D → octree
webxtile.write_webxtile(ds, "tiles/", spatial_dims=["easting", "northing", "depth"])

# 2-D → quadtree
webxtile.write_webxtile(ds, "tiles/", spatial_dims=["easting", "northing"])
```

### Tuning tile size

`max_leaf` controls the granularity of the spatial tree.  Smaller values create more tiles (better spatial filtering, more files); larger values create fewer, bigger tiles.

```python
# finer tiles – good for large datasets with selective queries
webxtile.write_webxtile(ds, "tiles/", max_leaf=16)

# coarser tiles – fewer files, faster for whole-dataset reads
webxtile.write_webxtile(ds, "tiles/", max_leaf=64)
```

Rule of thumb: choose `max_leaf` so that each leaf tile fits comfortably in memory for a single frontend render call (e.g. 32–128 points per axis).  The same guidance applies for both quadtree and octree datasets.

### Attaching CRS information

The `crs` and `z_crs` parameters are stored in `metadata.msgpack` for consumers that need to reproject coordinates.  They do not affect how data is stored.

```python
webxtile.write_webxtile(
    ds, "tiles/",
    crs="EPSG:32632",    # UTM zone 32N
    z_crs="EPSG:5773",   # EGM96 geoid
)
```

### xarray accessor

```python
ds.webxtile.to_webxtile("tiles/", max_leaf=32, crs="EPSG:3857")
```

---

## Reading a dataset

### Full resolution (default)

```python
ds = webxtile.read_webxtile("tiles/")
# or
ds = xr.open_dataset("tiles/", engine="webxtile")
```

All leaf tiles are loaded and merged.  The returned dataset is identical to the one that was written (same coordinates, dtypes, and attributes).

### Level-of-detail reads

```python
ds_lo = webxtile.read_webxtile("tiles/", level=0)   # root tile only
ds_l1 = webxtile.read_webxtile("tiles/", level=1)   # one level below root
ds_l2 = webxtile.read_webxtile("tiles/", level=2)   # two levels below root
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
ds_sub = webxtile.read_webxtile(
    "tiles/",
    bbox=[500_000, 6_200_000, 520_000, 6_220_000],
)

# 3-D bbox
ds_sub = webxtile.read_webxtile(
    "tiles/",
    bbox=[500_000, 6_200_000, -500, 520_000, 6_220_000, 0],
)

# Combined with level-of-detail (works the same for 2-D and 3-D)
ds_sub_lo = webxtile.read_webxtile(
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
webxtile.write_webxtile(ds, "tiles/", spatial_dims=["x", "y", "z"])

# 2-D quadtree with a time dimension
ds = xr.Dataset(
    {"mag": (["time", "lat", "lon"], data)},            # time is non-spatial
    coords={
        "time": ("time", times, {"units": "days since 2000-01-01", "axis": "T"}),
        "lat": ..., "lon": ...,
    },
)
webxtile.write_webxtile(ds, "tiles/", spatial_dims=["lat", "lon"])

ds2 = webxtile.read_webxtile("tiles/")
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

---

## Testing

The test suite lives under `py/tests/` and is split into two files:

| File | What it tests |
|------|---------------|
| `py/tests/test_roundtrip.py` | Pure Python: write → read roundtrips at full resolution, various bboxes, and multiple LOD levels (2-D and 3-D) |
| `py/tests/test_js.py` | Cross-language: Python writes tiles, Node.js reads them via `WebxtileResult`, both outputs are compared |

### Prerequisites

**Python tests** — `pytest` must be installed in the active environment:

```bash
pip install pytest
# or, if using the project virtualenv:
pip install -e ".[dev]"
```

**Cross-language tests** — Node.js (v18+) and the JS dependencies:

```bash
cd js/
npm install
```

### Running the tests

From the `py/` directory:

```bash
# Python roundtrip tests only
pytest tests/test_roundtrip.py -v

# Cross-language Python → JS tests only
pytest tests/test_js.py -v

# Everything
pytest tests/ -v
```

From the repository root (adjust the path to your virtualenv's `pytest`):

```bash
env/bin/pytest deps/webxtile/py/tests/ -v
```

### How the cross-language tests work

`test_js.py` calls `js/tests/read_tiles.mjs` as a subprocess via `node`.  That
script reads the msgpack tile files directly from the filesystem (no HTTP server
or IndexedDB required), feeds them into `WebxtileResult`, calls `toScatter()` /
`getCoord()`, and prints a JSON summary to stdout.  The Python test then
compares that JSON against the xarray Dataset returned by `read_webxtile()` for
the same query parameters.

Each cross-language test exercises a different combination of bbox and level:

- Full resolution (no filter)
- Three distinct bboxes: centre region, left strip, top-right quadrant
- Level 0, 1, and 2 (progressively coarser overviews)
- Bbox combined with level 1
- 3-D variants: full resolution, level 0, and a sub-volume bbox
