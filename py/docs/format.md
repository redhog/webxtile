# Gridtiles Format Specification

## Overview

A gridtiles dataset is a **directory** containing one `metadata.msgpack` file and one or more tile files (`*.msgpack`).  All files use [msgpack](https://msgpack.org/) encoding with [msgpack-numpy](https://github.com/lebedov/msgpack-numpy) extensions for numpy arrays.

The tree structure is determined by the number of spatial dimensions:

| Spatial dims | Tree type | Children per node | Child indices |
|:---:|---|:---:|---|
| 2 | **Quadtree** | up to 4 | `0` … `3` |
| 3 | **Octree** | up to 8 | `0` … `7` |

The number of spatial dimensions is fixed for the lifetime of a dataset and is recorded in `metadata.msgpack` as `spatial_dims`.  Everything else in the format — `bounds`, `shape`, `spatial_coords`, child indexing — follows directly from that count.

```
tiles/
├── metadata.msgpack      ← dataset-level metadata (CF attrs, coordinate values, …)
├── root.msgpack          ← root tile (level 0, lowest resolution)
├── root_0.msgpack        ← first child of root (level 1)
├── root_1.msgpack
├── …
├── root_0_0.msgpack      ← first child of root_0 (level 2)
├── root_0_1.msgpack
└── …
```

Tile naming is hierarchical: each child appends `_<i>` to the parent's stem, where `i` ∈ {0 … 3} for 2-D data (quadtree) or {0 … 7} for 3-D data (octree).  Children whose spatial slice would be empty (zero points on any axis) are omitted, so the actual number of children may be less than the maximum.

---

## `metadata.msgpack`

A single msgpack map with string keys:

| Key | Type | Description |
|-----|------|-------------|
| `version` | `int` | Format version (currently `1`). |
| `root_tile` | `str` | Filename of the root tile (always `"root.msgpack"`). |
| `spatial_dims` | `list[str]` | Names of the 2–3 dimensions used for spatial decomposition, in `[X, Y]` or `[X, Y, Z]` order.  **Length determines the tree type**: 2 → quadtree, 3 → octree. |
| `crs` | `str \| null` | Horizontal CRS identifier supplied by the writer, or `null`. |
| `z_crs` | `str \| null` | Vertical CRS identifier, or `null`. |
| `dim_sizes` | `map[str, int]` | Full-resolution size of every dimension in the dataset. |
| `coord_meta` | `map[str, CoordMeta]` | Per-coordinate metadata (see below). |
| `var_meta` | `map[str, VarMeta]` | Per-variable metadata (see below). |
| `global_attrs` | `map[str, any]` | Dataset-level CF attributes (`Conventions`, `title`, `institution`, …). |

### `CoordMeta`

| Key | Type | Description |
|-----|------|-------------|
| `dims` | `list[str]` | Dimension names this coordinate spans. |
| `dtype` | `str` | NumPy dtype string (e.g. `"float64"`). |
| `attrs` | `map[str, any]` | Coordinate attributes (`units`, `standard_name`, `axis`, …). |
| `values` | numpy array *(optional)* | Present **only** for non-spatial dimension coordinates (e.g. time). The array has the same length as the dimension and is the same in every tile, so it is stored once here rather than in each tile. |

### `VarMeta`

| Key | Type | Description |
|-----|------|-------------|
| `dims` | `list[str]` | Ordered dimension names for this variable. |
| `dtype` | `str` | Original NumPy dtype string. Variables are stored as `float32` inside tiles and cast back to this dtype on read. |
| `attrs` | `map[str, any]` | Variable attributes (`units`, `long_name`, `standard_name`, `_FillValue`, …). |

---

## Tile files

Each tile is a msgpack map:

| Key | Type | Leaf | Internal | Description |
|-----|------|:----:|:--------:|-------------|
| `level` | `int` | ✓ | ✓ | Octree depth (0 = root). |
| `is_leaf` | `bool` | `true` | `false` | Whether this is a leaf node. |
| `bounds` | `list[float]` (6 elements) | ✓ | ✓ | Axis-aligned bounding box in spatial coordinate units: `[x_min, y_min, z_min, x_max, y_max, z_max]`. **Always 6 elements** regardless of dimensionality — 2-D datasets pad the Z entries with `0.0`. Readers should use `len(spatial_dims)` from metadata to know how many entries are meaningful. |
| `shape` | `list[int]` | ✓ | ✓ | Number of grid points along each spatial dimension **in this tile** (full resolution for leaves, downsampled for internal nodes).  Length equals `len(spatial_dims)`: 2 elements for quadtree, 3 for octree. |
| `spatial_coords` | `map[str, float64 array]` | ✓ | ✓ | 1-D coordinate arrays keyed by spatial dimension name.  Has 2 entries for quadtree datasets, 3 for octree.  Leaf tiles contain the original coordinate values; internal tiles contain a sampled subset (see [LOD data model](#lod-data-model)). |
| `variables` | `map[str, float32 array]` | ✓ | ✓ | Data variable arrays at this tile's resolution. Shape matches `shape` for spatial dimensions; non-spatial dimensions are fully present (shape matches `dim_sizes`). |
| `children` | `list[str]` | — | ✓ | Filenames of child tiles (relative to the tile directory). Absent on leaf nodes. |

---

## LOD data model

### Spatial decomposition

At each internal node the dataset is split at the **midpoint index** of each spatial axis, producing up to 4 children (2-D quadtree) or 8 children (3-D octree).  Child `i` is assigned a contiguous rectangular sub-block of the parent's index space using the bit-pattern of `i` as a selector:

```
2-D children (quadtree):   xm = nx//2,  ym = ny//2
  0: x[0:xm],   y[0:ym]
  1: x[0:xm],   y[ym:ny]
  2: x[xm:nx],  y[0:ym]
  3: x[xm:nx],  y[ym:ny]

3-D children (octree):     xm = nx//2,  ym = ny//2,  zm = nz//2
  0: x[0:xm],   y[0:ym],   z[0:zm]
  1: x[0:xm],   y[0:ym],   z[zm:nz]
  2: x[0:xm],   y[ym:ny],  z[0:zm]
  3: x[0:xm],   y[ym:ny],  z[zm:nz]
  4: x[xm:nx],  y[0:ym],   z[0:zm]
  5: x[xm:nx],  y[0:ym],   z[zm:nz]
  6: x[xm:nx],  y[ym:ny],  z[0:zm]
  7: x[xm:nx],  y[ym:ny],  z[zm:nz]
```

In both cases the Z axis (bit 0) varies fastest and the X axis (most significant bit) varies slowest.  Children whose spatial slice is empty (zero points along any axis after the midpoint split) are skipped and do not appear in the parent's `children` list.

Recursion stops when the largest spatial dimension of a block is ≤ `max_leaf` points; that block becomes a leaf tile.

### Internal node data values — bilinear interpolation

Data values stored in internal (non-leaf) tiles are **bilinear / trilinear interpolations** of the full-resolution data at half the spatial resolution, computed with `scipy.ndimage.zoom(arr, 0.5, order=1)`.

Because each output point falls exactly at the midpoint between two adjacent input points along each axis, this is algebraically identical to **averaging adjacent pairs** — a 2-element moving average in 1-D, a 2×2 average in 2-D, or a 2×2×2 average in 3-D.  It is *not* nearest-neighbour sampling.

Concretely, for a 1-D array `[a, b, c, d, e, f, g, h]`, the half-resolution internal node stores `[(a+b)/2, (c+d)/2, (e+f)/2, (g+h)/2]`.

The spatial coordinate arrays in internal tiles are sampled from the original coordinate values (not interpolated), using `n//2` evenly spaced indices across the original range.  This means all coordinate values stored in the tree are a **subset of the original coordinates**, which makes exact index lookup possible during reconstruction.

### Reconstruction guarantee

Reading **all leaf tiles** and merging them gives back the original dataset without loss:

1. Each leaf tile stores the unmodified coordinate values and the original `float32` data.
2. The reader sorts all unique coordinate values from collected tiles into global arrays.
3. Each tile's data is placed at the exact positions in the output array determined by dict lookup on the coordinate values.
4. Non-spatial dimension coordinates (e.g. time) are restored from `metadata.msgpack`.

Reading at a specific `level` gives a lower-resolution grid assembled from whatever tiles exist at that depth (or the deepest available leaf for each branch).  The resulting dataset is a regular grid at that resolution; NaN fills any points not covered by the collected tiles.

---

## JS frontend compatibility

The `bounds` field is always 6 elements so that JavaScript readers can use a single fixed-size layout regardless of whether the dataset is 2-D or 3-D.  Readers determine how many `bounds` entries are meaningful by inspecting `len(spatial_dims)` from `metadata.msgpack`.

The companion JS client (`deps/gridtiles/js/gridtiles.js`) reads all fields described in this specification.  It uses `spatial_dims` from metadata to determine the tree type, reads `spatial_coords` for coordinate values, and uses `is_leaf` / `level` for LOD traversal.
