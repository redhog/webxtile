# WebXTile Format Analysis & Scalability Findings

## Format Specification

### Container Layout

Each dataset is a directory of msgpack files:

```
dataset/
  metadata.msgpack       # dataset-level metadata (CRS, dims, variable schemas)
  root.msgpack           # root tile (full dataset at lowest resolution)
  root_0.msgpack         # child 0 of root
  root_0_1.msgpack       # child 1 of root_0
  root_0_1_3.msgpack     # ...and so on
```

### Tree Structure

| Dimensionality | Tree type | Branching factor |
|----------------|-----------|-----------------|
| 2D             | Quadtree  | 4               |
| 3D             | Octree    | 8               |

Splitting happens at the **index midpoint** of each spatial axis (not the coordinate midpoint). Recursion terminates when the largest spatial axis ≤ `max_leaf` (default: 32 points).

Child index layout (2D quadtree):
```
0: x[0:xm],  y[0:ym]     1: x[0:xm],  y[ym:ny]
2: x[xm:nx], y[0:ym]     3: x[xm:nx], y[ym:ny]
```

Empty children (zero points on any axis) are omitted.

### Tile Structure (each `.msgpack` file)

| Field           | Type                    | Description                                      |
|-----------------|-------------------------|--------------------------------------------------|
| `level`         | int                     | Tree depth (0 = root)                            |
| `is_leaf`       | bool                    | Whether this is a leaf node                      |
| `bounds`        | float[6]                | AABB: `[x_min, y_min, z_min, x_max, y_max, z_max]` |
| `shape`         | int[]                   | Grid dimensions in this tile                     |
| `spatial_coords`| map[str → float64[]]    | 1D coordinate arrays per spatial axis            |
| `variables`     | map[str → float32[]]    | Data variables (flat, row-major)                 |
| `children`      | str[]                   | Relative filenames of child tiles (internal only) |

### Level-of-Detail (LOD)

Internal nodes store **bilinearly/trilinearly downsampled** data at 0.5× resolution via `scipy.ndimage.zoom(arr, 0.5, order=1)`. Leaf nodes store exact original values as float32. Reading all leaves and merging gives exact full-resolution reconstruction.

### Metadata File

`metadata.msgpack` contains: format version, spatial dimension names, CRS (EPSG), per-coordinate metadata (with embedded values for non-spatial dims like time), per-variable metadata (dims, dtype, attrs), and global CF attributes.

---

## Browser Limitations

### 1. File Count Explosion

The number of tiles scales as O(branching_factor^depth / (branching_factor - 1)):

| Grid size       | Tree | `max_leaf` | Depth | Tile count  |
|-----------------|------|-----------|-------|-------------|
| 256 × 256       | Quad | 32        | 3     | ~85         |
| 4096 × 4096     | Quad | 32        | 7     | ~21,000     |
| 16384 × 16384   | Quad | 32        | 9     | ~340,000    |
| 512 × 512 × 512 | Oct  | 32        | 4     | ~4,700      |
| 2048×2048×128   | Oct  | 32        | 6     | ~300,000    |

At hundreds of thousands of tiles: filesystem inodes become a practical constraint, and the total HTTP overhead (even with HTTP/2 multiplexing) adds significant latency.

### 2. No HTTP Range Requests

Each tile is an atomic msgpack blob fetched in full. There is no sub-tile byte-range access, so you cannot partially stream a tile or compose a larger logical tile from partial reads. Every fetch is all-or-nothing.

### 3. Memory Pressure from `toScatter()`

`toScatter()` expands each tile's compact 1D coordinate arrays into a full meshgrid before returning data. For a tile with shape [N, M] and K variables:

```
memory = N × M × (2 coords + K vars) × 4 bytes
```

A batch of 100 tiles each 64×64 with 5 variables = ~80 MB peak, before any rendering buffer. Deep LOD queries fetching many tiles at once can easily exceed 500 MB in the browser JS heap.

### 4. Non-Spatial Dimensions Truncated in `toScatter()`

`toScatter()` samples non-spatial axes (time, depth index, ensemble member, etc.) at **index 0 only**. Full ND access requires working directly with raw tile objects, which is significantly more complex on the consumer side.

### 5. HTTP Concurrency

The library enforces a 16-fetch semaphore to avoid browser connection exhaustion.

`loadBBox()` for level > 0 requires only **1 sequential round-trip**: the batch fetch of all target-level candidates. Candidate filenames are computed directly from the predictable `root_i_j_k.msgpack` naming scheme and the global bounds stored in `metadata.msgpack` (no root tile fetch needed). Parent fallback on 404s adds at most one additional sequential hop per 404 cluster. Level 0 still fetches the root tile (it is the only tile at that depth).

The coordinate-midpoint approximation can produce false positives (candidates that don't exist → 404 → parent fallback, one extra sequential hop per 404 cluster) when coordinate spacing is non-uniform. For regular grids this is rare.

`streamLeaves()` (full BFS traversal, explicitly background-only) **does** require O(depth) sequential round-trips because it reads child filenames from each tile's `children` field and processes level by level.

### 6. IndexedDB Cache — No Eviction

Tiles are persisted to IndexedDB as raw msgpack bytes. There is no LRU eviction policy. For large or frequently-updated datasets the cache grows without bound until the browser enforces its per-origin quota (typically 10–50 GB shared with all sites).

### 7. Coordinate Precision

Spatial coordinates are stored as float64 in tiles but data variables are float32. The reconstruction path uses a Python float (float64) as dict key — adequate for most geodata, but may silently lose precision if source data uses >7 significant digits in a variable.

---

## Suggested Solutions

### S0 — Increase `max_leaf` (zero code changes)

**`max_leaf` unit**: points along the **longest single spatial axis**. A leaf tile can be up to `max_leaf^ndim` total points — `max_leaf×max_leaf` for 2D, `max_leaf×max_leaf×max_leaf` for 3D.

**Problem**: the default `max_leaf=32` produces very fine-grained tiles (32³ = ~32K points each) suited for general-purpose use but not browser rendering, where round-trip latency dominates over payload size.

**Effect of increasing `max_leaf`**:

| `max_leaf` | 2D tile (max pts) | 3D tile (max pts) | Depth (4096² grid) | Depth (256³ grid) | Tiles (4096² grid) |
|-----------|-------------------|-------------------|-------------------|-------------------|--------------------|
| 32        | 1K                | 32K               | 7                 | 3                 | ~21,000            |
| 128       | 16K               | 2M                | 5                 | 1                 | ~1,300             |
| 256       | 65K               | **16M**           | 4                 | 0 (single tile)   | ~340               |
| 4096      | **16M**           | —                 | 1                 | —                 | ~4                 |

**Recommended values** to target ~16M points per leaf tile (a reasonable GPU buffer size):
- **3D datasets**: `max_leaf=256` — 256³ = 16.7M points per tile
- **2D datasets**: `max_leaf=4096` — 4096² = 16.8M points per tile

**Tradeoffs**:
- Fewer, larger tiles → fewer HTTP requests for any given query. Direct win.
- Each tile transfers more bytes — on slow connections a single large tile may be slower than several small parallel ones.
- LOD steps become coarser — at `max_leaf=4096` a 2D dataset has only 1–2 levels, so there is no smooth progressive load; the jump from overview to full resolution is large.
- Less spatial selectivity — a zoomed-in query still fetches the full tile even if only a corner is visible.

**When it works well**: geophysics workflows where users typically view the full dataset before zooming in, and where datasets fit within a few hundred MB. The LOD loss is acceptable if the whole dataset is usually needed anyway.

**When it does not help**: very large datasets (multi-GB) where spatial selectivity is critical, or 3D datasets with a non-square aspect ratio where one axis is much larger than the others.

---

### S1 — Directory Sharding for Large Tile Counts

**Problem**: hundreds of thousands of loose files in one directory.

**Solution**: shard tile files into a two-level directory tree using the first 2–4 characters of the tile name:

```
root_0_1_3.msgpack  →  tiles/ro/root_0_1_3.msgpack
```

This keeps any single directory under ~10K entries regardless of total tile count, which is safe for all major filesystems and S3-compatible object stores. Requires only a path-resolution change in the reader and writer.

---

### S2 — Single-File Archive with Byte-Offset Index

**Problem**: hundreds of thousands of loose files create operational overhead — object stores (S3, MinIO) charge per-object, have per-object metadata overhead, and become slow to list/sync at large scale. Deployment (copying, uploading, syncing) is dramatically slower with 300K files than with 1.

**What this does NOT fix**: the number of HTTP requests from the browser is unchanged — one fetch per tile, same as before. Latency and bandwidth are unaffected.

**Solution**: at write time, concatenate all tile msgpack blobs into a single binary archive file and write a separate index (`index.msgpack`) mapping tile name → `[byte_offset, byte_length]`. The JS reader fetches the index once, then issues `Range: bytes=X-Y` HTTP requests to retrieve individual tiles.

```js
// current
const response = await fetch(`${base}/${tileName}`);

// with single-file archive
const [offset, length] = index[tileName];
const response = await fetch(archiveUrl, {
    headers: { Range: `bytes=${offset}-${offset + length - 1}` }
});
```

The response body is raw msgpack bytes — identical to what the current decoder receives. No changes to the parsing layer. The only new JS code is a one-time index fetch and a header lookup per tile.

**Actual wins**: server-side only — fewer inodes, cheaper object storage, faster deployment and sync. Not a browser performance improvement.

---

### S3 — Subtree Bundling (latency reduction for `streamLeaves()`)

**Problem**: `streamLeaves()` (full BFS traversal used to progressively load the entire dataset) requires O(depth) sequential round-trips because it follows the `children` array in each tile to discover the next level.

`loadBBox()` is **not affected** by this problem: since `metadata.msgpack` now carries global bounds, candidate filenames at the target level are computed from the naming scheme alone and fetched in a single parallel batch (one round-trip for level > 0). S3 therefore provides no latency benefit for `loadBBox()` on regular grids. On irregular grids where coordinate midpoints diverge significantly from index midpoints, S3 would eliminate the occasional 404-and-parent-fallback hops, but these are minor compared to the main tile fetch.

**Solution**: bundle a root tile together with all its descendants into a single file. One HTTP request returns the entire subtree. This is the approach PMTiles uses for map vector tiles.

- For `streamLeaves()`: reduces sequential round-trips from O(depth) to O(1) — the entire tree arrives in one or a few bundles.
- For `loadBBox()`: no meaningful benefit on regular grids; marginal reduction in 404 overhead on irregular grids.

This requires a more significant redesign of both the writer (grouping tiles into bundles) and the reader (unpacking a bundle into the tile cache in one pass), but the msgpack format per tile remains unchanged.

---

### S4 — Lazy / Streaming `toScatter()` with Configurable Axes

**Problem**: `toScatter()` eagerly expands all tiles and only exposes index-0 for non-spatial dims.

**Solution**:
- Accept a `dims` argument specifying which non-spatial indices to include (or a slice).
- Return a lazy iterator instead of allocating one large buffer — let the caller page through tiles.
- For rendering use cases, expose a `toFlatArrays()` method that returns `{x, y, z, var1, var2, ...}` typed arrays without intermediate meshgrid allocation, reducing peak memory by ~50%.

---

### S5 — LRU Eviction for IndexedDB Cache

**Problem**: IndexedDB grows without bound.

**Solution**: maintain a small metadata table alongside the tile store with `{key, last_accessed, byte_size}`. On each write, check total cached bytes against a configurable limit (e.g. 500 MB) and evict the least-recently-used tiles until under the limit. This is a ~50-line addition to the JS cache layer.

---

### S6 — Parallel Tree Construction in the Python Writer

**Problem**: recursive tree building is single-threaded; large datasets are slow to write.

**Solution**: the subtrees rooted at each child of the root node are completely independent. Use `concurrent.futures.ProcessPoolExecutor` to write subtrees in parallel. For an octree this is up to 8× faster on multi-core machines, with no change to the output format.

---

### S7 — Configurable Downsampling Strategy

**Problem**: `scipy.ndimage.zoom(order=1)` (bilinear) averages values — appropriate for continuous fields but wrong for categorical data (land-use class, lithology code) or intensive quantities (count, flag).

**Solution**: accept a `resample` argument per variable in the writer:
- `"mean"` (default, current behavior)
- `"max"` / `"min"` — for envelope/extrema LOD
- `"nearest"` — for categorical variables
- `"sum"` — for counts/densities that must be conserved across scales

---

### Priority Recommendation

For the Nagelfluh use case (geophysics grids, browser-based visualization):

1. **S0** (increase `max_leaf`) — zero code changes; tune at write time. Use `max_leaf=256` for 3D, `max_leaf=4096` for 2D to target ~16M points per tile.
2. **S3** (subtree bundling) — primarily benefits `streamLeaves()` (O(depth) → O(1) round-trips); no meaningful gain for `loadBBox()` on regular grids. Lower priority now that `loadBBox()` already achieves one round-trip.
3. **S5** (IndexedDB LRU) — low effort, prevents silent quota exhaustion.
4. **S4** (streaming `toScatter()`) — important once datasets grow beyond single-session memory.
5. **S2** (single-file archive) — operational win for large deployments; no browser benefit.
6. **S1** (directory sharding) — quick defensive fix if S2 is deferred.
