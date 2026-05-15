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

Splitting happens at the **index midpoint** of each spatial axis (not the coordinate midpoint). Two independent termination conditions apply:

- **Per-axis**: axis `a` is split only if its physical coordinate span exceeds the **globally minimum physical span across all axes at the root** (a fixed threshold, computed once at write time). Axes at or below this threshold are never split.
- **Global leaf**: node becomes a leaf when `max(all axis sizes) ≤ max_leaf`.

The branching factor emerges from how many axes exceed the root's minimum physical span:

| Axes being split | Branching | Effective tree type |
|-----------------|-----------|---------------------|
| 3               | 8         | Octree              |
| 2               | 4         | Quadtree            |
| 1               | 2         | Binary tree         |
| 0               | —         | Leaf                |

**Why a fixed root threshold, not a per-node comparison**: comparing each axis to the *current node's* minimum changes as you descend — the minimum span shifts as large axes get halved — producing octree splits at the top levels and quadtree only deep in the tree. For geophysics this is backwards. The root's minimum physical span is a stable global property of the dataset geometry and gives correct, consistent results throughout the tree.

**Why physical extent, not grid point count**: a typical AEM dataset might be 50 km × 100 km × 500 m. At 10 m vertical spacing: 500×1000×50 grid points (z is min in grid counts — happens to work). At 1 cm vertical spacing: 500×1000×50,000 grid points — now z is the *maximum* in grid counts, so a grid-count rule splits z first at every level. The physical-extent rule sees 500 m depth vs 50–100 km horizontal in both cases and correctly suppresses z-splits regardless of vertical resolution.

**Result for geophysics**: the physical z extent (hundreds of metres) is always far smaller than horizontal extent (tens of kilometres), so z is never split — the tree is a **pure quadtree throughout** — tiles cover the full depth column at every level of detail, which is exactly what plan-view and cross-section rendering require.

Child index layout (example for quadtree node):
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

### S3 — Subtree Bundling (latency reduction + rendering/network decoupling)

**Problems addressed**:

1. `streamLeaves()` requires O(depth) sequential round-trips because it follows `children` arrays level by level.
2. Network-optimal tile size (~16M points, ~200 MB) and GPU-optimal tile size (tens to hundreds of thousands of points per draw call) are in direct conflict. 200 MB fits comfortably in browser RAM but not in GPU memory; the GPU needs many small upload chunks, but each HTTP request carries significant overhead.

`loadBBox()` is **not affected** by the latency problem: candidate filenames are computed from the naming scheme and fetched in a single parallel batch. S3 provides no latency benefit for `loadBBox()` on regular grids.

**Solution**: bundle a subtree into a single msgpack file with two levels baked in at write time:

```
{
  "bounds": [...],          # bundle-level AABB
  "level": 0,               # tree depth of bundle root
  "sub_tiles": [            # GPU-sized rendering chunks, pre-sorted
    { "bounds": [...], "shape": [...], "spatial_coords": {...}, "variables": {...} },
    { "bounds": [...], "shape": [...], "spatial_coords": {...}, "variables": {...} },
    ...
  ]
}
```

- **Outer level** (network granularity): one HTTP fetch per bundle, ~200 MB. One round-trip for `streamLeaves()` instead of O(depth).
- **Inner level** (GPU granularity): pre-partitioned, pre-sorted rendering sub-tiles packed as msgpack array fields. No byte-offset index needed — the client decodes the full msgpack into RAM (~200 MB, fine), then streams sub-tiles to the GPU one at a time.

The spatial partitioning and sorting of sub-tiles is done **once at write time**, not per client fetch. This is efficient because write-once / read-many: the client pays zero CPU for partitioning, just iterates the array and uploads.

**Interaction with S0**: increasing `max_leaf` to reduce tile count is only safe because this bundle structure handles rendering granularity independently. Without S3, large `max_leaf` values would produce tiles that are too large to upload to the GPU in one buffer.

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

### S8 — Existence Manifest Pyramid for Holey Grids

**Problem**: datasets with non-uniform spatial coverage ("holey" grids) degrade `loadBBox()` in two ways:

1. **False-positive candidates**: the naming-scheme heuristic generates candidate filenames for regions where no data exists. Each miss becomes a 404, triggering a parent fallback — one additional sequential round-trip per 404 cluster. In a sparse dataset (e.g., survey lines covering 10% of the bounding box) most candidates miss, so nearly every `loadBBox()` call incurs several sequential fallback hops instead of a single parallel batch.

2. **Flat manifest is impractical at scale**: simply listing all existing tile names in `metadata.msgpack` solves the 404 problem but blows up metadata size — 340K tiles × ~25 bytes/name ≈ 8 MB. Fetching 8 MB upfront on every page load to discover which tiles exist defeats the purpose of a tile pyramid.

**When this does NOT matter**: dense, uniform grids (all candidate filenames resolve to real tiles, 404 rate near zero). The current speculative approach works well there.

**Solution**: a separate, shallow "existence manifest pyramid" — a second pyramid of small msgpack files where each file lists the real data tile names within its spatial bounds.

```
dataset/
  metadata.msgpack          # unchanged
  root.msgpack              # unchanged
  root_0_1_3.msgpack        # unchanged
  ...
  manifest/
    m_root.msgpack          # covers entire dataset
    m_root_0.msgpack        # covers quadrant 0
    m_root_0_0.msgpack      # covers quadrant 0,0
    m_root_0_1.msgpack      # ...
```

Each manifest tile is a small msgpack object:

```
{
  "bounds": [x_min, y_min, z_min, x_max, y_max, z_max],
  "tiles": ["root.msgpack", "root_0.msgpack", "root_0_0.msgpack", ...]
}
```

**Pyramid depth is fixed and shallow** (e.g., 2–3 levels), independent of the data pyramid depth. The manifest pyramid does not need to match the data pyramid's branching: it needs only enough levels to keep individual manifest tiles small. Example sizing at depth 3 for a 340K-tile dataset:

| Manifest depth | Manifest tiles | Data tiles per manifest tile | Manifest tile size |
|---------------|---------------|-----------------------------|--------------------|
| 2 levels      | ~21            | ~16,000                     | ~400 KB            |
| 3 levels      | ~85            | ~4,000                      | ~100 KB            |
| 4 levels      | ~341           | ~1,000                      | ~25 KB             |

Depth 3 is a reasonable default: 85 manifest tiles total, ~100 KB each — fetching 1–4 of them per `loadBBox()` query costs ~100–400 KB versus the current approach of many sequential fallback hops.

**Query flow with the manifest pyramid**:

1. Determine which manifest tiles cover the query bbox (typically 1–4 tiles at the right manifest level). Their names follow the same `m_root_i_j.msgpack` naming scheme, computable directly from bounds — no sequential traversal.
2. Fetch those manifest tiles in parallel (one round-trip).
3. Take the union of their `tiles` arrays, filter to tiles intersecting the query bbox and target zoom level.
4. Fetch exactly those tiles in parallel — no 404s, no fallback hops.

Total round-trips for `loadBBox()`: **2** (manifest batch + data batch) instead of 1 + O(fallback hops).

**Write-time cost**: one additional pass over the tile tree at write time, building manifest objects bottom-up. Negligible compared to data tile generation.

**Interaction with S2**: S2's `index.msgpack` already maps every tile name to a byte offset, so it implicitly lists all existing tiles. For small datasets, the client could use S2's flat index as a substitute — fetch the full index once, parse it, then filter by bbox. However, S2's index is not spatially indexed; for large sparse datasets the client must download and parse the entire index (potentially many MB) before filtering. The manifest pyramid is the spatially-indexed counterpart: pay only for the regions you query.

**Interaction with S3**: if S3 (subtree bundling) is in use, the manifest tiles list bundle names instead of individual tile names — entries are fewer and larger, keeping manifest tiles even smaller.

**When to implement**: only worthwhile for datasets where coverage is significantly non-uniform (survey-line data, coastal datasets, multi-epoch composites with gaps). Dense raster grids covering their full bounding box gain nothing; the current speculative naming scheme has near-zero 404 rate for those.

---

### Priority Recommendation

For the Nagelfluh use case (geophysics grids, browser-based visualization):

1. **S3** (subtree bundling with embedded GPU sub-tiles) — solves both `streamLeaves()` latency (O(depth) → O(1) round-trips) and the rendering/network tile size conflict. Pre-partitioned sub-tiles in the bundle eliminate client-side partitioning work. Required before S0 is safe to use at large `max_leaf` values.
2. **S0** (increase `max_leaf`) — tune at write time once S3 is in place. Use `max_leaf=256` for 3D, `max_leaf=4096` for 2D to target ~16M points per network bundle.
3. **Adaptive axis splitting** (writer change) — split axis `a` only if its physical span exceeds the globally minimum physical span at the root (fixed threshold, computed once). A per-node comparison (current min) produces octs at the top levels and quads only deep in the tree — the wrong order for geophysics. A grid-count comparison breaks entirely at fine vertical resolution (1 cm bins give z 50,000 points, making it the maximum, not minimum). The fixed physical threshold gives a pure quadtree for flat datasets regardless of discretization. No format or reader changes required.
4. **S5** (IndexedDB LRU) — low effort, prevents silent quota exhaustion.
5. **S4** (streaming `toScatter()`) — important once datasets grow beyond single-session memory.
6. **S8** (existence manifest pyramid) — implement when datasets have non-uniform spatial coverage (survey lines, coastal data, gapped composites). Eliminates 404 fallback hops for holey grids at the cost of one extra parallel round-trip per `loadBBox()` call.
7. **S2** (single-file archive) — operational win for large deployments; no browser benefit.
8. **S1** (directory sharding) — quick defensive fix if S2 is deferred.
