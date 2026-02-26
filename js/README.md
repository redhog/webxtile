# gridtiles (JS)

Browser client for the [gridtiles](../py) octree format.  Read-only; designed
for partial bbox loads and level-of-detail rendering in web applications.

The Python library writes a directory of msgpack tile files.  This library
reads those files over HTTP, caches them in IndexedDB, and returns flat typed
arrays suitable for WebGL / point-cloud rendering.

## Installation

```bash
npm install gridtiles
```

Or install directly from the local source tree:

```bash
npm install ./deps/gridtiles/js
```

## Quick start

```js
import { GridtilesLoader } from "gridtiles";

const loader = new GridtilesLoader("https://example.com/tiles");
await loader.open();   // fetches metadata.msgpack

// Full-resolution load for a 2-D bounding box
const result = await loader.loadBBox([500000, 6200000, 520000, 6220000]);

// Flat arrays for rendering
const { coords, variables, count } = result.toScatter();
// coords.x, coords.y  — Float32Array, one value per grid point
// variables.resistivity — Float32Array, same length as coords
```

## API

### `new GridtilesLoader(baseUrl, [options])`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseUrl` | `string` | — | Base URL of the tile directory (no trailing slash). |
| `options.dbName` | `string` | `"gridtiles-cache"` | IndexedDB database name.  Use a unique name per dataset when serving multiple datasets from the same origin. |

---

### `loader.open()` → `Promise<object>`

Fetches `metadata.msgpack` and opens the IndexedDB tile cache.  Must be
awaited before calling `loadBBox`.

Returns the decoded metadata object.

---

### `loader.meta`

The metadata object loaded by `open()`, or `null` before `open()` is called.
Contains `spatial_dims`, `crs`, `z_crs`, `dim_sizes`, `var_meta`,
`coord_meta`, and `global_attrs`.

---

### `loader.loadBBox(bbox, [options])` → `Promise<GridResult>`

Load all tiles that intersect `bbox` down to the requested depth.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `number[] \| null` | `null` | Spatial bounding box. `null` = no filter (load everything). |
| `options.level` | `number \| null` | `null` | Maximum octree depth. `null` = leaf tiles (full resolution). `0` = root tile only (coarsest overview). |

**`bbox` format**

- 2-D data: `[x_min, y_min, x_max, y_max]`
- 3-D data: `[x_min, y_min, z_min, x_max, y_max, z_max]`

Coordinates must be in the same CRS as the dataset (see `loader.meta.crs`).

**Level-of-detail**

Each level halves the spatial resolution relative to the level above.  When a
branch of the octree terminates before the requested level is reached, the
deepest available tile for that branch is returned so that spatial coverage is
always complete.

```js
// Coarsest overview (root tile only)
const lo = await loader.loadBBox(null, { level: 0 });

// Medium detail
const mid = await loader.loadBBox(bbox, { level: 2 });

// Full resolution (default)
const hi = await loader.loadBBox(bbox);
```

---

### `loader.clearCache()` → `Promise<void>`

Evicts all tiles from both the in-memory session cache and IndexedDB.  Useful
when the server-side data has been regenerated.

---

### `GridResult`

Returned by `loadBBox`.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `meta` | `object` | Full metadata from `metadata.msgpack`. |
| `tiles` | `object[]` | Raw decoded tile objects. |
| `spatialDims` | `string[]` | Spatial dimension names, e.g. `["x","y"]`. |
| `crs` | `string \| null` | Horizontal CRS string or null. |
| `zCrs` | `string \| null` | Vertical CRS string or null. |
| `varMeta` | `object` | Per-variable metadata (`dims`, `dtype`, `attrs`). |
| `coordMeta` | `object` | Per-coordinate metadata (may include `values` for non-spatial coords). |

#### `result.toScatter()` → `{ coords, variables, count }`

Expands all tile grids into flat parallel arrays suitable for scatter-plot or
point-cloud rendering.

For each tile the 1-D `spatial_coords` arrays are turned into a full meshgrid;
every data variable is sampled at each resulting grid point.

- `coords` — `Object<string, Float32Array>` — one array per spatial dimension,
  e.g. `coords.x`, `coords.y`, `coords.z`.
- `variables` — `Object<string, Float32Array>` — one array per data variable,
  same length as `coords`.
- `count` — `number` — number of grid points (length of each array).

Variables that have non-spatial dimensions (e.g. time) are sampled at index 0
of each non-spatial axis.  Access `result.tiles` directly for full control.

```js
const { coords, variables, count } = result.toScatter();

// WebGL example
const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, coords.x, gl.STATIC_DRAW);
```

#### `result.getCoord(dimName)` → `Float64Array`

Returns the merged, sorted, deduplicated coordinate values for one spatial
dimension across all loaded tiles.  Useful for reconstructing a regular grid
axis without going through `toScatter`.

```js
const xValues = result.getCoord("x");  // Float64Array, sorted ascending
```

## Caching

Tiles are fetched once and stored as raw msgpack bytes in IndexedDB.  On
subsequent loads (within the same session or across sessions) tiles are served
from the cache without a network request.

The in-memory session cache additionally avoids repeated IndexedDB reads within
a single page load.

## Tile format reference

Each tile file is a msgpack map with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `level` | `int` | Octree depth (0 = root). |
| `is_leaf` | `bool` | `true` for leaf nodes, `false` for internal nodes. |
| `bounds` | `number[6]` | `[x_min, y_min, z_min, x_max, y_max, z_max]` (always 6 elements; padded with `0.0` for 2-D data). |
| `shape` | `int[]` | Grid point count per spatial dimension for this tile. |
| `spatial_coords` | `object` | 1-D coordinate array per spatial dimension. |
| `variables` | `object` | Data variable arrays (float32) at this tile's resolution. |
| `children` | `string[]` | Child tile filenames (internal nodes only). |

Numpy arrays in the msgpack files are encoded by `msgpack-numpy` as plain maps
`{ nd: true, type: "<f4", shape: [...], data: <bytes> }` and are decoded
transparently to `Float32Array` / `Float64Array` by this library.

## Dependencies

| Package | Purpose |
|---------|---------|
| [`@msgpack/msgpack`](https://github.com/msgpack/msgpack-javascript) | msgpack decoding |
| [`idb`](https://github.com/jakearchibald/idb) | Promise-based IndexedDB wrapper |
