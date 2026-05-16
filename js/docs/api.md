# webxtile JS — API Reference

## Classes

- [`WebxtileLoader`](#webxtileloader) — fetches, caches, and traverses a webxtile dataset
- [`WebxtileResult`](#webxtileresult) — holds tiles returned by `loadBBox`

---

## `WebxtileLoader`

```js
import { WebxtileLoader } from "webxtile";

const loader = new WebxtileLoader("https://example.com/tiles");
await loader.open();
```

### Constructor

```js
new WebxtileLoader(baseUrl, [options])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseUrl` | `string` | — | Base URL of the tile directory. Trailing slash is stripped. |
| `options.dbName` | `string` | `"webxtile-cache"` | IndexedDB database name. Use a unique name per dataset when serving multiple datasets from the same origin to avoid key collisions. |

---

### `loader.open()` → `Promise<object>`

Fetches `metadata.msgpack` and opens the IndexedDB tile cache in parallel.
Must be awaited before calling `loadBBox` or `streamLeaves`.

Returns the decoded metadata object (same as `loader.meta`).

---

### `loader.meta`

The metadata object loaded by `open()`, or `null` before `open()` is called.

| Field | Type | Description |
|-------|------|-------------|
| `spatial_dims` | `string[]` | Spatial dimension names in `[X, Y]` or `[X, Y, Z]` order. Length determines tree type: 2 → quadtree, 3 → octree. |
| `root_tile` | `string` | Filename of the root tile (always `"root.msgpack"`). |
| `crs` | `string \| null` | Horizontal CRS identifier or null. |
| `z_crs` | `string \| null` | Vertical CRS identifier or null. |
| `dim_sizes` | `object` | Full-resolution size of every dimension. |
| `var_meta` | `object` | Per-variable metadata — see [VarMeta](#varmeta). |
| `coord_meta` | `object` | Per-coordinate metadata — see [CoordMeta](#coordmeta). |
| `global_attrs` | `object` | Dataset-level CF attributes. |

---

### `loader.loadBBox(bbox, level)` → `Promise<WebxtileResult>`

Load tiles that intersect `bbox` down to octree depth `level`.

```js
// Level 0: root tile only (coarsest overview)
const lo = await loader.loadBBox(null, 0);

// Level 2: medium detail within a bbox
const mid = await loader.loadBBox([500000, 6200000, 520000, 6220000], 2);

// Level 5: fine detail, no spatial filter
const hi = await loader.loadBBox(null, 5);
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `bbox` | `number[] \| null` | Spatial bounding box (see format below). `null` = no spatial filter. |
| `level` | `number` | Octree depth to descend to. Required — pass an explicit value. |

**`bbox` format**

- 2-D (quadtree): `[x_min, y_min, x_max, y_max]`
- 3-D (octree): `[x_min, y_min, z_min, x_max, y_max, z_max]`

Coordinates must be in the same CRS as the dataset (`loader.meta.crs`).

**Algorithm — direct fetch with parent fallback**

`loadBBox` fetches target-level tiles directly without traversing intermediate
levels, avoiding the O(branching_factor^1 + … + branching_factor^level) intermediate
fetches that a top-down BFS would require.

1. **Fetch root** — the root tile (`root.msgpack`) is fetched once to obtain
   its `bounds`.  If the root does not intersect `bbox`, return empty
   immediately.  If `level === 0`, return the root tile.

2. **Generate candidates** — enumerate filenames of all tiles at depth `level`
   whose *approximate* bounding box intersects `bbox`.  Bounds are computed
   by recursively halving each axis at its coordinate midpoint
   `(min + max) / 2`, which approximates the true index-midpoint split used by
   the writer.  For uniformly-spaced grids the approximation is exact.  For
   non-uniform grids a small number of extra candidates (false positives) may
   be generated; these produce 404 responses and are handled by the fallback.

3. **Batch fetch** — all candidates are fetched in parallel batches of 16.
   Responses are stored in a local map; a 404 is recorded as `null`.

4. **Parent fallback** — for each candidate that returned 404, strip the last
   `_N` child-index suffix from the filename to obtain the parent, and check
   whether a tile exists there.  Repeat upward until an existing tile is found
   (the root is always available as the ultimate fallback).  A result `Set`
   deduplicates ancestors that cover multiple missing siblings.

This means at most **two sequential network round trips** regardless of tree
depth: one for the root, one for the target-level batch.  Parent fallback adds
at most one additional sequential hop per cluster of missing tiles.

`loadBBox` holds a priority token (`_bboxActive` ref-count) for its entire
duration; `streamLeaves` will not start a new fetch batch while any
`loadBBox` call is in flight. See [Concurrency and priority](#concurrency-and-priority).

---

### `loader.streamLeaves([options])` → `AsyncGenerator<object>`

Async generator that yields every **leaf tile** in the dataset via a full BFS
traversal. Intended for background pre-loading of the complete dataset.

```js
const ac = new AbortController();

for await (const tile of loader.streamLeaves({ signal: ac.signal })) {
  myCache.add(tile);
}

// Cancel at any time:
ac.abort();
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `options.signal` | `AbortSignal` | Optional cancellation signal. The generator returns cleanly when aborted. |

**Algorithm — background BFS leaf traversal**

1. Start with the root tile on the frontier.
2. **Wait for bbox idle**: before starting each batch of up to 16 tiles, await
   `_waitBboxIdle()`.  This suspends the generator while any `loadBBox` call
   holds the priority token, freeing the fetch semaphore for interactive
   queries.  The check is repeated before every batch, not just at the start of
   each BFS level, so a `loadBBox` arriving mid-level is serviced promptly.
3. Fetch the batch in parallel.
4. For each tile:
   - If it is a leaf (`is_leaf == true` or no children), **yield** it.
   - Otherwise push its unseen children onto the next frontier.
5. A `visited` set prevents cycles if tile data is malformed.
6. Check the abort signal before each batch; return cleanly if aborted.

Unlike `loadBBox`, `streamLeaves` applies **no bbox filter** — every tile in
the tree is visited.

---

### `loader.clearCache()` → `Promise<void>`

Evicts all tiles from both the in-memory session cache and IndexedDB.
Call this when the server-side dataset has been regenerated.

---

## `WebxtileResult`

Returned by `loader.loadBBox`. Holds the collected tiles and the dataset
metadata.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `meta` | `object` | Full decoded `metadata.msgpack`. |
| `tiles` | `object[]` | Raw decoded tile objects in collection order. |
| `spatialDims` | `string[]` | Spatial dimension names, e.g. `["x","y"]`. |
| `crs` | `string \| null` | Horizontal CRS or null. |
| `zCrs` | `string \| null` | Vertical CRS or null. |
| `varMeta` | `object` | Per-variable metadata keyed by variable name. |
| `coordMeta` | `object` | Per-coordinate metadata keyed by coord name. |

---

### `result.getCoord(dimName)` → `Float64Array`

Returns the merged, sorted, deduplicated coordinate values for one spatial
dimension across all loaded tiles.

```js
const xValues = result.getCoord("x");  // Float64Array, sorted ascending
```

Useful for reconstructing a regular grid axis when you need the coordinate
positions but not the full scatter expansion.

---

## Concurrency and priority

### Fetch semaphore

A single global semaphore (shared across all `WebxtileLoader` instances) limits
in-flight HTTP requests to **16** at a time.  Without this cap, large bbox
queries or deep tree traversals would queue thousands of requests and exhaust
the browser's connection pool (`ERR_INSUFFICIENT_RESOURCES` in Chrome).

Each `fetch()` call acquires a slot before starting and releases it in
`finally`, so the cap holds even when requests fail or time out.

### HTTP timeout

Every `fetch()` is given a **60-second** timeout via `AbortController`.
A request that stalls longer than 60 s is aborted and throws
`Error: Timeout fetching <url>`.

### IDB semaphore

Each loader serialises its IndexedDB access with a per-instance semaphore
(limit 1).  This prevents `idb` transaction conflicts when multiple concurrent
`_loadTile` calls race to read or write the same store.

### Priority — `loadBBox` over `streamLeaves`

Interactive bbox queries must not be starved by a running `streamLeaves`
background traversal.  The mechanism:

- `loadBBox` increments `_bboxActive` on entry and decrements it in `finally`.
- `streamLeaves` calls `_waitBboxIdle()` before starting each batch of 16.
  `_waitBboxIdle()` returns immediately when `_bboxActive === 0`; otherwise it
  queues a callback that fires the moment `loadBBox` decrements the count back
  to zero.
- The check happens before every batch (not just every BFS level), so a
  `loadBBox` call that arrives while `streamLeaves` is partway through a level
  is serviced as soon as the current in-flight batch completes.

---

## Three-tier tile cache

Tiles are loaded through three layers, checked in order:

| Tier | Scope | Stored as |
|------|-------|-----------|
| In-memory `Map` | Session (page lifetime) | Decoded JS objects |
| IndexedDB | Persistent (across sessions) | Raw `Uint8Array` msgpack bytes |
| Network (`fetch`) | On miss | — |

After a network fetch the raw bytes are written to IndexedDB (fire-and-forget,
not awaited) and the decoded object is stored in the in-memory map.  On a cache
hit from IndexedDB the bytes are decoded and promoted to the in-memory map so
subsequent accesses within the session skip IndexedDB entirely.

---

## Metadata sub-types

### `VarMeta`

| Field | Type | Description |
|-------|------|-------------|
| `dims` | `string[]` | Ordered dimension names for this variable. |
| `dtype` | `string` | Original NumPy dtype string (e.g. `"float32"`). |
| `attrs` | `object` | CF attributes (`units`, `long_name`, `standard_name`, `_FillValue`, …). |

### `CoordMeta`

| Field | Type | Description |
|-------|------|-------------|
| `dims` | `string[]` | Dimension names this coordinate spans. |
| `dtype` | `string` | NumPy dtype string. |
| `attrs` | `object` | CF attributes (`units`, `standard_name`, `axis`, …). |
| `values` | typed array *(optional)* | Present only for non-spatial coordinates (e.g. time). Same value in every tile, so stored once in metadata. |

---

## See also

- [Format Specification](../../py/docs/format.md) — tile file layout, msgpack schema, LOD data model
- [Python API Reference](../../py/docs/api.md) — writing webxtile datasets
