# webxtile (JS)

Browser client for the [webxtile](../py) octree format.  Read-only; designed
for partial bbox loads and level-of-detail rendering in web applications.

The Python library writes a directory of msgpack tile files.  This library
reads those files over HTTP, caches them in IndexedDB, and returns flat typed
arrays suitable for WebGL / point-cloud rendering.

## Installation

```bash
npm install webxtile
```

Or install directly from the local source tree:

```bash
npm install ./deps/webxtile/js
```

## Documentation

[Full API reference with algorithms →](docs/api.md)

## Quick start

```js
import { WebxtileLoader } from "webxtile";

const loader = new WebxtileLoader("https://example.com/tiles");
await loader.open();   // fetches metadata.msgpack

// Load tiles at level 3 for a 2-D bounding box
const result = await loader.loadBBox([500000, 6200000, 520000, 6220000], 3);

// Flat arrays for WebGL / point-cloud rendering
const { coords, variables, count } = result.toScatter();
// coords.x, coords.y  — Float32Array, one value per grid point
// variables.resistivity — Float32Array, same length as coords

// Background: stream all leaf tiles (full resolution, whole dataset)
const ac = new AbortController();
for await (const tile of loader.streamLeaves({ signal: ac.signal })) {
  myTileCache.add(tile);
}
```

## API summary

| Method / property | Description |
|---|---|
| `new WebxtileLoader(baseUrl, [options])` | Create a loader for a tile directory. |
| `loader.open()` | Fetch `metadata.msgpack` and open IndexedDB cache. Must be awaited first. |
| `loader.meta` | Decoded metadata (`spatial_dims`, `crs`, `var_meta`, …). |
| `loader.loadBBox(bbox, level)` | Load tiles intersecting `bbox` at octree depth `level`. |
| `loader.streamLeaves([options])` | Async generator — yields all leaf tiles in BFS order. Pauses while `loadBBox` is in flight. |
| `loader.clearCache()` | Evict all tiles from in-memory cache and IndexedDB. |
| `result.toScatter()` | Expand tiles into flat `Float32Array` scatter arrays for rendering. |
| `result.getCoord(dim)` | Merged sorted coordinate values for one dimension across all tiles. |

See [docs/api.md](docs/api.md) for full parameter descriptions, algorithms, and the concurrency/priority model.

## Dependencies

| Package | Purpose |
|---------|---------|
| [`@msgpack/msgpack`](https://github.com/msgpack/msgpack-javascript) | msgpack decoding |
| [`idb`](https://github.com/jakearchibald/idb) | Promise-based IndexedDB wrapper |
