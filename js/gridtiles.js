/**
 * gridtiles.js
 * Browser client for the gridtiles octree format.
 * Read-only; designed for partial bbox loads for web visualisation.
 *
 * @example
 *   import { GridtilesLoader } from "gridtiles";
 *
 *   const loader = new GridtilesLoader("https://example.com/tiles");
 *   await loader.open();   // loads metadata.msgpack
 *
 *   // Full-resolution load for a 2-D bbox
 *   const result = await loader.loadBBox([x0, y0, x1, y1]);
 *   // Level-of-detail (0 = coarsest overview)
 *   const lo = await loader.loadBBox([x0, y0, x1, y1], { level: 2 });
 *
 *   // Flat scatter arrays for WebGL / point-cloud rendering
 *   const { coords, variables, count } = result.toScatter();
 *   // coords.x, coords.y — Float32Array, one value per grid point
 *   // variables.resistivity — Float32Array, same length
 */

import { decode } from "@msgpack/msgpack";
import { openDB } from "idb";

// ─── NumPy array decoding ─────────────────────────────────────────────────────
//
// msgpack_numpy encodes numpy arrays as plain msgpack maps:
//   { nd: true, type: '<f4', shape: [n, …], data: <bin> }
//
// The 'type' string follows NumPy's dtype.str convention:
//   byte order prefix: '<' (little-endian), '>' (big-endian), '|' (n/a)
//   kind+itemsize:     'u1','u2','u4','i1','i2','i4','f4','f8'

const _DTYPE_CTORS = {
  '|u1': Uint8Array,
  '<u2': Uint16Array,  '>u2': Uint16Array,
  '<u4': Uint32Array,  '>u4': Uint32Array,
  '|i1': Int8Array,
  '<i2': Int16Array,   '>i2': Int16Array,
  '<i4': Int32Array,   '>i4': Int32Array,
  '<f4': Float32Array, '>f4': Float32Array,
  '<f8': Float64Array, '>f8': Float64Array,
};

function _numpyToTyped(obj) {
  const Ctor = _DTYPE_CTORS[obj.type];
  if (!Ctor) throw new Error(`Unsupported numpy dtype: "${obj.type}"`);
  const src = obj.data instanceof Uint8Array ? obj.data : new Uint8Array(obj.data);
  // slice() copies to a new, correctly-aligned ArrayBuffer
  const buf = src.buffer.slice(src.byteOffset, src.byteOffset + src.byteLength);
  return new Ctor(buf);
}

function _decodeNumpy(v) {
  if (v === null || typeof v !== 'object') return v;
  if (v.nd === true && 'type' in v && 'data' in v) return _numpyToTyped(v);
  if (Array.isArray(v)) return v.map(_decodeNumpy);
  const out = {};
  for (const [k, val] of Object.entries(v)) out[k] = _decodeNumpy(val);
  return out;
}

// ─── Low-level fetch & decode ─────────────────────────────────────────────────

async function _fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  return new Uint8Array(await res.arrayBuffer());
}

function _decodeMsgpack(bytes) {
  return _decodeNumpy(decode(bytes));
}

// ─── Bounding-box intersection ────────────────────────────────────────────────

/**
 * Test whether a tile's bounds overlaps a query bbox.
 *
 * @param {number[]} tileBounds - always 6 elements [x0,y0,z0, x1,y1,z1]
 *   (padded with 0.0 for 2-D data so the layout is always consistent)
 * @param {number[]|null} bbox  - 4 or 6 elements [x0,y0,(z0,) x1,y1,(z1)]
 * @param {number} nSpatial     - number of spatial axes (2 or 3)
 */
function _intersects(tileBounds, bbox, nSpatial) {
  if (bbox === null) return true;
  for (let i = 0; i < nSpatial; i++) {
    // bbox layout: [min…, max…]  e.g. [xmin, ymin, xmax, ymax] for nSpatial=2
    if (bbox[i + nSpatial] < tileBounds[i])      return false; // bbox_max < tile_min
    if (bbox[i]             > tileBounds[i + 3]) return false; // bbox_min > tile_max
  }
  return true;
}

// ─── GridResult ───────────────────────────────────────────────────────────────

/**
 * Holds the tiles collected for a bbox / level query.
 *
 * The two main entry points for data consumption are:
 *   - `toScatter()` — flat parallel arrays per dimension and variable,
 *     suitable for point-cloud or scatter-plot style WebGL rendering.
 *   - `getCoord(dimName)` — merged sorted coordinate values for one dim.
 *   - `tiles` / `meta` — raw access for custom processing.
 */
export class GridResult {
  /**
   * @param {object}   meta  - decoded metadata.msgpack
   * @param {object[]} tiles - decoded tile objects
   */
  constructor(meta, tiles) {
    this._meta  = meta;
    this._tiles = tiles;
  }

  /** Full metadata object (version, spatial_dims, crs, dim_sizes, …). */
  get meta() { return this._meta; }

  /** Array of decoded tile objects as stored in the octree files. */
  get tiles() { return this._tiles; }

  /**
   * Spatial dimension names in writer order, e.g. `["x", "y"]` or
   * `["x", "y", "z"]`.
   * @type {string[]}
   */
  get spatialDims() { return this._meta.spatial_dims; }

  /** Horizontal CRS identifier string or null. */
  get crs()  { return this._meta.crs  ?? null; }

  /** Vertical CRS identifier string or null. */
  get zCrs() { return this._meta.z_crs ?? null; }

  /**
   * Per-variable metadata from metadata.msgpack.
   * Each entry: `{ dims: string[], dtype: string, attrs: object }`.
   * @type {Object<string, {dims: string[], dtype: string, attrs: object}>}
   */
  get varMeta() { return this._meta.var_meta ?? {}; }

  /**
   * Per-coordinate metadata from metadata.msgpack.
   * Each entry: `{ dims: string[], dtype: string, attrs: object, values?: TypedArray }`.
   * @type {Object<string, object>}
   */
  get coordMeta() { return this._meta.coord_meta ?? {}; }

  /**
   * Returns the merged, sorted, deduplicated coordinate values for one
   * spatial dimension across all loaded tiles.
   *
   * @param {string} dimName
   * @returns {Float64Array}
   */
  getCoord(dimName) {
    const seen = new Set();
    const vals = [];
    for (const tile of this._tiles) {
      const arr = tile.spatial_coords?.[dimName];
      if (!arr) continue;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (!seen.has(v)) { seen.add(v); vals.push(v); }
      }
    }
    vals.sort((a, b) => a - b);
    return new Float64Array(vals);
  }

  /**
   * Flatten all loaded tiles into parallel scatter arrays.
   *
   * For each tile the 1-D spatial coordinate arrays (`spatial_coords`) are
   * expanded into a full meshgrid and every data variable is read at each
   * resulting grid point.  The output arrays are all the same length
   * (`count`).
   *
   * Variables with non-spatial dimensions (e.g. time) are sampled at index 0
   * of every non-spatial axis.  For full control over non-spatial dimensions
   * use the raw `tiles` property.
   *
   * @returns {{ coords: Object<string,Float32Array>,
   *             variables: Object<string,Float32Array>,
   *             count: number }}
   *
   * @example
   *   const { coords, variables, count } = result.toScatter();
   *   gl.bufferData(gl.ARRAY_BUFFER, coords.x, gl.STATIC_DRAW);
   */
  toScatter() {
    const spatialDims = this._meta.spatial_dims;
    const nD = spatialDims.length;

    const cBufs = {};
    for (const d of spatialDims) cBufs[d] = [];
    const vBufs = {};

    for (const tile of this._tiles) {
      const sc = tile.spatial_coords ?? {};

      // 1-D coordinate arrays for this tile's spatial dims
      const dimArrs  = spatialDims.map(d => sc[d] ?? new Float64Array(0));
      const nPerDim  = dimArrs.map(a => a.length);
      const nTotal   = nPerDim.reduce((a, b) => a * b, 1);
      if (nTotal === 0) continue;

      // Row-major strides for the spatial meshgrid (dim 0 = outermost / slowest)
      const spStrides = new Array(nD);
      spStrides[nD - 1] = 1;
      for (let d = nD - 2; d >= 0; d--) spStrides[d] = spStrides[d + 1] * nPerDim[d + 1];

      // Fill coordinate buffers: for each flat spatial index decode per-dim idx
      for (let flat = 0; flat < nTotal; flat++) {
        for (let d = 0; d < nD; d++) {
          cBufs[spatialDims[d]].push(dimArrs[d][Math.floor(flat / spStrides[d]) % nPerDim[d]]);
        }
      }

      // Fill variable buffers
      for (const [varName, rawArr] of Object.entries(tile.variables ?? {})) {
        if (!(varName in vBufs)) vBufs[varName] = [];
        const vmeta    = this._meta.var_meta?.[varName];
        if (!vmeta) continue;

        const varDims  = vmeta.dims;
        // Map each variable dimension to its spatial axis index (or -1 if non-spatial)
        const spAxis   = varDims.map(d => spatialDims.indexOf(d));

        // Size of each variable dimension in this tile
        const varShape = varDims.map((d, vi) => {
          const si = spAxis[vi];
          return si >= 0 ? nPerDim[si] : (this._meta.dim_sizes?.[d] ?? 1);
        });

        // Row-major strides for rawArr in its own dimension order
        const varStrides = new Array(varDims.length);
        varStrides[varDims.length - 1] = 1;
        for (let d = varDims.length - 2; d >= 0; d--) {
          varStrides[d] = varStrides[d + 1] * varShape[d + 1];
        }

        for (let flat = 0; flat < nTotal; flat++) {
          // Decode per-spatial-dim indices for this meshgrid point
          const spIdxs = new Array(nD);
          for (let d = 0; d < nD; d++) {
            spIdxs[d] = Math.floor(flat / spStrides[d]) % nPerDim[d];
          }
          // Translate to a linear index into rawArr (non-spatial dims → 0)
          let vi = 0;
          for (let vd = 0; vd < varDims.length; vd++) {
            const si = spAxis[vd];
            vi += (si >= 0 ? spIdxs[si] : 0) * varStrides[vd];
          }
          vBufs[varName].push(rawArr[vi] ?? NaN);
        }
      }
    }

    const count = Object.values(cBufs)[0]?.length ?? 0;
    return {
      coords:    Object.fromEntries(Object.entries(cBufs).map(([k, v]) => [k, new Float32Array(v)])),
      variables: Object.fromEntries(Object.entries(vBufs).map(([k, v]) => [k, new Float32Array(v)])),
      count,
    };
  }
}

// ─── GridtilesLoader ──────────────────────────────────────────────────────────

/**
 * Loader for a gridtiles octree dataset served over HTTP.
 *
 * Tiles are persisted to IndexedDB after the first network fetch so that
 * repeated loads within a session (or across sessions) avoid redundant
 * requests.
 *
 * @example
 *   const loader = new GridtilesLoader("https://host/tiles");
 *   await loader.open();
 *
 *   // Load leaves inside a 2-D bbox (full resolution)
 *   const r = await loader.loadBBox([500000, 6200000, 520000, 6220000]);
 *
 *   // Coarse overview (level 2)
 *   const lo = await loader.loadBBox(null, { level: 2 });
 */
export class GridtilesLoader {
  /**
   * @param {string} baseUrl    - Base URL of the tile directory (trailing
   *   slash optional).
   * @param {object} [options]
   * @param {string} [options.dbName="gridtiles-cache"] - IndexedDB database
   *   name.  Use a unique name per dataset if you serve multiple datasets from
   *   the same origin.
   */
  constructor(baseUrl, { dbName = 'gridtiles-cache' } = {}) {
    this._base       = baseUrl.replace(/\/$/, '');
    this._dbName     = dbName;
    this._meta       = null;   // set by open()
    this._db         = null;   // IDBDatabase, set by open()
    this._memCache   = new Map(); // filename → decoded tile (session-level)
  }

  // ── Initialisation ──────────────────────────────────────────────────────────

  /**
   * Load `metadata.msgpack` and open the IndexedDB tile cache.
   * Must be awaited before calling `loadBBox`.
   *
   * @returns {Promise<object>} Decoded metadata object.
   */
  async open() {
    const [meta, db] = await Promise.all([
      this._fetchAndDecode('metadata.msgpack'),
      openDB(this._dbName, 1, {
        upgrade(db) {
          db.createObjectStore('tiles');
        },
      }),
    ]);
    this._meta = meta;
    this._db   = db;
    return meta;
  }

  /**
   * Metadata loaded from `metadata.msgpack`.
   * `null` until `open()` resolves.
   * @type {object|null}
   */
  get meta() { return this._meta; }

  // ── Tile fetch and cache ────────────────────────────────────────────────────

  async _fetchAndDecode(filename) {
    const bytes = await _fetchBytes(`${this._base}/${filename}`);
    return _decodeMsgpack(bytes);
  }

  async _loadTile(filename) {
    // 1. In-memory session cache
    if (this._memCache.has(filename)) return this._memCache.get(filename);

    // 2. IndexedDB persistent cache (raw bytes stored → decode on retrieval)
    if (this._db) {
      const cached = await this._db.get('tiles', filename);
      if (cached instanceof Uint8Array) {
        const tile = _decodeMsgpack(cached);
        this._memCache.set(filename, tile);
        return tile;
      }
    }

    // 3. Network fetch
    const bytes = await _fetchBytes(`${this._base}/${filename}`);

    // Persist raw bytes for future use; ignore quota errors silently
    if (this._db) {
      this._db.put('tiles', bytes, filename).catch(() => {});
    }

    const tile = _decodeMsgpack(bytes);
    this._memCache.set(filename, tile);
    return tile;
  }

  // ── Octree traversal ────────────────────────────────────────────────────────

  /**
   * Recursively collect all tiles that satisfy the bbox and level constraints,
   * mirroring the Python `_collect_tiles` logic.
   *
   * @param {string}        filename  - tile filename relative to base URL
   * @param {number[]|null} bbox      - query bbox (null = no spatial filter)
   * @param {number|null}   level     - max depth (null = leaves)
   * @param {number}        nSpatial  - 2 or 3
   * @returns {Promise<object[]>}
   */
  async _collectTiles(filename, bbox, level, nSpatial) {
    const tile = await this._loadTile(filename);

    // Prune branches that don't intersect the query bbox
    if (!_intersects(tile.bounds, bbox, nSpatial)) return [];

    const isLeaf    = tile.is_leaf ?? (tile.children == null);
    const tileLevel = tile.level ?? 0;

    // Return this tile if it is a leaf or we have hit the requested depth
    if (isLeaf || (level !== null && tileLevel >= level)) return [tile];

    // Recurse into children in parallel for throughput
    const children = tile.children ?? [];
    const childGroups = await Promise.all(
      children.map(child => this._collectTiles(child, bbox, level, nSpatial))
    );
    const collected = childGroups.flat();

    // If the bbox filtered out all children, fall back to this (coarser) tile
    // so the caller always receives at least a low-res result for the region
    return collected.length > 0 ? collected : [tile];
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Load all tiles intersecting `bbox` down to the requested `level`.
   *
   * @param {number[]|null} [bbox=null]
   *   Spatial bounding box in the same coordinate system as the dataset.
   *   - 2-D: `[x_min, y_min, x_max, y_max]`
   *   - 3-D: `[x_min, y_min, z_min, x_max, y_max, z_max]`
   *   Pass `null` to load the entire dataset (no spatial filter).
   *
   * @param {object}      [options={}]
   * @param {number|null} [options.level=null]
   *   Maximum octree depth to descend.
   *   - `null` (default): load all leaf tiles (full resolution).
   *   - `0`: load only the root tile (coarsest overview).
   *   - `N`: load tiles at depth N; uses the deepest available leaf for
   *     branches that terminate before depth N.
   *
   * @returns {Promise<GridResult>}
   */
  async loadBBox(bbox = null, { level = null } = {}) {
    if (!this._meta) throw new Error('Call open() before loadBBox()');

    const nSpatial = this._meta.spatial_dims.length;
    const rootFile = this._meta.root_tile ?? 'root.msgpack';
    const tiles    = await this._collectTiles(rootFile, bbox, level, nSpatial);
    return new GridResult(this._meta, tiles);
  }

  /**
   * Clear all cached tiles from both the in-memory cache and IndexedDB.
   * Useful when the server-side data has been regenerated.
   *
   * @returns {Promise<void>}
   */
  async clearCache() {
    this._memCache.clear();
    if (this._db) {
      const tx    = this._db.transaction('tiles', 'readwrite');
      const store = tx.objectStore('tiles');
      await store.clear();
      await tx.done;
    }
  }
}
