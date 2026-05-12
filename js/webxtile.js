/**
 * webxtile.js
 * Browser client for the webxtile octree format.
 * Read-only; designed for partial bbox loads for web visualisation.
 *
 * @example
 *   import { WebxtileLoader } from "webxtile";
 *
 *   const loader = new WebxtileLoader("https://example.com/tiles");
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

// ─── Fetch concurrency limiter ────────────────────────────────────────────────
// Unbounded parallel tile fetches exhaust the browser's pending-request pool
// (Chrome: ERR_INSUFFICIENT_RESOURCES). Limit total in-flight network fetches
// across all WebxtileLoader instances so the browser stays responsive.

const _MAX_CONCURRENT_FETCHES = 16;
let _concurrentFetches = 0;
const _fetchWaiters = [];

function _acquireFetchSlot() {
  return new Promise(resolve => {
    if (_concurrentFetches < _MAX_CONCURRENT_FETCHES) {
      _concurrentFetches++;
      resolve();
    } else {
      _fetchWaiters.push(resolve);
    }
  });
}

function _releaseFetchSlot() {
  if (_fetchWaiters.length > 0) {
    _fetchWaiters.shift()();
  } else {
    _concurrentFetches--;
  }
}

// ─── Tile-name helpers ────────────────────────────────────────────────────────

/**
 * Return the parent tile filename, or null if `filename` is the root.
 *
 * Tile names follow the pattern root_i_j_k….msgpack where each _N suffix is
 * a child-index digit.  Stripping the last _N gives the parent stem.
 *
 *   root_0_3.msgpack  →  root_0.msgpack
 *   root_0.msgpack    →  root.msgpack
 *   root.msgpack      →  null
 */
function _parentFilename(filename) {
  const stem = filename.replace(/\.msgpack$/, '');
  const us   = stem.lastIndexOf('_');
  if (us < 0 || !/^\d+$/.test(stem.slice(us + 1))) return null;
  return stem.slice(0, us) + '.msgpack';
}

/**
 * Generate filenames of all tiles at `targetLevel` whose approximate bbox
 * intersects `bbox`.  Bounds are computed by halving each axis at its
 * coordinate midpoint — an approximation of the true index-midpoint split.
 * False positives (extra 404 requests) are harmless; false negatives cannot
 * occur because the caller's parent-fallback walk covers any missed tiles.
 */
function _candidatesAtLevel(rootFilename, rootBounds, bbox, targetLevel, nSpatial) {
  const branchFactor = 1 << nSpatial;          // 4 for 2-D, 8 for 3-D
  const rootStem     = rootFilename.replace(/\.msgpack$/, '');
  const out          = [];

  function recurse(stem, bounds, depth) {
    if (!_intersects(bounds, bbox, nSpatial)) return;
    if (depth === targetLevel) { out.push(stem + '.msgpack'); return; }
    for (let ci = 0; ci < branchFactor; ci++) {
      const cb = bounds.slice();
      for (let d = 0; d < nSpatial; d++) {
        const bit = nSpatial - 1 - d;           // X = MSB, Z = LSB
        const mid = (bounds[d] + bounds[d + 3]) / 2;
        if ((ci >> bit) & 1) cb[d]     = mid;   // upper half
        else                 cb[d + 3] = mid;   // lower half
      }
      recurse(stem + '_' + ci, cb, depth + 1);
    }
  }

  recurse(rootStem, rootBounds, 0);
  return out;
}

// ─── Low-level fetch & decode ─────────────────────────────────────────────────

const _textDecoder = new TextDecoder();

function _msgpackKeyConverter(key) {
  // msgpack_numpy (Python) may emit binary keys (msgpack bin) for dict keys
  // that are bytes objects. @msgpack/msgpack decodes those as Uint8Array, which
  // then causes a "key must be string or number" error. Convert to string.
  if (key instanceof Uint8Array) return _textDecoder.decode(key);
  return key;
}

function _decodeMsgpack(bytes) {
  return _decodeNumpy(decode(bytes, { mapKeyConverter: _msgpackKeyConverter }));
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

// ─── WebxtileResult ───────────────────────────────────────────────────────────

/**
 * Holds the tiles collected for a bbox / level query.
 *
 * The two main entry points for data consumption are:
 *   - `toScatter()` — flat parallel arrays per dimension and variable,
 *     suitable for point-cloud or scatter-plot style WebGL rendering.
 *   - `getCoord(dimName)` — merged sorted coordinate values for one dim.
 *   - `tiles` / `meta` — raw access for custom processing.
 */
export class WebxtileResult {
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

    // ── Pass 1: geometry per tile, total point count, variable name union ──────
    const tileInfo = [];
    let totalCount = 0;
    const varNames = new Set();

    for (const tile of this._tiles) {
      const sc      = tile.spatial_coords ?? {};
      const dimArrs = spatialDims.map(d => sc[d] ?? new Float64Array(0));
      const nPerDim = dimArrs.map(a => a.length);
      const nTotal  = nPerDim.reduce((a, b) => a * b, 1);
      tileInfo.push({ dimArrs, nPerDim, nTotal });
      totalCount += nTotal;
      for (const name of Object.keys(tile.variables ?? {})) varNames.add(name);
    }

    // ── Pass 2: allocate output arrays (no intermediate JS arrays) ─────────────
    const coords    = Object.fromEntries(spatialDims.map(d => [d, new Float32Array(totalCount)]));
    const variables = Object.fromEntries([...varNames].map(n => [n, new Float32Array(totalCount).fill(NaN)]));

    // ── Pass 3: fill ───────────────────────────────────────────────────────────
    let offset = 0;
    for (let ti = 0; ti < this._tiles.length; ti++) {
      const tile              = this._tiles[ti];
      const { dimArrs, nPerDim, nTotal } = tileInfo[ti];
      if (nTotal === 0) continue;

      // Row-major strides for the spatial meshgrid (dim 0 = outermost / slowest)
      const spStrides = new Array(nD);
      spStrides[nD - 1] = 1;
      for (let d = nD - 2; d >= 0; d--) spStrides[d] = spStrides[d + 1] * nPerDim[d + 1];

      for (let flat = 0; flat < nTotal; flat++) {
        for (let d = 0; d < nD; d++) {
          coords[spatialDims[d]][offset + flat] = dimArrs[d][Math.floor(flat / spStrides[d]) % nPerDim[d]];
        }
      }

      for (const [varName, rawArr] of Object.entries(tile.variables ?? {})) {
        const varOut = variables[varName];
        if (!varOut) continue;
        const vmeta = this._meta.var_meta?.[varName];
        if (!vmeta) continue;

        const varDims    = vmeta.dims;
        const spAxis     = varDims.map(d => spatialDims.indexOf(d));
        const varShape   = varDims.map((d, vi) => {
          const si = spAxis[vi];
          return si >= 0 ? nPerDim[si] : (this._meta.dim_sizes?.[d] ?? 1);
        });
        const varStrides = new Array(varDims.length);
        varStrides[varDims.length - 1] = 1;
        for (let d = varDims.length - 2; d >= 0; d--) {
          varStrides[d] = varStrides[d + 1] * varShape[d + 1];
        }

        for (let flat = 0; flat < nTotal; flat++) {
          let vi = 0;
          for (let vd = 0; vd < varDims.length; vd++) {
            const si = spAxis[vd];
            vi += (si >= 0 ? Math.floor(flat / spStrides[si]) % nPerDim[si] : 0) * varStrides[vd];
          }
          varOut[offset + flat] = rawArr[vi] ?? NaN;
        }
      }

      offset += nTotal;
    }

    return { coords, variables, count: totalCount };
  }
}

// ─── WebxtileLoader ───────────────────────────────────────────────────────────

/**
 * Loader for a webxtile octree dataset served over HTTP.
 *
 * Tiles are persisted to IndexedDB after the first network fetch so that
 * repeated loads within a session (or across sessions) avoid redundant
 * requests.
 *
 * @example
 *   const loader = new WebxtileLoader("https://host/tiles");
 *   await loader.open();
 *
 *   // Load leaves inside a 2-D bbox (full resolution)
 *   const r = await loader.loadBBox([500000, 6200000, 520000, 6220000]);
 *
 *   // Coarse overview (level 2)
 *   const lo = await loader.loadBBox(null, { level: 2 });
 */
export class WebxtileLoader {
  /**
   * @param {string} baseUrl    - Base URL of the tile directory (trailing
   *   slash optional).
   * @param {object} [options]
   * @param {string} [options.dbName="webxtile-cache"] - IndexedDB database
   *   name.  Use a unique name per dataset if you serve multiple datasets from
   *   the same origin.
   */
  constructor(baseUrl, { dbName = 'webxtile-cache', acquire = _acquireFetchSlot, release = _releaseFetchSlot } = {}) {
    this._base          = baseUrl.replace(/\/$/, '');
    this._dbName        = dbName;
    this._meta          = null;   // set by open()
    this._db            = null;   // IDBDatabase, set by open()
    this._memCache      = new Map(); // filename → decoded tile (session-level)
    this._acquire       = acquire;
    this._release       = release;
    this._idbConcurrent = 0;
    this._idbWaiters    = [];
    this._bboxActive    = 0;      // count of in-flight loadBBox calls
    this._bboxIdleWaiters = [];   // resolved when _bboxActive drops to zero
  }

  _acquireIdb() {
    return new Promise(resolve => {
      if (this._idbConcurrent < 1) {
        this._idbConcurrent++;
        resolve();
      } else {
        this._idbWaiters.push(resolve);
      }
    });
  }

  _releaseIdb() {
    if (this._idbWaiters.length > 0) {
      this._idbWaiters.shift()();
    } else {
      this._idbConcurrent--;
    }
  }

  // Resolves immediately when no loadBBox call is in flight; otherwise waits.
  _waitBboxIdle() {
    if (this._bboxActive === 0) return Promise.resolve();
    return new Promise(r => this._bboxIdleWaiters.push(r));
  }

  async _doFetch(url, nullable) {
    await this._acquire();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60_000);
    try {
      const res = await fetch(url, { signal: controller.signal });
      if (res.status === 404 && nullable) return null;
      if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
      return new Uint8Array(await res.arrayBuffer());
    } catch (err) {
      if (err.name === 'AbortError') throw new Error(`Timeout fetching ${url}`);
      throw err;
    } finally {
      clearTimeout(timer);
      this._release();
    }
  }

  async _fetchBytes(url)       { return this._doFetch(url, false); }
  async _fetchBytesOrNull(url) { return this._doFetch(url, true);  }

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
    const bytes = await this._fetchBytes(`${this._base}/${filename}`);
    return _decodeMsgpack(bytes);
  }

  async _loadTile(filename) {
    const tile = await this._loadTileOrNull(filename);
    if (tile === null) throw new Error(`Tile not found: ${filename}`);
    return tile;
  }

  async _loadTileOrNull(filename) {
    if (this._memCache.has(filename)) return this._memCache.get(filename);

    if (this._db) {
      await this._acquireIdb();
      let cached;
      try {
        cached = await this._db.get('tiles', filename);
      } finally {
        this._releaseIdb();
      }
      if (cached instanceof Uint8Array) {
        const tile = _decodeMsgpack(cached);
        tile._filename = filename;
        this._memCache.set(filename, tile);
        return tile;
      }
    }

    const bytes = await this._fetchBytesOrNull(`${this._base}/${filename}`);
    if (bytes === null) return null;

    if (this._db) {
      this._acquireIdb().then(() =>
        this._db.put('tiles', bytes, filename).catch(() => {}).finally(() => this._releaseIdb())
      );
    }

    const tile = _decodeMsgpack(bytes);
    tile._filename = filename;
    this._memCache.set(filename, tile);
    return tile;
  }

  // ── Octree traversal ────────────────────────────────────────────────────────

  /**
   * Fetch tiles for a bbox query without traversing intermediate levels.
   *
   * Algorithm:
   *   1. Fetch the root tile (needed for its bounds).
   *   2. Generate candidate filenames at `level` by recursively halving the
   *      root bounds and keeping only those that intersect `bbox`.
   *   3. Batch-fetch all candidates in parallel (16 at a time).
   *   4. For each candidate that returned 404, walk up the parent chain
   *      (_parentFilename strips one _N suffix per step) until an existing
   *      tile is found.  A Set deduplicates ancestors shared by multiple
   *      missing candidates.
   *
   * This means only two sequential round trips regardless of tree depth:
   * one for the root, one for the target-level batch.  Parent fallback adds
   * at most one additional sequential hop per 404 cluster.
   *
   * @param {string}        rootFilename
   * @param {number[]|null} bbox     - spatial filter; null = no filter
   * @param {number}        level    - target octree depth (0 = root only)
   * @param {number}        nSpatial - 2 or 3
   * @returns {Promise<object[]>}
   */
  async _collectBBox(rootFilename, bbox, level, nSpatial) {
    const rootTile = await this._loadTile(rootFilename);
    if (!_intersects(rootTile.bounds, bbox, nSpatial)) return [];
    if (level === 0) return [rootTile];

    const candidates = _candidatesAtLevel(rootFilename, rootTile.bounds, bbox, level, nSpatial);

    // seen: filename → tile (null means confirmed 404)
    // Pre-seed root so the parent-walk always terminates.
    const seen = new Map([[rootFilename, rootTile]]);

    // Batch-fetch all candidates in parallel.
    for (let i = 0; i < candidates.length; i += 16) {
      const batch = candidates.slice(i, i + 16).filter(fn => !seen.has(fn));
      const tiles = await Promise.all(batch.map(fn => this._loadTileOrNull(fn)));
      batch.forEach((fn, j) => seen.set(fn, tiles[j]));
    }

    // For each candidate, resolve to its deepest existing ancestor.
    const resultFiles = new Set();
    for (const candidate of candidates) {
      let fn = candidate;
      while (seen.get(fn) == null) {               // null (404) or unseen
        const parent = _parentFilename(fn);
        if (parent === null) { fn = rootFilename; break; }  // hit root
        if (!seen.has(parent)) seen.set(parent, await this._loadTileOrNull(parent));
        fn = parent;
      }
      resultFiles.add(fn);
    }

    return [...resultFiles].map(fn => seen.get(fn));
  }

  /**
   * Async generator that streams every leaf tile in the dataset via a
   * full BFS traversal.  Background use only — yields between batches and
   * pauses automatically whenever a `loadBBox` call is in flight so that
   * interactive queries are not starved of fetch slots.
   *
   * @param {object}      [options={}]
   * @param {AbortSignal} [options.signal] - cancellation signal
   * @yields {object} decoded tile objects (same shape as `WebxtileResult.tiles`)
   *
   * @example
   *   const stream = loader.streamLeaves();
   *   for await (const tile of stream) {
   *     cache.add(tile);
   *   }
   *
   *   // Cancel at any time:
   *   const ac = new AbortController();
   *   for await (const tile of loader.streamLeaves({ signal: ac.signal })) { … }
   *   ac.abort();
   */
  async *streamLeaves({ signal } = {}) {
    if (!this._meta) throw new Error('Call open() before streamLeaves()');
    const nSpatial   = this._meta.spatial_dims.length;
    const rootFile   = this._meta.root_tile ?? 'root.msgpack';
    const visited    = new Set([rootFile]);
    let   frontier   = [rootFile];

    while (frontier.length > 0) {
      if (signal?.aborted) return;

      // Pause while any loadBBox call holds priority.
      await this._waitBboxIdle();
      if (signal?.aborted) return;

      const next = [];
      for (let i = 0; i < frontier.length; i += 16) {
        // Re-check priority before each batch — a loadBBox may arrive mid-loop.
        await this._waitBboxIdle();
        if (signal?.aborted) return;

        const batch = frontier.slice(i, i + 16);
        const tiles = await Promise.all(batch.map(fn => this._loadTile(fn)));

        for (const tile of tiles) {
          const isLeaf   = tile.is_leaf ?? (tile.children == null);
          const children = tile.children ?? [];

          if (isLeaf || children.length === 0) {
            yield tile;
            continue;
          }
          const newChildren = children.filter(fn => !visited.has(fn));
          for (const fn of newChildren) visited.add(fn);
          next.push(...newChildren);
        }
      }

      frontier = next;
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Load tiles intersecting `bbox` at octree depth `level`.
   *
   * For branches that terminate above `level` the deepest available tile is
   * used.  For branches where no child passes the bbox filter the coarser
   * parent is used as a fallback.
   *
   * To progressively load the full dataset to leaves use `streamLeaves()`.
   *
   * @param {number[]|null} bbox
   *   Spatial bounding box in the same CRS as the dataset.
   *   - 2-D: `[x_min, y_min, x_max, y_max]`
   *   - 3-D: `[x_min, y_min, z_min, x_max, y_max, z_max]`
   *   Pass `null` for no spatial filter (whole dataset at this level).
   *
   * @param {number} level
   *   Octree depth to descend to.  0 = root tile only (coarsest overview).
   *
   * @returns {Promise<WebxtileResult>}
   */
  async loadBBox(bbox, level) {
    if (!this._meta) throw new Error('Call open() before loadBBox()');
    if (level == null) throw new Error('loadBBox() requires an explicit level. Use streamLeaves() to load all leaves.');

    this._bboxActive++;
    try {
      const nSpatial = this._meta.spatial_dims.length;
      const rootFile = this._meta.root_tile ?? 'root.msgpack';
      const tiles    = await this._collectBBox(rootFile, bbox, level, nSpatial);
      return new WebxtileResult(this._meta, tiles);
    } finally {
      this._bboxActive--;
      if (this._bboxActive === 0) {
        const waiters = this._bboxIdleWaiters.splice(0);
        for (const r of waiters) r();
      }
    }
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
      await this._acquireIdb();
      try {
        const tx = this._db.transaction('tiles', 'readwrite');
        await tx.objectStore('tiles').clear();
        await tx.done;
      } finally {
        this._releaseIdb();
      }
    }
  }
}
