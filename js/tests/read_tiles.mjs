/**
 * read_tiles.mjs — Node.js helper for cross-language tests.
 *
 * Reads a webxtile directory from the local filesystem, applies optional
 * bbox / level filtering, and writes a JSON summary to stdout for comparison
 * with Python output.
 *
 * Usage:
 *   node read_tiles.mjs <tilesDir> [bboxJSON] [level] [--sub-tiles]
 *
 * Arguments:
 *   tilesDir    – path to the directory produced by write_webxtile()
 *   bboxJSON    – optional JSON array, e.g. '[10,5,80,45]' or 'null'
 *   level       – optional integer (0 = root); omit or 'null' for full resolution
 *   --sub-tiles – if set, output sub-tile info instead of the normal grid
 *
 * Output (stdout): JSON object with shape
 *   {
 *     "spatial_dims": ["x","y"],
 *     "split_axes":   ["x","y"],
 *     "coords": { "x": [...], "y": [...] },   // sorted unique Float64 values
 *     "variables": { "varName": [[...],[...]] }, // 2-D grid [ix][iy]
 *     "count": <number of scatter points>,
 *     // when --sub-tiles:
 *     "sub_tile_count": <total sub-tiles>,
 *     "sub_tile_shapes": [[nx,ny,...], ...]
 *   }
 *
 * The `variables` field contains a 2-D regular grid (sorted by x, then y) so
 * that value-level comparison with Python's xarray Dataset is straightforward.
 */

import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { Decoder } from '@msgpack/msgpack';
import { WebxtileResult } from '../webxtile.js';

// msgpack_numpy (Python) encodes numpy array dicts with bytes keys (e.g. b'nd',
// b'type'). @msgpack/msgpack v3 lets us remap them to ordinary strings.
const _decoder = new Decoder({
  mapKeyConverter: k => k instanceof Uint8Array ? new TextDecoder().decode(k) : k,
});

// ─── NumPy array decoding (same logic as in webxtile.js) ─────────────────────
// (WebxtileResult needs already-decoded tiles; we decode them here with fs)

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

function readMsgpack(filepath) {
  const bytes = readFileSync(filepath);
  return _decodeNumpy(_decoder.decode(bytes));
}

// ─── Tile traversal (mirrors Python _collect_tiles / JS _collectTiles) ────────

function _intersects(tileBounds, bbox, nSpatial) {
  if (bbox === null) return true;
  for (let i = 0; i < nSpatial; i++) {
    if (bbox[i + nSpatial] < tileBounds[i])  return false;
    if (bbox[i]             > tileBounds[i + 3]) return false;
  }
  return true;
}

function collectTiles(tilesDir, filename, { bbox, level, nSpatial }) {
  const tile = readMsgpack(join(tilesDir, filename));

  if (!_intersects(tile.bounds, bbox, nSpatial)) return [];

  const isLeaf    = tile.is_leaf ?? (tile.children == null);
  const tileLevel = tile.level ?? 0;

  if (isLeaf || (level !== null && tileLevel >= level)) return [tile];

  const children = tile.children ?? [];
  const result = [];
  for (const child of children) {
    result.push(...collectTiles(tilesDir, child, { bbox, level, nSpatial }));
  }
  return result.length > 0 ? result : [tile];
}

// ─── Scatter from sub-tiles ───────────────────────────────────────────────────
// Expands each sub-tile's 1-D coord arrays to a meshgrid and concatenates
// across all sub-tiles, producing the same {coords, variables, count} shape
// that WebxtileResult.toScatter() used to return.

function _scatter(result) {
  const spatialDims = result.spatialDims;
  const nD = spatialDims.length;

  const coordParts = Object.fromEntries(spatialDims.map(d => [d, []]));
  const varParts   = {};
  let count = 0;

  for (const st of result.subTiles()) {
    const sc      = st.spatial_coords ?? {};
    const dimArrs = spatialDims.map(d => sc[d] ?? new Float64Array(0));
    const nPerDim = dimArrs.map(a => a.length);
    const nTotal  = nPerDim.reduce((a, b) => a * b, 1);
    if (nTotal === 0) continue;

    const strides = new Array(nD);
    strides[nD - 1] = 1;
    for (let d = nD - 2; d >= 0; d--) strides[d] = strides[d + 1] * nPerDim[d + 1];

    for (let di = 0; di < nD; di++) {
      const arr = new Float32Array(nTotal);
      for (let flat = 0; flat < nTotal; flat++) {
        arr[flat] = dimArrs[di][Math.floor(flat / strides[di]) % nPerDim[di]];
      }
      coordParts[spatialDims[di]].push(arr);
    }

    for (const [varName, varArr] of Object.entries(st.variables ?? {})) {
      if (!varParts[varName]) varParts[varName] = [];
      varParts[varName].push(varArr instanceof Float32Array ? varArr : new Float32Array(varArr));
    }

    count += nTotal;
  }

  const merge = parts => {
    const total  = parts.reduce((s, a) => s + a.length, 0);
    const merged = new Float32Array(total);
    let off = 0;
    for (const a of parts) { merged.set(a, off); off += a.length; }
    return merged;
  };

  const coords    = Object.fromEntries(spatialDims.map(d => [d, merge(coordParts[d])]));
  const variables = Object.fromEntries(Object.entries(varParts).map(([k, p]) => [k, merge(p)]));
  return { coords, variables, count };
}

// ─── Grid extraction ─────────────────────────────────────────────────────────

function buildGrid(result, { subTilesMode = false } = {}) {
  const spatialDims = result.spatialDims;
  const { coords, variables, count } = _scatter(result);

  // Sorted unique coordinate values (Float64 precision, rounded to 6 dp)
  const sortedCoords = {};
  for (const dim of spatialDims) {
    sortedCoords[dim] = Array.from(result.getCoord(dim)).map(v => +v.toFixed(6));
  }

  const base = {
    spatial_dims: spatialDims,
    split_axes:   result.splitAxes,
    coords: sortedCoords,
    count,
  };

  // Sub-tile summary (for gpu_tile_size tests)
  if (subTilesMode) {
    const subTileShapes = [];
    for (const st of result.subTiles()) {
      subTileShapes.push(st.shape ?? []);
    }
    return {
      ...base,
      sub_tile_count:  subTileShapes.length,
      sub_tile_shapes: subTileShapes,
    };
  }

  if (spatialDims.length !== 2) {
    // For 3-D datasets return only flattened scatter for simplicity
    return {
      ...base,
      scatter: {
        coords:    Object.fromEntries(
          Object.entries(coords).map(([k,v]) => [k, Array.from(v).map(x => +x.toFixed(4))])
        ),
        variables: Object.fromEntries(
          Object.entries(variables).map(([k,v]) => [k, Array.from(v).map(x => +x.toFixed(4))])
        ),
      },
    };
  }

  // 2-D: build a sorted index map to assemble a regular grid.
  const PREC = 4;
  const fmt  = v => (+v).toFixed(PREC);

  const [xDim, yDim] = spatialDims;
  const xVals = sortedCoords[xDim];
  const yVals = sortedCoords[yDim];
  const xIdx  = new Map(xVals.map((v, i) => [fmt(v), i]));
  const yIdx  = new Map(yVals.map((v, i) => [fmt(v), i]));
  const nx = xVals.length, ny = yVals.length;

  // Allocate NaN grids
  const grids = {};
  for (const vname of Object.keys(variables)) {
    grids[vname] = Array.from({ length: nx }, () => new Float32Array(ny).fill(NaN));
  }

  const xCoords = coords[xDim];
  const yCoords = coords[yDim];
  for (let pt = 0; pt < count; pt++) {
    const xi = xIdx.get(fmt(xCoords[pt]));
    const yi = yIdx.get(fmt(yCoords[pt]));
    if (xi === undefined || yi === undefined) continue;
    for (const [vname, vArr] of Object.entries(variables)) {
      grids[vname][xi][yi] = vArr[pt];
    }
  }

  return {
    ...base,
    variables: Object.fromEntries(
      Object.entries(grids).map(([k, rows]) => [
        k, rows.map(row => Array.from(row).map(v => +v.toFixed(4))),
      ])
    ),
  };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
if (args.length < 1) {
  process.stderr.write('Usage: node read_tiles.mjs <tilesDir> [bboxJSON] [level] [--sub-tiles]\n');
  process.exit(1);
}

const subTilesMode = args.includes('--sub-tiles');
const posArgs      = args.filter(a => a !== '--sub-tiles');

const tilesDir = resolve(posArgs[0]);
const bbox     = posArgs[1] && posArgs[1] !== 'null' ? JSON.parse(posArgs[1]) : null;
const level    = posArgs[2] && posArgs[2] !== 'null' ? parseInt(posArgs[2], 10) : null;

const meta     = readMsgpack(join(tilesDir, 'metadata.msgpack'));
const rootFile = meta.root_tile ?? 'root.msgpack';
const nSpatial = meta.spatial_dims.length;

const tiles  = collectTiles(tilesDir, rootFile, { bbox, level, nSpatial });
const result = new WebxtileResult(meta, tiles);

const output = buildGrid(result, { subTilesMode });
process.stdout.write(JSON.stringify(output, null, 2) + '\n');
