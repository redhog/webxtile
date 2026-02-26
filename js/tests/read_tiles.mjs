/**
 * read_tiles.mjs — Node.js helper for cross-language tests.
 *
 * Reads a webxtile directory from the local filesystem, applies optional
 * bbox / level filtering, and writes a JSON summary to stdout for comparison
 * with Python output.
 *
 * Usage:
 *   node read_tiles.mjs <tilesDir> [bboxJSON] [level]
 *
 * Arguments:
 *   tilesDir  – path to the directory produced by write_webxtile()
 *   bboxJSON  – optional JSON array, e.g. '[10,5,80,45]' or 'null'
 *   level     – optional integer (0 = root); omit or 'null' for full resolution
 *
 * Output (stdout): JSON object with shape
 *   {
 *     "spatial_dims": ["x","y"],
 *     "coords": { "x": [...], "y": [...] },   // sorted unique Float64 values
 *     "variables": { "varName": [[...],[...]] }, // 2-D grid [ix][iy]
 *     "count": <number of scatter points>
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

// ─── Grid extraction (convert scatter → indexed 2-D grid for comparison) ─────

function buildGrid(result) {
  const spatialDims = result.spatialDims;
  const { coords, variables, count } = result.toScatter();

  // Sorted unique coordinate values (Float64 precision, rounded to 6 dp)
  const sortedCoords = {};
  for (const dim of spatialDims) {
    sortedCoords[dim] = Array.from(result.getCoord(dim)).map(v => +v.toFixed(6));
  }

  if (spatialDims.length !== 2) {
    // For 3-D datasets return only flattened scatter for simplicity
    return {
      spatial_dims: spatialDims,
      coords: sortedCoords,
      scatter: {
        coords:    Object.fromEntries(
          Object.entries(coords).map(([k,v]) => [k, Array.from(v).map(x => +x.toFixed(4))])
        ),
        variables: Object.fromEntries(
          Object.entries(variables).map(([k,v]) => [k, Array.from(v).map(x => +x.toFixed(4))])
        ),
      },
      count,
    };
  }

  // 2-D: build a sorted index map to assemble a regular grid.
  // toScatter() converts spatial coords to Float32Array, so we must look up
  // scatter values using a precision safe for float32 (4 dp) rather than the
  // full float64 precision returned by getCoord().
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
    spatial_dims: spatialDims,
    coords: sortedCoords,
    variables: Object.fromEntries(
      Object.entries(grids).map(([k, rows]) => [
        k, rows.map(row => Array.from(row).map(v => +v.toFixed(4))),
      ])
    ),
    count,
  };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
if (args.length < 1) {
  process.stderr.write('Usage: node read_tiles.mjs <tilesDir> [bboxJSON] [level]\n');
  process.exit(1);
}

const tilesDir = resolve(args[0]);
const bbox     = args[1] && args[1] !== 'null' ? JSON.parse(args[1]) : null;
const level    = args[2] && args[2] !== 'null' ? parseInt(args[2], 10) : null;

const meta     = readMsgpack(join(tilesDir, 'metadata.msgpack'));
const rootFile = meta.root_tile ?? 'root.msgpack';
const nSpatial = meta.spatial_dims.length;

const tiles  = collectTiles(tilesDir, rootFile, { bbox, level, nSpatial });
const result = new WebxtileResult(meta, tiles);

const output = buildGrid(result);
process.stdout.write(JSON.stringify(output, null, 2) + '\n');
