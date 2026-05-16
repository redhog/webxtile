/**
 * Browser integration tests for WebxtileLoader.
 *
 * Uses Playwright + Chromium so that IndexedDB is the real browser API.
 * A bundle is built once via esbuild and injected into each test page.
 * Tile requests are intercepted via page.route() and served from in-memory
 * fixtures generated with @msgpack/msgpack encode().
 *
 * Tests:
 *   1. Basic load        — returns all leaf tiles
 *   2. IDB round-trip    — second load (after memCache clear) reads from IDB
 *   3. IDB queue         — at most 1 concurrent IDB transaction at any time
 *   4. Fetch limiter     — at most 16 concurrent network requests at any time
 *   5. Stress            — 20 concurrent loadBBox calls all complete correctly
 */

import { test, expect }          from '@playwright/test';
import { encode }                 from '@msgpack/msgpack';
import { execSync }               from 'child_process';
import { resolve, dirname }       from 'path';
import { fileURLToPath }          from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const JS_DIR    = resolve(__dirname, '..');
const BUNDLE    = resolve(__dirname, 'webxtile.bundle.js');

// ─── Bundle (built once per test run) ────────────────────────────────────────

test.beforeAll(() => {
  execSync(
    'node_modules/.bin/esbuild webxtile.js --bundle --format=iife --global-name=webxtile --outfile=tests/webxtile.bundle.js',
    { cwd: JS_DIR },
  );
});

// ─── Tile fixture generation ──────────────────────────────────────────────────
//
// Produces a minimal 2-level tree: 1 root + n leaf tiles.
// The root lists all leaves in its `children` array so a single loadBBox(null)
// triggers one BFS batch of n concurrent _loadTile() calls — the exact pattern
// that exposed the IDB scheduler stall.

function buf(obj) {
  return Buffer.from(encode(obj));
}

function makeTileSet(n = 16) {
  const map = {};
  const side = Math.ceil(Math.sqrt(n));
  const size = 100 / side;

  for (let i = 0; i < n; i++) {
    const col = i % side, row = Math.floor(i / side);
    map[`root_${i}.msgpack`] = buf({
      level: 1, is_leaf: true,
      bounds: [col * size, row * size, 0, (col + 1) * size, (row + 1) * size, 0],
      shape: [2, 2],
      spatial_coords: {
        x: [col * size + size * 0.25, col * size + size * 0.75],
        y: [row * size + size * 0.25, row * size + size * 0.75],
      },
      variables: { temp: [i, i + 0.1, i + 0.2, i + 0.3] },
    });
  }

  map['root.msgpack'] = buf({
    level: 0, is_leaf: false,
    bounds: [0, 0, 0, 100, 100, 0],
    shape: [side, side],
    spatial_coords: {
      x: Array.from({ length: side }, (_, k) => (k + 0.5) * size),
      y: Array.from({ length: side }, (_, k) => (k + 0.5) * size),
    },
    variables: { temp: Array.from({ length: side * side }, (_, k) => k) },
    children: Array.from({ length: n }, (_, k) => `root_${k}.msgpack`),
  });

  map['metadata.msgpack'] = buf({
    version: 1,
    spatial_dims: ['x', 'y'],
    split_axes:   ['x', 'y'],
    root_tile: 'root.msgpack',
    bounds: [0, 0, 0, 100, 100, 0],
    var_meta: { temp: { dims: ['x', 'y'], dtype: '<f4', attrs: {} } },
    dim_sizes: {}, coord_meta: {},
  });

  return map;
}

// ─── Page helpers ─────────────────────────────────────────────────────────────

// Serve tiles at http://tiles.local/<filename> from the in-memory map.
// Optionally add a per-response delay (ms) so concurrent requests overlap.
async function serveTiles(page, tiles, { base = 'http://tiles.local', delay = 0 } = {}) {
  await page.route(`${base}/**`, async (route) => {
    if (delay) await new Promise(r => setTimeout(r, delay));
    const name = route.request().url().slice(base.length + 1);
    const data = tiles[name];
    if (data) route.fulfill({ body: data, contentType: 'application/octet-stream' });
    else      route.fulfill({ status: 404 });
  });
}

// Navigate to a page with a stable http origin (required for IndexedDB) and
// inject the bundled webxtile.js as a global.
async function loadPage(page) {
  await page.route('http://test.local/', route =>
    route.fulfill({ contentType: 'text/html', body: '<!DOCTYPE html><html><body></body></html>' })
  );
  await page.goto('http://test.local/');
  await page.addScriptTag({ path: BUNDLE });
}

// ─── Tests ────────────────────────────────────────────────────────────────────

test('basic load: returns all leaf tiles', async ({ page }) => {
  await serveTiles(page, makeTileSet(16));
  await loadPage(page);

  const count = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const r = await loader.loadBBox(null);
    return r.tiles.length;
  });

  expect(count).toBe(16);
});

test('IDB round-trip: second load (memCache cleared) reads from IDB, not network', async ({ page }) => {
  await serveTiles(page, makeTileSet(16));
  await loadPage(page);

  const { first, second, networkHits } = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const r1 = await loader.loadBBox(null);  // fills both IDB and memCache

    // Wrap _fetchBytes to count tile network hits after the cache is cleared
    let networkHits = 0;
    const origFetch = loader._fetchBytes.bind(loader);
    loader._fetchBytes = url => { networkHits++; return origFetch(url); };

    loader._memCache.clear();
    const r2 = await loader.loadBBox(null);  // should read everything from IDB

    return { first: r1.tiles.length, second: r2.tiles.length, networkHits };
  });

  expect(first).toBe(16);
  expect(second).toBe(16);
  expect(networkHits).toBe(0);  // zero network fetches — all came from IDB
});

test('IDB queue: never more than 1 concurrent IDB transaction', async ({ page }) => {
  await serveTiles(page, makeTileSet(16));
  await loadPage(page);

  const { maxConcurrent, totalAcquires } = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    await loader.loadBBox(null);   // populate IDB

    loader._memCache.clear();      // next load must go through IDB

    // Instrument the IDB queue to track concurrent active transactions.
    // Capture prototype methods before overriding so the queue mechanics
    // (the _idbConcurrent counter and _idbWaiters list) still work correctly.
    let concurrent = 0, maxConcurrent = 0, totalAcquires = 0;
    const origAcquire = loader._acquireIdb.bind(loader);
    const origRelease = loader._releaseIdb.bind(loader);

    loader._acquireIdb = async function () {
      await origAcquire();   // blocks until the queue slot is free
      concurrent++;
      totalAcquires++;
      maxConcurrent = Math.max(maxConcurrent, concurrent);
    };
    loader._releaseIdb = function () {
      concurrent--;
      return origRelease();
    };

    await loader.loadBBox(null);
    return { maxConcurrent, totalAcquires };
  });

  // The 16-tile BFS batch issues 16 concurrent _loadTile calls; without the
  // queue each would open its own IDB transaction and Chrome's scheduler stalls.
  expect(maxConcurrent).toBeLessThanOrEqual(1);

  // Sanity-check: the queue was actually exercised (tiles came from IDB).
  expect(totalAcquires).toBeGreaterThanOrEqual(16);
});

test('fetch limiter: never more than 16 concurrent in-flight network requests', async ({ page }) => {
  const tiles = makeTileSet(16);

  // Count concurrency on the Node.js side: hold each response open for 10 ms
  // so that simultaneous requests genuinely overlap.
  let concurrent = 0, maxConcurrent = 0;
  await page.route('http://tiles.local/**', async route => {
    concurrent++;
    maxConcurrent = Math.max(maxConcurrent, concurrent);
    await new Promise(r => setTimeout(r, 10));
    concurrent--;
    const name = route.request().url().slice('http://tiles.local/'.length);
    const data = tiles[name];
    if (data) route.fulfill({ body: data, contentType: 'application/octet-stream' });
    else      route.fulfill({ status: 404 });
  });

  await loadPage(page);

  // Two loaders with separate IDB databases so each must fetch all tiles from
  // the network independently.  2 × 16 = 32 concurrent leaf-fetch attempts
  // flow through the global fetch limiter; it must cap them at 16.
  await page.evaluate(async () => {
    const a = new webxtile.WebxtileLoader('http://tiles.local', { dbName: 'tiles-a' });
    const b = new webxtile.WebxtileLoader('http://tiles.local', { dbName: 'tiles-b' });
    await Promise.all([a.open(), b.open()]);
    await Promise.all([a.loadBBox(null), b.loadBBox(null)]);
  });

  expect(maxConcurrent).toBeLessThanOrEqual(16);
  expect(maxConcurrent).toBeGreaterThan(0);
});

test('stress: 20 concurrent loadBBox calls all complete and return correct tile count', async ({ page }) => {
  await serveTiles(page, makeTileSet(16));
  await loadPage(page);

  // This would hang permanently if the IDB queue had a deadlock or lost waker.
  const ok = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const results = await Promise.all(
      Array.from({ length: 20 }, () => loader.loadBBox(null))
    );
    return results.every(r => r.tiles.length === 16);
  });

  expect(ok).toBe(true);
});

// ─── Adaptive split-axes ───────────────────────────────────────────────────────

// Builds a quadtree (split only x and y) with 3-D data.
function makeQuadtreeTileSet() {
  const map = {};
  // 4 leaf tiles (quadtree = 2 split axes → 4 children)
  const half = 50;
  const tiles = [
    { ci: 0, xr: [0, half],  yr: [0, half]  },
    { ci: 1, xr: [0, half],  yr: [half, 100] },
    { ci: 2, xr: [half, 100], yr: [0, half]  },
    { ci: 3, xr: [half, 100], yr: [half, 100] },
  ];
  for (const { ci, xr, yr } of tiles) {
    map[`root_${ci}.msgpack`] = buf({
      level: 1, is_leaf: true,
      // z bounds span full range [0, 10] for all tiles (quadtree keeps full depth)
      bounds: [xr[0], yr[0], 0, xr[1], yr[1], 10],
      shape: [2, 2, 4],
      spatial_coords: {
        x: [xr[0] + 12.5, xr[0] + 37.5],
        y: [yr[0] + 12.5, yr[0] + 37.5],
        z: [0, 3.33, 6.67, 10],
      },
      variables: { val: Array.from({ length: 2 * 2 * 4 }, (_, k) => k * 0.1) },
    });
  }

  map['root.msgpack'] = buf({
    level: 0, is_leaf: false,
    bounds: [0, 0, 0, 100, 100, 10],
    shape: [2, 2, 4],
    spatial_coords: { x: [25, 75], y: [25, 75], z: [0, 3.33, 6.67, 10] },
    variables: { val: Array.from({ length: 16 }, (_, k) => k * 0.1) },
    children: ['root_0.msgpack', 'root_1.msgpack', 'root_2.msgpack', 'root_3.msgpack'],
  });

  map['metadata.msgpack'] = buf({
    version: 1,
    spatial_dims: ['x', 'y', 'z'],
    split_axes:   ['x', 'y'],   // quadtree
    root_tile: 'root.msgpack',
    bounds: [0, 0, 0, 100, 100, 10],
    var_meta: { val: { dims: ['x', 'y', 'z'], dtype: '<f4', attrs: {} } },
    dim_sizes: {}, coord_meta: {},
  });

  return map;
}

test('split_axes: quadtree (2 split axes) returns 4 leaf tiles at level 1', async ({ page }) => {
  await serveTiles(page, makeQuadtreeTileSet());
  await loadPage(page);

  const result = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const r = await loader.loadBBox(null, 1);
    return {
      tileCount: r.tiles.length,
      splitAxes: r.splitAxes,
    };
  });

  expect(result.tileCount).toBe(4);
  expect(result.splitAxes).toEqual(['x', 'y']);
});

test('split_axes: quadtree leaf tiles all have full z depth', async ({ page }) => {
  await serveTiles(page, makeQuadtreeTileSet());
  await loadPage(page);

  const zCounts = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const r = await loader.loadBBox(null, 1);
    return r.tiles.map(t => (t.spatial_coords?.z?.length ?? 0));
  });

  // Every leaf tile should have all 4 z values (no z splitting)
  expect(zCounts.every(c => c === 4)).toBe(true);
});

test('split_axes: loadBBox with null level (tree traversal) returns all leaves', async ({ page }) => {
  await serveTiles(page, makeQuadtreeTileSet());
  await loadPage(page);

  const count = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const r = await loader.loadBBox(null);  // null level → tree traversal
    return r.tiles.length;
  });

  expect(count).toBe(4);
});

// ─── GPU sub-tiles ─────────────────────────────────────────────────────────────

function makeSubTileTileSet(gpuSize = 8) {
  const map = {};
  const nx = 20, ny = 20;
  // Build sub_tiles: chunk nx×ny into gpuSize×gpuSize sub-grids
  const subTiles = [];
  for (let xi = 0; xi < nx; xi += gpuSize) {
    const xEnd = Math.min(xi + gpuSize, nx);
    for (let yi = 0; yi < ny; yi += gpuSize) {
      const yEnd = Math.min(yi + gpuSize, ny);
      const sx = xEnd - xi, sy = yEnd - yi;
      subTiles.push({
        bounds: [xi, yi, 0, xEnd, yEnd, 0],
        shape:  [sx, sy],
        spatial_coords: {
          x: Array.from({ length: sx }, (_, k) => xi + k),
          y: Array.from({ length: sy }, (_, k) => yi + k),
        },
        variables: { val: Array.from({ length: sx * sy }, (_, k) => k * 0.1) },
      });
    }
  }

  map['root.msgpack'] = buf({
    level: 0, is_leaf: true,
    bounds: [0, 0, 0, nx, ny, 0],
    shape:  [nx, ny],
    spatial_coords: {
      x: Array.from({ length: nx }, (_, k) => k),
      y: Array.from({ length: ny }, (_, k) => k),
    },
    variables: { val: Array.from({ length: nx * ny }, (_, k) => k * 0.1) },
    sub_tiles: subTiles,
  });

  map['metadata.msgpack'] = buf({
    version: 1,
    spatial_dims: ['x', 'y'],
    split_axes:   ['x', 'y'],
    root_tile: 'root.msgpack',
    bounds: [0, 0, 0, nx, ny, 0],
    gpu_tile_size: gpuSize,
    var_meta: { val: { dims: ['x', 'y'], dtype: '<f4', attrs: {} } },
    dim_sizes: {}, coord_meta: {},
  });

  return map;
}

test('sub_tiles: subTiles() yields chunks with shape ≤ gpu_tile_size', async ({ page }) => {
  const gpuSize = 8;
  await serveTiles(page, makeSubTileTileSet(gpuSize));
  await loadPage(page);

  const shapes = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const r = await loader.loadBBox(null);
    const shapes = [];
    for (const st of r.subTiles()) {
      shapes.push(st.shape ?? []);
    }
    return shapes;
  });

  expect(shapes.length).toBeGreaterThan(0);
  for (const shape of shapes) {
    for (const dim of shape) {
      expect(dim).toBeLessThanOrEqual(gpuSize);
    }
  }
});

test('sub_tiles: subTiles() total point count matches tile total', async ({ page }) => {
  const gpuSize = 8;
  const nx = 20, ny = 20;
  await serveTiles(page, makeSubTileTileSet(gpuSize));
  await loadPage(page);

  const counts = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const r = await loader.loadBBox(null);
    let subTileTotal = 0;
    for (const st of r.subTiles()) {
      subTileTotal += (st.shape ?? []).reduce((a, b) => a * b, 1);
    }
    return { subTileTotal, tileTotal: r.toScatter().count };
  });

  expect(counts.subTileTotal).toBe(nx * ny);
  expect(counts.subTileTotal).toBe(counts.tileTotal);
});

test('sub_tiles: fallback when no sub_tiles field yields the tile itself', async ({ page }) => {
  await serveTiles(page, makeTileSet(4));
  await loadPage(page);

  const { subCount, tileCount } = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const r = await loader.loadBBox(null);
    let subCount = 0;
    for (const _ of r.subTiles()) subCount++;
    return { subCount, tileCount: r.tiles.length };
  });

  // Without sub_tiles field, each tile yields itself → counts equal
  expect(subCount).toBe(tileCount);
});

// ─── Manifest tiles ────────────────────────────────────────────────────────────

function makeManifestTileSet(n = 4) {
  // A sparse tileset: only n/2 of the n tiles actually exist.
  // The manifest records which ones exist so the client skips 404 fetches.
  const map = {};
  const side = Math.ceil(Math.sqrt(n));
  const size = 100 / side;
  const branch = 4; // quadtree
  const manifestDepth = 1;

  // Only create even-indexed tiles (sparse coverage)
  const existingIndices = new Set();
  for (let i = 0; i < n; i += 2) existingIndices.add(i);

  for (let i = 0; i < n; i++) {
    if (!existingIndices.has(i)) continue;
    const col = i % side, row = Math.floor(i / side);
    map[`root_${i}.msgpack`] = buf({
      level: 1, is_leaf: true,
      bounds: [col * size, row * size, 0, (col + 1) * size, (row + 1) * size, 0],
      shape: [2, 2],
      spatial_coords: {
        x: [col * size + size * 0.25, col * size + size * 0.75],
        y: [row * size + size * 0.25, row * size + size * 0.75],
      },
      variables: { temp: [i, i + 0.1, i + 0.2, i + 0.3] },
    });
  }

  map['root.msgpack'] = buf({
    level: 0, is_leaf: false,
    bounds: [0, 0, 0, 100, 100, 0],
    shape: [side, side],
    spatial_coords: {
      x: Array.from({ length: side }, (_, k) => (k + 0.5) * size),
      y: Array.from({ length: side }, (_, k) => (k + 0.5) * size),
    },
    variables: { temp: Array.from({ length: side * side }, (_, k) => k) },
    children: Array.from({ length: n }, (_, k) => `root_${k}.msgpack`),
  });

  // Build manifest bitmap for root: covers levels 0 and 1 (manifestDepth=1)
  // Slots in BFS order: [root, root_0, root_1, ..., root_{n-1}]
  // Total slots = 1 + n
  const totalSlots = 1 + n; // depth=1: root + n children
  const bits = new Uint8Array(Math.ceil(totalSlots / 8));
  // Root itself always exists (slot 0)
  bits[0] |= 1;
  // Children: set bit if existingIndices has this index
  for (let i = 0; i < n; i++) {
    if (existingIndices.has(i)) {
      const slot = 1 + i;
      bits[slot >> 3] |= (1 << (slot & 7));
    }
  }

  map['manifest/m_root.msgpack'] = buf({
    bounds: [0, 0, 0, 100, 100, 0],
    root:   'root.msgpack',
    depth:  manifestDepth,
    tiles:  Array.from(bits),  // plain array for msgpack encoding
  });

  map['metadata.msgpack'] = buf({
    version: 1,
    spatial_dims: ['x', 'y'],
    split_axes:   ['x', 'y'],
    root_tile: 'root.msgpack',
    bounds: [0, 0, 0, 100, 100, 0],
    manifest_depth: manifestDepth,
    var_meta: { temp: { dims: ['x', 'y'], dtype: '<f4', attrs: {} } },
    dim_sizes: {}, coord_meta: {},
  });

  return map;
}

test('manifest: loadBBox makes zero 404 requests for sparse dataset', async ({ page }) => {
  const n = 4;
  const tiles = makeManifestTileSet(n);

  let errorCount = 0;
  await page.route('http://tiles.local/**', async route => {
    const name = route.request().url().slice('http://tiles.local/'.length);
    const data = tiles[name];
    if (data) {
      route.fulfill({ body: data, contentType: 'application/octet-stream' });
    } else {
      errorCount++;
      route.fulfill({ status: 404 });
    }
  });

  await loadPage(page);

  const tileCount = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    const r = await loader.loadBBox(null, 1);
    return r.tiles.length;
  });

  // Only even-indexed tiles exist (n/2 = 2 tiles at level 1)
  expect(tileCount).toBe(Math.ceil(n / 2));
  // The manifest should have prevented any 404 fetches for data tiles
  expect(errorCount).toBe(0);
});

test('manifest: dataset without manifest falls back to speculative fetch', async ({ page }) => {
  // Use a normal tileset without a manifest — should still work (just 404s)
  await serveTiles(page, makeTileSet(4));
  await loadPage(page);

  const count = await page.evaluate(async () => {
    const loader = new webxtile.WebxtileLoader('http://tiles.local');
    await loader.open();
    // Verify no manifest_depth in metadata
    if (loader.meta.manifest_depth != null) return -1;
    const r = await loader.loadBBox(null, 1);
    return r.tiles.length;
  });

  expect(count).toBe(4);
});
