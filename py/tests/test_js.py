"""Cross-language roundtrip tests: Python writes → JavaScript reads.

For each scenario the test:
1. Writes a webxtile directory with Python's write_webxtile().
2. Invokes the Node.js helper (js/tests/read_tiles.mjs) to read the same
   directory, optionally with a bbox / level filter.
3. Compares the coordinates and data values reported by JavaScript against
   those returned by Python's read_webxtile().

Both sides round floats to 4 decimal places before comparison, which is
appropriate for float32 data (≈7 significant decimal digits).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from webxtile import write_webxtile, read_webxtile

# ─── Paths ───────────────────────────────────────────────────────────────────

# Repository root → deps/webxtile/js/tests/read_tiles.mjs
_THIS_DIR  = Path(__file__).parent
_JS_SCRIPT = _THIS_DIR.parent.parent / "js" / "tests" / "read_tiles.mjs"


# ─── Dataset factories ────────────────────────────────────────────────────────

def make_2d_dataset(nx=64, ny=48):
    x = np.linspace(0.0, 100.0, nx)
    y = np.linspace(0.0, 50.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    temperature   = (xx + yy * 0.5).astype(np.float32)
    precipitation = (np.sin(xx / 20.0) * np.cos(yy / 10.0)).astype(np.float32)
    return xr.Dataset(
        {
            "temperature":   (["x", "y"], temperature),
            "precipitation": (["x", "y"], precipitation),
        },
        coords={"x": x, "y": y},
    )


def make_3d_dataset(nx=32, ny=24, nz=16):
    x = np.linspace(0.0, 100.0, nx)
    y = np.linspace(0.0, 50.0, ny)
    z = np.linspace(0.0, 20.0, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    values = (xx + yy * 0.5 + zz * 0.1).astype(np.float32)
    return xr.Dataset(
        {"temperature": (["x", "y", "z"], values)},
        coords={"x": x, "y": y, "z": z},
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_js(tiles_dir, bbox=None, level=None):
    """Call the Node.js helper and return the parsed JSON output."""
    bbox_arg  = json.dumps(bbox)  if bbox  is not None else "null"
    level_arg = str(level)        if level is not None else "null"
    result = subprocess.run(
        ["node", str(_JS_SCRIPT), str(tiles_dir), bbox_arg, level_arg],
        capture_output=True,
        text=True,
        check=False,
        cwd=_JS_SCRIPT.parent.parent,  # js/ dir so node_modules are found
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Node.js script failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[:2000]}\n"
            f"stderr: {result.stderr[:2000]}"
        )
    return json.loads(result.stdout)


def _py_grid(ds, spatial_dims):
    """Return coords and rounded variable arrays from a Python Dataset.

    Returns a dict mirroring the JS output:
      {"coords": {dim: [...]}, "variables": {var: [[...][...]]}}
    """
    coords = {d: [round(float(v), 6) for v in ds[d].values] for d in spatial_dims}
    variables = {}
    for vname in ds.data_vars:
        arr = ds[vname].values.astype(np.float32)
        if arr.ndim == 2:
            variables[vname] = [
                [round(float(v), 4) for v in row] for row in arr
            ]
        else:
            # 3-D: flatten to a list of lists of lists
            variables[vname] = [
                [[round(float(v), 4) for v in plane] for plane in mat]
                for mat in arr
            ]
    return {"coords": coords, "variables": variables}


def _compare(js_out, py_out, label=""):
    """Assert that JS and Python produce the same coordinates and variable values."""
    for dim, py_vals in py_out["coords"].items():
        js_vals = js_out["coords"].get(dim, [])
        np.testing.assert_allclose(
            js_vals, py_vals, atol=1e-4,
            err_msg=f"{label}: coord '{dim}' mismatch",
        )

    for vname, py_grid in py_out["variables"].items():
        js_grid = js_out["variables"].get(vname)
        assert js_grid is not None, f"{label}: JS missing variable '{vname}'"
        py_arr = np.array(py_grid, dtype=np.float32)
        js_arr = np.array(js_grid, dtype=np.float32)
        # Allow small tolerance for float32 round-trip through JSON
        # (NaN positions should match)
        py_valid = ~np.isnan(py_arr)
        js_valid = ~np.isnan(js_arr)
        np.testing.assert_array_equal(
            py_valid, js_valid,
            err_msg=f"{label}: '{vname}' NaN mask differs between Python and JS",
        )
        np.testing.assert_allclose(
            js_arr[js_valid], py_arr[py_valid], atol=5e-3,
            err_msg=f"{label}: '{vname}' values differ between Python and JS",
        )


# ─── 2-D cross-language tests ─────────────────────────────────────────────────

def test_js_full_resolution_2d():
    """JS full-resolution read must match Python full-resolution read."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16)

        py_ds  = read_webxtile(d)
        js_out = _run_js(d)

    _compare(js_out, _py_grid(py_ds, ["x", "y"]), label="full-res 2D")


def test_js_bbox_centre_2d():
    """JS bbox read must match Python bbox read for the central sub-region."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16)

        bbox   = [25.0, 12.5, 75.0, 37.5]
        py_ds  = read_webxtile(d, bbox=bbox)
        js_out = _run_js(d, bbox=bbox)

    _compare(js_out, _py_grid(py_ds, ["x", "y"]), label="bbox-centre 2D")


def test_js_bbox_left_strip_2d():
    """JS and Python must agree on a narrow left strip of the domain."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16)

        bbox   = [0.0, 0.0, 30.0, 50.0]
        py_ds  = read_webxtile(d, bbox=bbox)
        js_out = _run_js(d, bbox=bbox)

    _compare(js_out, _py_grid(py_ds, ["x", "y"]), label="bbox-left-strip 2D")


def test_js_bbox_top_right_2d():
    """JS and Python must agree on the top-right quadrant."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16)

        bbox   = [50.0, 25.0, 100.0, 50.0]
        py_ds  = read_webxtile(d, bbox=bbox)
        js_out = _run_js(d, bbox=bbox)

    _compare(js_out, _py_grid(py_ds, ["x", "y"]), label="bbox-top-right 2D")


def test_js_level_0_2d():
    """JS and Python must return the same coarse root-tile overview."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=8)

        py_ds  = read_webxtile(d, level=0)
        js_out = _run_js(d, level=0)

    _compare(js_out, _py_grid(py_ds, ["x", "y"]), label="level-0 2D")


def test_js_level_1_2d():
    """JS and Python must match at level 1 (one octree level below root)."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=8)

        py_ds  = read_webxtile(d, level=1)
        js_out = _run_js(d, level=1)

    _compare(js_out, _py_grid(py_ds, ["x", "y"]), label="level-1 2D")


def test_js_level_2_2d():
    """JS and Python must match at level 2."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=8)

        py_ds  = read_webxtile(d, level=2)
        js_out = _run_js(d, level=2)

    _compare(js_out, _py_grid(py_ds, ["x", "y"]), label="level-2 2D")


def test_js_bbox_and_level_combined_2d():
    """Combining bbox + level=1 must give matching results in JS and Python."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=8)

        bbox   = [10.0, 5.0, 90.0, 45.0]
        py_ds  = read_webxtile(d, bbox=bbox, level=1)
        js_out = _run_js(d, bbox=bbox, level=1)

    _compare(js_out, _py_grid(py_ds, ["x", "y"]), label="bbox+level-1 2D")


# ─── 3-D cross-language tests ─────────────────────────────────────────────────

def _compare_3d_coords(js_out, py_ds, label=""):
    """For 3-D datasets, just compare the sorted unique coordinate values."""
    for dim in ["x", "y", "z"]:
        py_vals = sorted(float(v) for v in py_ds[dim].values)
        js_vals = js_out["coords"].get(dim, [])
        np.testing.assert_allclose(
            sorted(js_vals), py_vals, atol=1e-4,
            err_msg=f"{label}: 3D coord '{dim}' mismatch",
        )


def test_js_full_resolution_3d():
    """JS full-resolution 3-D read must return the same coordinates as Python."""
    ds = make_3d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"], max_leaf=8)

        py_ds  = read_webxtile(d)
        js_out = _run_js(d)

    _compare_3d_coords(js_out, py_ds, label="full-res 3D")

    # Also check that scatter count matches
    expected_count = len(py_ds["x"]) * len(py_ds["y"]) * len(py_ds["z"])
    assert js_out["count"] == expected_count, (
        f"3D scatter count mismatch: JS={js_out['count']}, "
        f"Python={expected_count}"
    )


def test_js_level_0_3d():
    """JS and Python must return the same number of points at level 0 (3-D)."""
    ds = make_3d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"], max_leaf=4)

        py_ds  = read_webxtile(d, level=0)
        js_out = _run_js(d, level=0)

    py_count = len(py_ds["x"]) * len(py_ds["y"]) * len(py_ds["z"])
    assert js_out["count"] == py_count, (
        f"3D level-0 count: JS={js_out['count']}, Python={py_count}"
    )
    _compare_3d_coords(js_out, py_ds, label="level-0 3D")


def test_js_bbox_3d():
    """JS and Python bbox reads must return the same coordinates for a 3-D sub-volume."""
    ds = make_3d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"], max_leaf=8)

        bbox   = [20.0, 10.0, 5.0, 80.0, 40.0, 15.0]
        py_ds  = read_webxtile(d, bbox=bbox)
        js_out = _run_js(d, bbox=bbox)

    _compare_3d_coords(js_out, py_ds, label="bbox 3D")
