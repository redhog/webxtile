"""Tests for adaptive axis splitting, GPU sub-tiles, and manifest tiles."""
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import pytest

from webxtile import write_webxtile, read_webxtile
from webxtile import (
    _compute_split_axes,
    _child_isel,
    _bfs_tile_names,
    _build_manifest,
    _unpack,
)


# ─── Dataset factories ────────────────────────────────────────────────────────

def make_geophysics_ds(nx=64, ny=64, nz=16):
    """3-D dataset where x/y span >> z span (geophysics survey geometry)."""
    x = np.linspace(0.0, 50_000.0, nx)   # 50 km horizontal
    y = np.linspace(0.0, 50_000.0, ny)
    z = np.linspace(0.0, 500.0, nz)       # 500 m depth
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    vals = (xx / 50_000 + yy / 50_000 + zz / 500).astype(np.float32)
    return xr.Dataset(
        {"resistivity": (["x", "y", "z"], vals)},
        coords={"x": x, "y": y, "z": z},
    )


def make_cubic_ds(n=16):
    """3-D dataset with equal x/y/z spans (cubic domain)."""
    c = np.linspace(0.0, 100.0, n)
    xx, yy, zz = np.meshgrid(c, c, c, indexing="ij")
    return xr.Dataset(
        {"val": (["x", "y", "z"], (xx + yy + zz).astype(np.float32))},
        coords={"x": c, "y": c, "z": c},
    )


def make_2d_ds(nx=64, ny=48):
    x = np.linspace(0.0, 100.0, nx)
    y = np.linspace(0.0, 100.0, ny)   # equal span to x → quadtree fallback
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return xr.Dataset(
        {"temperature": (["x", "y"], (xx + yy).astype(np.float32))},
        coords={"x": x, "y": y},
    )


# ─── Feature 1: Adaptive axis splitting ──────────────────────────────────────

def test_compute_split_axes_geophysics():
    """Geophysics dataset: only x and y should be split (not z)."""
    ds = make_geophysics_ds()
    split = _compute_split_axes(ds, ["x", "y", "z"])
    assert set(split) == {"x", "y"}
    assert "z" not in split


def test_compute_split_axes_cubic():
    """Cubic domain with equal spans: all three axes must be split (octree fallback)."""
    ds = make_cubic_ds()
    split = _compute_split_axes(ds, ["x", "y", "z"])
    assert set(split) == {"x", "y", "z"}


def test_compute_split_axes_2d():
    """2-D dataset with equal spans splits both axes (quadtree)."""
    ds = make_2d_ds()
    split = _compute_split_axes(ds, ["x", "y"])
    assert set(split) == {"x", "y"}


def test_compute_split_axes_1d_like():
    """Dataset with one very large axis and one trivial axis → split only large."""
    x = np.linspace(0.0, 10_000.0, 64)
    z = np.linspace(0.0, 1.0, 4)
    ds = xr.Dataset(
        {"val": (["x", "z"], np.zeros((64, 4), dtype=np.float32))},
        coords={"x": x, "z": z},
    )
    split = _compute_split_axes(ds, ["x", "z"])
    assert split == ["x"]


def test_split_axes_stored_in_metadata():
    """split_axes field must be written to metadata.msgpack."""
    ds = make_geophysics_ds()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"])
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert "split_axes" in meta
    assert set(meta["split_axes"]) == {"x", "y"}


def test_geophysics_tiles_have_full_z_depth():
    """For a geophysics dataset (quadtree), leaf tiles should span all z values."""
    ds = make_geophysics_ds(nx=32, ny=32, nz=8)
    z_vals = set(float(v) for v in ds["z"].values)
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"], max_leaf=8)
        # Check a leaf tile – all leaf tiles should contain all z values
        root = _unpack(Path(d) / "root.msgpack")
        # Follow children to find a leaf
        def first_leaf(tile_dict, path):
            if tile_dict.get("is_leaf", False):
                return tile_dict
            for child_name in tile_dict.get("children", []):
                child = _unpack(Path(path) / child_name)
                leaf = first_leaf(child, path)
                if leaf:
                    return leaf
            return None
        leaf = first_leaf(root, d)
        assert leaf is not None
        leaf_z = set(float(v) for v in leaf["spatial_coords"]["z"])
        # Leaf z values must be a subset of original z values
        assert leaf_z.issubset(z_vals)
        # The leaf must contain ALL z values (no z splitting)
        assert leaf_z == z_vals, (
            f"Leaf tile has z={leaf_z}, expected {z_vals}. "
            "Quadtree should give full-depth columns."
        )


def test_child_isel_count():
    """_child_isel returns 2^len(split_axes) children."""
    ds = make_geophysics_ds(nx=32, ny=32, nz=8)
    children_2 = _child_isel(ds, ["x", "y", "z"], ["x", "y"])
    children_3 = _child_isel(ds, ["x", "y", "z"], ["x", "y", "z"])
    assert len(children_2) == 4   # quadtree
    assert len(children_3) == 8   # octree


def test_child_isel_partitions_split_axes():
    """_child_isel children are non-overlapping and cover the full split-axis range."""
    ds = make_geophysics_ds(nx=32, ny=32, nz=8)
    children = _child_isel(ds, ["x", "y", "z"], ["x", "y"])
    x_slices = [c["x"] for c in children]
    # All x slices together should cover [0, 32)
    x_covered = set()
    for sl in x_slices:
        x_covered.update(range(*sl.indices(32)))
    assert x_covered == set(range(32))


def test_child_isel_nonsplit_axis_full():
    """Non-split axes must get full slice in every child."""
    ds = make_geophysics_ds(nx=32, ny=32, nz=8)
    children = _child_isel(ds, ["x", "y", "z"], ["x", "y"])
    nz = ds.sizes["z"]
    for c in children:
        assert c["z"] == slice(0, nz), "z should be full slice in all quadtree children"


def test_adaptive_split_roundtrip():
    """Full roundtrip with adaptive splitting preserves data values."""
    ds = make_geophysics_ds(nx=32, ny=32, nz=8)
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"], max_leaf=8)
        ds2 = read_webxtile(d)
    for dim in ["x", "y", "z"]:
        np.testing.assert_array_equal(ds2[dim].values, ds[dim].values)
    np.testing.assert_array_almost_equal(
        ds2["resistivity"].values, ds["resistivity"].values, decimal=4
    )


# ─── Feature 2: GPU sub-tiles ─────────────────────────────────────────────────

def test_sub_tiles_present_in_leaf():
    """Leaf tiles must contain a sub_tiles field when gpu_tile_size is set."""
    ds = make_2d_ds(nx=64, ny=64)
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=32, gpu_tile_size=16)
        # Find a leaf tile
        root = _unpack(Path(d) / "root.msgpack")
        first_child = root["children"][0]
        def get_leaf(name, path):
            t = _unpack(Path(path) / name)
            if t.get("is_leaf"):
                return t
            for c in t.get("children", []):
                l = get_leaf(c, path)
                if l:
                    return l
        leaf = get_leaf(first_child, d)
        assert leaf is not None
        assert "sub_tiles" in leaf
        assert len(leaf["sub_tiles"]) >= 1


def test_sub_tiles_max_size():
    """Each sub-tile shape must not exceed gpu_tile_size along split axes."""
    ds = make_2d_ds(nx=64, ny=64)
    gpu_size = 16
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=32, gpu_tile_size=gpu_size)
        def check_leaf(name, path):
            t = _unpack(Path(path) / name)
            if t.get("is_leaf"):
                for st in t.get("sub_tiles", []):
                    for dim_size in st["shape"]:
                        assert dim_size <= gpu_size, (
                            f"Sub-tile dimension {dim_size} > gpu_tile_size {gpu_size}"
                        )
            else:
                for c in t.get("children", []):
                    check_leaf(c, path)
        check_leaf("root.msgpack", d)


def test_sub_tiles_reconstruct():
    """Concatenating all sub-tile variable arrays reconstructs the leaf tile's data."""
    ds = make_2d_ds(nx=32, ny=32)
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=64, gpu_tile_size=8)
        root = _unpack(Path(d) / "root.msgpack")

        # Find first leaf
        def get_first_leaf(name, path):
            t = _unpack(Path(path) / name)
            if t.get("is_leaf"):
                return t
            for c in t.get("children", []):
                l = get_first_leaf(c, path)
                if l:
                    return l

        leaf = get_first_leaf("root.msgpack", d)
        assert "sub_tiles" in leaf

        # Collect all x and y coords from sub-tiles
        all_x = set()
        all_y = set()
        for st in leaf["sub_tiles"]:
            all_x.update(float(v) for v in st["spatial_coords"]["x"])
            all_y.update(float(v) for v in st["spatial_coords"]["y"])

        # Must match leaf's own spatial coords
        leaf_x = set(float(v) for v in leaf["spatial_coords"]["x"])
        leaf_y = set(float(v) for v in leaf["spatial_coords"]["y"])
        assert all_x == leaf_x, "Sub-tiles don't cover all x coords of leaf"
        assert all_y == leaf_y, "Sub-tiles don't cover all y coords of leaf"


def test_no_sub_tiles_when_not_requested():
    """When gpu_tile_size is None, no sub_tiles field should appear in leaf tiles."""
    ds = make_2d_ds()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16)
        def check_no_sub_tiles(name, path):
            t = _unpack(Path(path) / name)
            assert "sub_tiles" not in t, f"Unexpected sub_tiles in {name}"
            for c in t.get("children", []):
                check_no_sub_tiles(c, path)
        check_no_sub_tiles("root.msgpack", d)


def test_gpu_tile_size_in_metadata():
    """gpu_tile_size must appear in metadata when set."""
    ds = make_2d_ds()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], gpu_tile_size=64)
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert meta.get("gpu_tile_size") == 64


def test_sub_tiles_3d_geophysics():
    """GPU sub-tiles work correctly for a 3-D geophysics dataset (quadtree)."""
    ds = make_geophysics_ds(nx=32, ny=32, nz=8)
    gpu_size = 8
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"], max_leaf=16, gpu_tile_size=gpu_size)
        def check_leaf(name, path):
            t = _unpack(Path(path) / name)
            if t.get("is_leaf"):
                assert "sub_tiles" in t
                for st in t["sub_tiles"]:
                    # Only x and y are split axes; z sub-tiles can be any size
                    sc = st["spatial_coords"]
                    assert len(sc["x"]) <= gpu_size
                    assert len(sc["y"]) <= gpu_size
            else:
                for c in t.get("children", []):
                    check_leaf(c, path)
        check_leaf("root.msgpack", d)


# ─── Feature 3: Manifest tiles ────────────────────────────────────────────────

def test_manifest_directory_created():
    """write_manifest=True must create a manifest/ subdirectory."""
    ds = make_2d_ds()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16, write_manifest=True)
        assert (Path(d) / "manifest").is_dir()


def test_manifest_root_file_exists():
    """m_root.msgpack must exist in the manifest directory."""
    ds = make_2d_ds()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16, write_manifest=True)
        assert (Path(d) / "manifest" / "m_root.msgpack").exists()


def test_manifest_depth_in_metadata():
    """manifest_depth must appear in metadata when write_manifest=True."""
    ds = make_2d_ds()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16,
                       write_manifest=True, manifest_depth=4)
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert meta.get("manifest_depth") == 4


def test_manifest_depth_not_in_metadata_when_not_written():
    """manifest_depth should not be in metadata when write_manifest=False."""
    ds = make_2d_ds()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16, write_manifest=False)
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert "manifest_depth" not in meta


def test_manifest_bitmap_length():
    """Manifest tile bitmap length must match the BFS slot count."""
    manifest_depth = 3
    ds = make_2d_ds()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16,
                       write_manifest=True, manifest_depth=manifest_depth)
        m_root = _unpack(Path(d) / "manifest" / "m_root.msgpack")
        meta = _unpack(Path(d) / "metadata.msgpack")
        branch = 1 << len(meta["split_axes"])
        # Total slots: sum_{l=0}^{manifest_depth} branch^l = (branch^(D+1)-1)/(branch-1)
        expected_slots = sum(branch ** l for l in range(manifest_depth + 1))
        expected_bytes = (expected_slots + 7) // 8
        assert len(m_root["tiles"]) == expected_bytes


def test_manifest_bitmap_correctness():
    """Each existing tile's bit must be 1; non-existing slots must be 0."""
    manifest_depth = 2
    ds = make_2d_ds(nx=64, ny=64)
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=8,
                       write_manifest=True, manifest_depth=manifest_depth)
        meta     = _unpack(Path(d) / "metadata.msgpack")
        m_root   = _unpack(Path(d) / "manifest" / "m_root.msgpack")
        split_axes = meta["split_axes"]
        branch = 1 << len(split_axes)
        all_slots = _bfs_tile_names("root", branch, manifest_depth)

        for i, name in enumerate(all_slots):
            exists_on_disk = (Path(d) / name).exists()
            byte_i, bit_i = i >> 3, i & 7
            bit_set = bool(m_root["tiles"][byte_i] & (1 << bit_i))
            assert bit_set == exists_on_disk, (
                f"Slot {i} ({name}): disk={exists_on_disk}, bitmap={bit_set}"
            )


def test_manifest_depth_controls_slot_count():
    """Manifest tiles at depth D cover (branch^(D+1)-1)/(branch-1) slots."""
    ds = make_2d_ds()
    for manifest_depth in (2, 3, 4):
        with tempfile.TemporaryDirectory() as d:
            write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=4,
                           write_manifest=True, manifest_depth=manifest_depth)
            meta   = _unpack(Path(d) / "metadata.msgpack")
            m_root = _unpack(Path(d) / "manifest" / "m_root.msgpack")
            branch = 1 << len(meta["split_axes"])
            expected_slots = (branch ** (manifest_depth + 1) - 1) // (branch - 1)
            expected_bytes = (expected_slots + 7) // 8
            assert len(m_root["tiles"]) == expected_bytes, (
                f"depth={manifest_depth}: expected {expected_bytes} bytes, "
                f"got {len(m_root['tiles'])}"
            )


def test_bfs_tile_names_count():
    """_bfs_tile_names returns the correct number of names for branch/depth."""
    for branch in (2, 4, 8):
        for depth in (1, 2, 3):
            names = _bfs_tile_names("root", branch, depth)
            expected = (branch ** (depth + 1) - 1) // (branch - 1)
            assert len(names) == expected, (
                f"branch={branch}, depth={depth}: expected {expected} names, got {len(names)}"
            )


def test_bfs_tile_names_order():
    """BFS names must start with the root and grow level by level."""
    names = _bfs_tile_names("root", 4, 2)
    assert names[0] == "root.msgpack"
    # Level 1: indices 1–4
    for i in range(4):
        assert names[1 + i] == f"root_{i}.msgpack"
    # Level 2: indices 5–20
    for i in range(4):
        for j in range(4):
            assert names[5 + i * 4 + j] == f"root_{i}_{j}.msgpack"


def test_manifest_not_written_when_not_requested():
    """No manifest directory should be created when write_manifest=False."""
    ds = make_2d_ds()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], write_manifest=False)
        assert not (Path(d) / "manifest").exists()
