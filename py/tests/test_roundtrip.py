"""Python-only roundtrip tests for the webxtile library.

Tests write_webxtile / read_webxtile at full resolution, different bounding
boxes, and different level-of-detail levels for both 2-D and 3-D datasets.
"""
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr
import pytest

from webxtile import write_webxtile, read_webxtile
from webxtile import (
    _parse_epsg,
    _has_cf_grid_mapping,
    _detect_epsg_from_cf,
    _detect_z_epsg_from_cf,
    _cf_attrs_from_epsg,
    _resolve_crs_for_write,
)
from webxtile import _unpack

try:
    from pyproj import CRS as ProjCRS
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

pyproj_required = pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")


# ─── Dataset factories ────────────────────────────────────────────────────────

def make_2d_dataset(nx=64, ny=48):
    """Return a 64×48 2-D xarray Dataset with two variables."""
    x = np.linspace(0.0, 100.0, nx)
    y = np.linspace(0.0, 50.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    temperature  = (xx + yy * 0.5).astype(np.float32)
    precipitation = np.sin(xx / 20.0) * np.cos(yy / 10.0).astype(np.float32)
    return xr.Dataset(
        {
            "temperature":   (["x", "y"], temperature,  {"units": "K"}),
            "precipitation": (["x", "y"], precipitation, {"units": "mm/d"}),
        },
        coords={"x": x, "y": y},
        attrs={"source": "synthetic_test", "version": 1},
    )


def make_3d_dataset(nx=32, ny=24, nz=16):
    """Return a 32×24×16 3-D xarray Dataset."""
    x = np.linspace(0.0, 100.0, nx)
    y = np.linspace(0.0, 50.0, ny)
    z = np.linspace(0.0, 20.0, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    values = (xx + yy * 0.5 + zz * 0.1).astype(np.float32)
    return xr.Dataset(
        {"temperature": (["x", "y", "z"], values, {"units": "K"})},
        coords={"x": x, "y": y, "z": z},
        attrs={"source": "synthetic_3d"},
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _assert_coords_equal(ds_orig, ds_read, dims):
    for dim in dims:
        np.testing.assert_array_equal(
            ds_read[dim].values, ds_orig[dim].values,
            err_msg=f"coordinate '{dim}' differs after roundtrip",
        )


def _assert_data_equal(ds_orig, ds_read, decimal=4):
    for var in ds_orig.data_vars:
        np.testing.assert_array_almost_equal(
            ds_read[var].values, ds_orig[var].values,
            decimal=decimal,
            err_msg=f"variable '{var}' differs after roundtrip",
        )


# ─── 2-D tests ────────────────────────────────────────────────────────────────

def test_full_roundtrip_2d():
    """Full-resolution read must reproduce the original Dataset exactly."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"])
        ds2 = read_webxtile(d)  # level=None → all leaf tiles

    _assert_coords_equal(ds, ds2, ["x", "y"])
    _assert_data_equal(ds, ds2)
    assert ds2.attrs == ds.attrs


def test_variable_attrs_preserved():
    """Variable and coordinate attributes must survive the roundtrip."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"])
        ds2 = read_webxtile(d)

    for var in ds.data_vars:
        assert ds2[var].attrs == ds[var].attrs, (
            f"attrs mismatch for variable '{var}'"
        )


def test_xarray_accessor():
    """ds.webxtile.to_webxtile() must produce the same result as write_webxtile."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        write_webxtile(ds, d1, spatial_dims=["x", "y"])
        ds.webxtile.to_webxtile(d2)

        ds1 = read_webxtile(d1)
        ds2 = read_webxtile(d2)

    _assert_coords_equal(ds1, ds2, ["x", "y"])
    for var in ds.data_vars:
        np.testing.assert_array_equal(ds1[var].values, ds2[var].values)


def test_bbox_read_returns_subset():
    """BBox-filtered read must return only tiles whose bounds intersect the box."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16)

        # A box that covers roughly the centre quarter of the domain
        x0, x1 = 25.0, 75.0
        y0, y1 = 12.5, 37.5
        ds_bbox = read_webxtile(d, bbox=[x0, y0, x1, y1])

    # The returned x / y ranges must be no wider than the full dataset
    assert ds_bbox["x"].values.min() >= ds["x"].values.min()
    assert ds_bbox["x"].values.max() <= ds["x"].values.max()
    assert ds_bbox["y"].values.min() >= ds["y"].values.min()
    assert ds_bbox["y"].values.max() <= ds["y"].values.max()

    # Must be strictly smaller (or at most equal if tiles happened to span
    # the whole domain)
    assert len(ds_bbox["x"]) <= len(ds["x"])
    assert len(ds_bbox["y"]) <= len(ds["y"])


def test_bbox_values_match_original():
    """Points that appear in both the full and bbox reads must have identical values."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16)

        x0, x1 = 30.0, 70.0
        y0, y1 = 10.0, 40.0
        ds_bbox = read_webxtile(d, bbox=[x0, y0, x1, y1])

    # Find x/y coords common to both reads
    x_common = np.intersect1d(ds["x"].values, ds_bbox["x"].values)
    y_common = np.intersect1d(ds["y"].values, ds_bbox["y"].values)
    assert len(x_common) > 0 and len(y_common) > 0, (
        "no common coordinates found – check bbox or dataset definition"
    )

    xi_orig = np.isin(ds["x"].values, x_common)
    yi_orig = np.isin(ds["y"].values, y_common)
    xi_bbox = np.isin(ds_bbox["x"].values, x_common)
    yi_bbox = np.isin(ds_bbox["y"].values, y_common)

    for var in ds.data_vars:
        orig_slice = ds[var].values[np.ix_(xi_orig, yi_orig)]
        bbox_slice = ds_bbox[var].values[np.ix_(xi_bbox, yi_bbox)]
        np.testing.assert_array_almost_equal(
            bbox_slice, orig_slice, decimal=4,
            err_msg=f"variable '{var}': bbox values differ from original",
        )


def test_level_0_is_coarsest():
    """Reading at level 0 (root tile) must give fewer points than full res."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=8)
        ds_root = read_webxtile(d, level=0)
        ds_full = read_webxtile(d)

    assert len(ds_root["x"]) < len(ds_full["x"])
    assert len(ds_root["y"]) < len(ds_full["y"])


def test_levels_are_monotonically_coarser():
    """Higher levels return finer data; level N+1 must be at least as fine as N."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=8)
        sizes = []
        for lv in range(4):
            ds_lv = read_webxtile(d, level=lv)
            sizes.append(len(ds_lv["x"]) * len(ds_lv["y"]))

    for i in range(len(sizes) - 1):
        assert sizes[i] <= sizes[i + 1], (
            f"level {i} returned {sizes[i]} points but level {i+1} returned "
            f"{sizes[i+1]} — expected non-decreasing"
        )


def test_bbox_and_level_combined():
    """Combining bbox + level must return a spatially filtered coarse overview."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=8)

        x0, x1 = 20.0, 80.0
        y0, y1 = 10.0, 40.0
        ds_bbox_full = read_webxtile(d, bbox=[x0, y0, x1, y1])
        ds_bbox_l0   = read_webxtile(d, bbox=[x0, y0, x1, y1], level=0)

    # BBox+level=0 must have ≤ points as BBox+full-res
    total_full = len(ds_bbox_full["x"]) * len(ds_bbox_full["y"])
    total_l0   = len(ds_bbox_l0["x"])   * len(ds_bbox_l0["y"])
    assert total_l0 <= total_full


def test_different_bboxes_cover_different_regions():
    """Two non-overlapping bboxes must return data centred in different regions.

    With nx=64, max_leaf=16, x in [0,100] each leaf tile spans ~24.6 x-units.
    Using bboxes [0,40] and [60,100] for x guarantees they touch different
    halves; the mean x-coordinate must reflect this.
    """
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], max_leaf=16)

        ds_left  = read_webxtile(d, bbox=[0.0,  0.0,  40.0, 50.0])
        ds_right = read_webxtile(d, bbox=[60.0, 0.0, 100.0, 50.0])

    assert ds_left["x"].values.mean() < ds_right["x"].values.mean()


# ─── 3-D tests ────────────────────────────────────────────────────────────────

def test_full_roundtrip_3d():
    """Full-resolution roundtrip for a 3-D dataset."""
    ds = make_3d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"])
        ds2 = read_webxtile(d)

    _assert_coords_equal(ds, ds2, ["x", "y", "z"])
    _assert_data_equal(ds, ds2)
    assert ds2.attrs == ds.attrs


def test_bbox_read_3d():
    """BBox filtering in 3-D returns only tiles that intersect the sub-volume.

    With nx=32, max_leaf=8 and x in [0,100] each leaf tile spans ~22.6 x-units.
    A bbox with x=[25,75] excludes the leftmost tile (x_max≈22.6) and the
    rightmost tile (x_min≈77.4), so fewer unique x values are returned.
    """
    ds = make_3d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"], max_leaf=8)

        # Narrow enough to exclude at least the boundary tiles on every axis
        ds_sub = read_webxtile(d, bbox=[25.0, 15.0, 5.0, 75.0, 35.0, 15.0])

    assert len(ds_sub["x"]) < len(ds["x"])
    assert len(ds_sub["y"]) < len(ds["y"])


def test_level_0_3d():
    """Level 0 of a 3-D dataset must be coarser than full resolution."""
    ds = make_3d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"], max_leaf=4)
        ds_root = read_webxtile(d, level=0)
        ds_full = read_webxtile(d)

    n_root = len(ds_root["x"]) * len(ds_root["y"]) * len(ds_root["z"])
    n_full = len(ds_full["x"])  * len(ds_full["y"])  * len(ds_full["z"])
    assert n_root < n_full


# ─── CRS / EPSG helper unit tests ────────────────────────────────────────────

def test_parse_epsg_normalisation():
    """_parse_epsg normalises various input forms to 'EPSG:NNNN'."""
    assert _parse_epsg("EPSG:4326") == "EPSG:4326"
    assert _parse_epsg("epsg:4326") == "EPSG:4326"
    assert _parse_epsg(4326)        == "EPSG:4326"
    assert _parse_epsg("4326")      == "EPSG:4326"
    assert _parse_epsg(None)        is None
    # Non-numeric identifiers pass through unchanged
    assert _parse_epsg("WGS84")     == "WGS84"


def test_has_cf_grid_mapping_positive():
    """_has_cf_grid_mapping returns True when a coord has grid_mapping_name."""
    ds = make_2d_dataset()
    ds2 = ds.assign_coords(
        crs_var=xr.Variable([], 0, attrs={"grid_mapping_name": "latitude_longitude"})
    )
    assert _has_cf_grid_mapping(ds2) is True


def test_has_cf_grid_mapping_negative():
    """_has_cf_grid_mapping returns False when no grid_mapping_name present."""
    ds = make_2d_dataset()
    assert _has_cf_grid_mapping(ds) is False


@pyproj_required
def test_detect_epsg_from_global_attr():
    """EPSG detected from non-standard 'epsg_code' global attribute."""
    ds = make_2d_dataset()
    ds.attrs["epsg_code"] = "EPSG:32633"
    result = _detect_epsg_from_cf(ds, ProjCRS)
    assert result == "EPSG:32633"


@pyproj_required
def test_detect_epsg_from_global_attr_bare_int():
    """Bare integer epsg_code attribute is normalised to 'EPSG:NNNN'."""
    ds = make_2d_dataset()
    ds.attrs["epsg_code"] = 4326
    result = _detect_epsg_from_cf(ds, ProjCRS)
    assert result == "EPSG:4326"


@pyproj_required
def test_detect_epsg_from_grid_mapping_variable():
    """EPSG detected from a CF grid_mapping coordinate variable via pyproj."""
    ds = make_2d_dataset()
    wgs84_cf = ProjCRS.from_epsg(4326).to_cf()
    ds2 = ds.assign_coords(
        crs_var=xr.Variable([], 0, attrs=wgs84_cf)
    )
    result = _detect_epsg_from_cf(ds2, ProjCRS)
    assert result == "EPSG:4326"


@pyproj_required
def test_detect_epsg_from_longitude_latitude_standard_names():
    """EPSG detected when coordinate standard_names are longitude / latitude."""
    ds = make_2d_dataset()
    ds["x"].attrs["standard_name"] = "longitude"
    ds["y"].attrs["standard_name"] = "latitude"
    result = _detect_epsg_from_cf(ds, ProjCRS)
    assert result is not None
    assert "4326" in result


@pyproj_required
def test_detect_epsg_returns_none_without_cf_info():
    """Returns None when dataset has no CRS metadata."""
    ds = make_2d_dataset()  # plain dataset, no CRS info
    result = _detect_epsg_from_cf(ds, ProjCRS)
    assert result is None


@pyproj_required
def test_detect_z_epsg_from_global_attr():
    """Vertical EPSG detected from non-standard 'epsg_z_code' global attribute."""
    ds = make_3d_dataset()
    ds.attrs["epsg_z_code"] = "EPSG:5773"
    result = _detect_z_epsg_from_cf(ds, ProjCRS)
    assert result == "EPSG:5773"


@pyproj_required
def test_cf_attrs_from_epsg_wgs84():
    """CF attrs for EPSG:4326 include grid_mapping_name='latitude_longitude'."""
    attrs = _cf_attrs_from_epsg("EPSG:4326", ProjCRS)
    assert "grid_mapping_name" in attrs
    assert attrs["grid_mapping_name"] == "latitude_longitude"


@pyproj_required
def test_cf_attrs_from_epsg_projected():
    """CF attrs for a projected CRS include a non-geographic grid_mapping_name.

    Uses EPSG:32632 (UTM zone 32N), which pyproj maps to CF ``transverse_mercator``.
    EPSG:3857 (Pseudo-Mercator) is deliberately avoided because some pyproj
    versions can only export it as ``crs_wkt`` without a CF grid_mapping_name.
    """
    attrs = _cf_attrs_from_epsg("EPSG:32632", ProjCRS)
    assert "grid_mapping_name" in attrs
    assert attrs["grid_mapping_name"] == "transverse_mercator"


@pyproj_required
def test_cf_attrs_from_epsg_invalid():
    """Invalid EPSG code returns an empty dict without raising."""
    attrs = _cf_attrs_from_epsg("EPSG:999999999", ProjCRS)
    assert attrs == {}


@pyproj_required
def test_resolve_crs_for_write_detects_from_cf():
    """_resolve_crs_for_write detects EPSG from CF metadata when crs=None."""
    ds = make_2d_dataset()
    ds.attrs["epsg_code"] = "EPSG:4326"
    crs, z_crs, cf_attrs, z_cf_attrs = _resolve_crs_for_write(ds, None, None)
    assert crs == "EPSG:4326"
    assert z_crs is None
    assert "grid_mapping_name" in cf_attrs


@pyproj_required
def test_resolve_crs_for_write_explicit_overrides_attr():
    """Explicit crs parameter takes precedence over dataset epsg_code attribute."""
    ds = make_2d_dataset()
    ds.attrs["epsg_code"] = "EPSG:4326"
    crs, z_crs, cf_attrs, _ = _resolve_crs_for_write(ds, "EPSG:3857", None)
    assert crs == "EPSG:3857"
    assert cf_attrs.get("grid_mapping_name") != "latitude_longitude"


# ─── Integration: CRS stored in metadata and roundtripped ────────────────────

@pyproj_required
def test_epsg_stored_as_metadata_field():
    """crs and z_crs top-level fields are written to metadata.msgpack."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], crs="EPSG:4326")
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert meta["crs"] == "EPSG:4326"
    assert meta["z_crs"] is None


@pyproj_required
def test_epsg_code_in_global_attrs():
    """epsg_code is written as a non-standard attribute in global_attrs."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], crs="EPSG:3857")
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert meta["global_attrs"]["epsg_code"] == "EPSG:3857"


@pyproj_required
def test_epsg_code_visible_on_read():
    """epsg_code attribute is accessible on the reconstructed Dataset."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], crs="EPSG:4326")
        ds2 = read_webxtile(d)
    assert ds2.attrs.get("epsg_code") == "EPSG:4326"


@pyproj_required
def test_z_epsg_code_in_global_attrs():
    """epsg_z_code is written as a non-standard attribute in global_attrs."""
    ds = make_3d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y", "z"], z_crs="EPSG:5773")
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert meta["global_attrs"]["epsg_z_code"] == "EPSG:5773"
    assert meta["z_crs"] == "EPSG:5773"


@pyproj_required
def test_crs_cf_attrs_stored_in_metadata():
    """crs_cf_attrs dict is stored in metadata when crs is provided."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], crs="EPSG:4326")
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert "crs_cf_attrs" in meta
    assert meta["crs_cf_attrs"].get("grid_mapping_name") == "latitude_longitude"


@pyproj_required
def test_spatial_ref_coord_added_on_read():
    """A spatial_ref grid-mapping coordinate is added to the reconstructed Dataset."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], crs="EPSG:4326")
        ds2 = read_webxtile(d)
    assert "spatial_ref" in ds2.coords
    assert ds2["spatial_ref"].attrs.get("grid_mapping_name") == "latitude_longitude"


@pyproj_required
def test_spatial_ref_not_duplicated_when_already_present():
    """spatial_ref is not duplicated if the original dataset already had one."""
    ds = make_2d_dataset()
    wgs84_cf = ProjCRS.from_epsg(4326).to_cf()
    ds = ds.assign_coords(
        spatial_ref=xr.Variable([], 0, attrs=wgs84_cf)
    )
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"])
        ds2 = read_webxtile(d)
    # Only one spatial_ref should exist
    assert "spatial_ref" in ds2.coords
    assert ds2["spatial_ref"].attrs.get("grid_mapping_name") == "latitude_longitude"


@pyproj_required
def test_auto_detect_epsg_from_cf_on_write():
    """When crs not specified, EPSG is auto-detected from CF metadata."""
    ds = make_2d_dataset()
    ds.attrs["epsg_code"] = "EPSG:32632"  # UTM zone 32N
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"])
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert meta["crs"] == "EPSG:32632"
    assert "grid_mapping_name" in meta["crs_cf_attrs"]


@pyproj_required
def test_cf_attrs_populated_from_epsg_when_no_cf_in_dataset():
    """CF attributes are generated from EPSG when dataset has no grid_mapping."""
    ds = make_2d_dataset()  # no CF CRS metadata
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"], crs="EPSG:4326")
        meta = _unpack(Path(d) / "metadata.msgpack")
    cf = meta["crs_cf_attrs"]
    assert cf.get("grid_mapping_name") == "latitude_longitude"
    # pyproj populates geodetic parameters
    assert "semi_major_axis" in cf or "crs_wkt" in cf


@pyproj_required
def test_no_epsg_no_crs_cf_attrs():
    """When no CRS is specified or detectable, crs_cf_attrs is empty."""
    ds = make_2d_dataset()
    with tempfile.TemporaryDirectory() as d:
        write_webxtile(ds, d, spatial_dims=["x", "y"])
        meta = _unpack(Path(d) / "metadata.msgpack")
    assert meta.get("crs") is None
    assert meta.get("crs_cf_attrs", {}) == {}


@pyproj_required
def test_existing_epsg_attr_not_overwritten():
    """If the dataset already has epsg_code in attrs, it is not overwritten."""
    ds = make_2d_dataset()
    ds.attrs["epsg_code"] = "EPSG:4326"
    with tempfile.TemporaryDirectory() as d:
        # Pass a different crs parameter
        write_webxtile(ds, d, spatial_dims=["x", "y"], crs="EPSG:3857")
        meta = _unpack(Path(d) / "metadata.msgpack")
    # The parameter wins at the crs field level...
    assert meta["crs"] == "EPSG:3857"
    # ...but the pre-existing epsg_code attr in ds.attrs is preserved as-is
    assert meta["global_attrs"]["epsg_code"] == "EPSG:4326"
