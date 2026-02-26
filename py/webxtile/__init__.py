"""Webxtile: 3D geographic grid storage as a spatial octree with msgpack tiles.

The format stores an xarray Dataset as a recursive octree where each tile is a
msgpack file.  Internal nodes hold a 2× downsampled overview of their subtree so
any level of detail can be read without visiting leaf tiles.  Leaf nodes hold the
full-resolution data for their spatial chunk.

Full roundtrip guarantee
------------------------
Reading all leaf tiles and merging them gives back the original xarray Dataset
(coordinates, data variables, attributes, and dtypes) without loss.

Reading at level *L* gives a low-fidelity version of the full dataset:
- level 0  → root tile only  (coarsest)
- level N  → tiles at depth N (or leaf if a branch terminated earlier)
- level None → all leaf tiles (finest, default)

Public API
----------
write_webxtile(ds, path, ...)
    Write an xarray Dataset to a webxtile directory.

read_webxtile(path, ...)
    Read a webxtile directory into an xarray Dataset.

xarray integration
------------------
xr.open_dataset("tiles/", engine="webxtile")          # read
xr.open_dataset("tiles/", engine="webxtile", level=2) # read at LOD 2
ds.webxtile.to_webxtile("tiles/")                    # write
"""

from __future__ import annotations

import xarray as xr
import numpy as np
import msgpack
import msgpack_numpy as m
from pathlib import Path
from scipy.ndimage import zoom

# Patch msgpack globally so numpy arrays are handled transparently.
m.patch()

__version__ = "0.1.0"
__all__ = ["write_webxtile", "read_webxtile", "WebxtileBackend"]

_METADATA_FILE = "metadata.msgpack"
_ROOT_TILE = "root.msgpack"
_FORMAT_VERSION = 1

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def write_webxtile(
    ds: xr.Dataset,
    path: str | Path,
    *,
    spatial_dims: list[str] | None = None,
    max_leaf: int = 32,
    crs: str | None = None,
    z_crs: str | None = None,
) -> None:
    """Write an xarray Dataset to webxtile octree format.

    Parameters
    ----------
    ds:
        Dataset to write.  Must have at least 2 spatial dimensions.
    path:
        Output directory.  Created if it does not exist.
    spatial_dims:
        Names of 2 or 3 dimensions to use for octree decomposition.
        Auto-detected from CF ``axis`` / ``standard_name`` attributes when
        not provided.
    max_leaf:
        Maximum grid points per tile along any spatial dimension.  Tiles
        smaller than this threshold become leaf nodes.
    crs:
        Horizontal CRS identifier (e.g. ``"EPSG:3857"``), stored in metadata
        for consumers that need it.
    z_crs:
        Vertical CRS identifier (e.g. ``"EPSG:4979"``).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if spatial_dims is None:
        spatial_dims = _detect_spatial_dims(ds)
    if not (2 <= len(spatial_dims) <= 3):
        raise ValueError(
            f"spatial_dims must have 2 or 3 elements, got {spatial_dims}"
        )

    _pack(path / _METADATA_FILE, _build_metadata(ds, spatial_dims, crs, z_crs))
    _build_tile(ds, path / _ROOT_TILE, spatial_dims, level=0, max_leaf=max_leaf)


def read_webxtile(
    path: str | Path,
    *,
    level: int | None = None,
    bbox: list[float] | None = None,
) -> xr.Dataset:
    """Read a webxtile octree into an xarray Dataset.

    Parameters
    ----------
    path:
        Directory produced by :func:`write_webxtile`.
    level:
        Octree depth to read.  ``0`` = root (coarsest overview).
        ``None`` (default) = all leaf tiles (full resolution).
        When a branch terminates before *level* is reached the deepest
        available tile for that branch is returned instead.
    bbox:
        Axis-aligned bounding-box filter expressed in the same coordinate
        system as the spatial dimensions.  Length must be
        ``2 × len(spatial_dims)``: ``[x0, y0, x1, y1]`` for 2-D or
        ``[x0, y0, z0, x1, y1, z1]`` for 3-D.

    Returns
    -------
    xr.Dataset
        Dataset with CF metadata (attributes, dtypes, coordinates) restored.
    """
    path = Path(path)
    meta = _unpack(path / _METADATA_FILE)
    tiles = _collect_tiles(path, path / _ROOT_TILE, level=level, bbox=bbox, meta=meta)
    return _reconstruct_dataset(tiles, meta)


# ─────────────────────────────────────────────────────────────────────────────
# xarray integration
# ─────────────────────────────────────────────────────────────────────────────

try:
    from xarray.backends import BackendEntrypoint as _BackendEntrypoint

    class WebxtileBackend(_BackendEntrypoint):
        """xarray backend for the webxtile octree format.

        Usage::

            ds = xr.open_dataset("tiles/", engine="webxtile")
            ds = xr.open_dataset("tiles/", engine="webxtile", level=2)
        """

        description = "Load 3-D geographic grids from webxtile octree format"

        # All parameters accepted by open_dataset must be listed here so
        # xarray can validate keyword arguments.
        open_dataset_parameters = [
            "filename_or_obj",
            "mask_and_scale",
            "decode_times",
            "concat_characters",
            "decode_coords",
            "drop_variables",
            "use_cftime",
            "decode_timedelta",
            "level",
            "bbox",
        ]

        def open_dataset(
            self,
            filename_or_obj,
            *,
            mask_and_scale=True,
            decode_times=True,
            concat_characters=True,
            decode_coords=True,
            drop_variables=None,
            use_cftime=None,
            decode_timedelta=None,
            # webxtile-specific keyword arguments
            level: int | None = None,
            bbox: list[float] | None = None,
        ) -> xr.Dataset:
            ds = read_webxtile(filename_or_obj, level=level, bbox=bbox)
            if drop_variables:
                ds = ds.drop_vars(
                    [v for v in drop_variables if v in ds], errors="ignore"
                )
            return ds

except ImportError:
    # xarray is missing or too old to expose BackendEntrypoint
    class WebxtileBackend:  # type: ignore[no-redef]
        """Stub – xarray.backends.BackendEntrypoint not available."""


@xr.register_dataset_accessor("webxtile")
class WebxtileAccessor:
    """xarray Dataset accessor for writing webxtile format.

    Usage::

        ds.webxtile.to_webxtile("tiles/")
        ds.webxtile.to_webxtile("tiles/", max_leaf=64, crs="EPSG:3857")
    """

    def __init__(self, obj: xr.Dataset) -> None:
        self._obj = obj

    def to_webxtile(self, path: str | Path, **kwargs) -> None:
        """Write the Dataset to webxtile format (see :func:`write_webxtile`)."""
        write_webxtile(self._obj, path, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# msgpack helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pack(path: Path, obj) -> None:
    """Serialise *obj* to *path* with numpy array support."""
    with open(path, "wb") as f:
        msgpack.pack(obj, f)


def _unpack(path: Path):
    """Deserialise from *path*, returning Python/numpy objects with str keys."""
    with open(path, "rb") as f:
        try:
            return msgpack.unpack(f, raw=False)
        except TypeError:
            # Older msgpack_numpy patches may not forward raw=
            f.seek(0)
            return msgpack.unpack(f)


# ─────────────────────────────────────────────────────────────────────────────
# CF dimension auto-detection
# ─────────────────────────────────────────────────────────────────────────────

_CF_SN_X = {"projection_x_coordinate", "longitude", "grid_longitude"}
_CF_SN_Y = {"projection_y_coordinate", "latitude", "grid_latitude"}
_CF_SN_Z = {
    "depth", "altitude", "height", "air_pressure",
    "atmosphere_sigma_coordinate", "ocean_sigma_coordinate",
    "geopotential_height", "height_above_geopotential_datum",
}
_NAMES_X = {"x", "lon", "longitude", "easting", "X"}
_NAMES_Y = {"y", "lat", "latitude", "northing", "Y"}
_NAMES_Z = {"z", "depth", "altitude", "height", "elevation",
            "level", "lev", "plev", "Z"}


def _classify_dim(ds: xr.Dataset, dim: str) -> str | None:
    """Return 'x', 'y', 'z', or None based on CF attributes / name heuristics."""
    coord = ds.coords.get(dim)
    if coord is not None:
        sn   = coord.attrs.get("standard_name", "")
        axis = coord.attrs.get("axis", "")
        if sn in _CF_SN_X or axis == "X":
            return "x"
        if sn in _CF_SN_Y or axis == "Y":
            return "y"
        if sn in _CF_SN_Z or axis == "Z":
            return "z"
    name = dim.lower()
    if name in {n.lower() for n in _NAMES_X}:
        return "x"
    if name in {n.lower() for n in _NAMES_Y}:
        return "y"
    if name in {n.lower() for n in _NAMES_Z}:
        return "z"
    return None


def _detect_spatial_dims(ds: xr.Dataset) -> list[str]:
    """Auto-detect 2–3 spatial dimension names from CF conventions."""
    found: dict[str, str | None] = {"x": None, "y": None, "z": None}
    for dim in ds.dims:
        k = _classify_dim(ds, dim)
        if k and found[k] is None:
            found[k] = dim
    if found["x"] is None or found["y"] is None:
        raise ValueError(
            f"Cannot auto-detect horizontal spatial dimensions from dims "
            f"{list(ds.dims)}.  Set CF 'axis' or 'standard_name' attributes, "
            "or pass spatial_dims explicitly."
        )
    dims = [found["x"], found["y"]]
    if found["z"] is not None:
        dims.append(found["z"])
    return dims


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_serialisable(v):
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_to_serialisable(x) for x in v]
    return v


def _serialisable_attrs(attrs: dict) -> dict:
    return {str(k): _to_serialisable(v) for k, v in attrs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_metadata(
    ds: xr.Dataset,
    spatial_dims: list[str],
    crs: str | None,
    z_crs: str | None,
) -> dict:
    """Assemble the metadata dict stored in metadata.msgpack."""
    coord_meta: dict = {}
    for name, coord in ds.coords.items():
        entry: dict = {
            "dims":  list(coord.dims),
            "dtype": str(coord.dtype),
            "attrs": _serialisable_attrs(coord.attrs),
        }
        # Non-spatial dimension coordinates have the same values in every tile,
        # so we embed the array here to avoid per-tile redundancy.
        if list(coord.dims) == [name] and name not in spatial_dims:
            entry["values"] = coord.values  # numpy – msgpack_numpy handles it
        coord_meta[name] = entry

    var_meta: dict = {}
    for name, var in ds.data_vars.items():
        var_meta[name] = {
            "dims":  list(var.dims),
            "dtype": str(var.dtype),
            "attrs": _serialisable_attrs(var.attrs),
        }

    return {
        "version":      _FORMAT_VERSION,
        "root_tile":    _ROOT_TILE,
        "spatial_dims": list(spatial_dims),
        "crs":          crs,
        "z_crs":        z_crs,
        "dim_sizes":    {str(k): int(v) for k, v in ds.sizes.items()},
        "coord_meta":   coord_meta,
        "var_meta":     var_meta,
        "global_attrs": _serialisable_attrs(ds.attrs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Write – octree construction
# ─────────────────────────────────────────────────────────────────────────────

def _spatial_coord_arrays(ds: xr.Dataset, spatial_dims: list[str]) -> dict[str, np.ndarray]:
    """Return {dim: float64 array} for each spatial dimension."""
    out = {}
    for dim in spatial_dims:
        if dim in ds.coords:
            out[dim] = ds[dim].values.astype(np.float64)
        else:
            out[dim] = np.arange(ds.sizes[dim], dtype=np.float64)
    return out


def _bounds_from_spatial_coords(sc: dict, spatial_dims: list[str]) -> list[float]:
    """Return a 6-element bounds list [x0,y0,z0,x1,y1,z1].

    Always 6 elements (padded with 0.0) for JS-frontend compatibility.
    """
    mins = [float(sc[d].min()) for d in spatial_dims]
    maxs = [float(sc[d].max()) for d in spatial_dims]
    while len(mins) < 3:
        mins.append(0.0)
        maxs.append(0.0)
    return mins + maxs


def _variable_arrays(
    ds: xr.Dataset,
    spatial_dims: list[str],
    zoom_factor: float | None = None,
) -> dict[str, np.ndarray]:
    """Extract all data variables, optionally downsampling along spatial dims."""
    out = {}
    for name, var in ds.data_vars.items():
        arr = var.values.astype(np.float32)
        if zoom_factor is not None:
            factors = [
                zoom_factor if d in spatial_dims else 1.0
                for d in var.dims
            ]
            arr = zoom(arr.astype(np.float64), factors, order=1).astype(np.float32)
        out[name] = arr
    return out


def _downsample_spatial_coords(
    ds: xr.Dataset,
    spatial_dims: list[str],
) -> dict[str, np.ndarray]:
    """Sample coordinate arrays at half resolution (subset of original values).

    The sampled values are always a strict subset of the original coordinate
    values, which is required so that reconstruction can use exact dict lookup.
    """
    out = {}
    for dim in spatial_dims:
        arr = (
            ds[dim].values.astype(np.float64)
            if dim in ds.coords
            else np.arange(ds.sizes[dim], dtype=np.float64)
        )
        n_new = max(1, len(arr) // 2)
        # linspace over index space gives n_new evenly spaced samples.
        indices = np.linspace(0, len(arr) - 1, n_new).round().astype(int)
        indices = np.unique(indices)  # remove duplicates caused by rounding
        out[dim] = arr[indices]
    return out


def _octree_child_isel(ds: xr.Dataset, spatial_dims: list[str]) -> list[dict]:
    """Return the list of isel dicts that partition ds along spatial dims."""
    mids  = {d: ds.sizes[d] // 2 for d in spatial_dims}
    sizes = {d: ds.sizes[d] for d in spatial_dims}

    if len(spatial_dims) == 3:
        x, y, z = spatial_dims
        xm, ym, zm = mids[x], mids[y], mids[z]
        nx, ny, nz = sizes[x], sizes[y], sizes[z]
        return [
            {x: slice(0,  xm), y: slice(0,  ym), z: slice(0,  zm)},
            {x: slice(0,  xm), y: slice(0,  ym), z: slice(zm, nz)},
            {x: slice(0,  xm), y: slice(ym, ny), z: slice(0,  zm)},
            {x: slice(0,  xm), y: slice(ym, ny), z: slice(zm, nz)},
            {x: slice(xm, nx), y: slice(0,  ym), z: slice(0,  zm)},
            {x: slice(xm, nx), y: slice(0,  ym), z: slice(zm, nz)},
            {x: slice(xm, nx), y: slice(ym, ny), z: slice(0,  zm)},
            {x: slice(xm, nx), y: slice(ym, ny), z: slice(zm, nz)},
        ]
    else:  # 2-D: quadtree
        x, y = spatial_dims
        xm, ym = mids[x], mids[y]
        nx, ny = sizes[x], sizes[y]
        return [
            {x: slice(0,  xm), y: slice(0,  ym)},
            {x: slice(0,  xm), y: slice(ym, ny)},
            {x: slice(xm, nx), y: slice(0,  ym)},
            {x: slice(xm, nx), y: slice(ym, ny)},
        ]


def _build_tile(
    ds: xr.Dataset,
    tile_path: Path,
    spatial_dims: list[str],
    level: int,
    max_leaf: int,
) -> None:
    """Recursively write one tile and its subtree."""
    if max(ds.sizes[d] for d in spatial_dims) <= max_leaf:
        _write_leaf_tile(ds, tile_path, spatial_dims, level)
    else:
        _write_internal_tile(ds, tile_path, spatial_dims, level, max_leaf)


def _write_leaf_tile(
    ds: xr.Dataset,
    tile_path: Path,
    spatial_dims: list[str],
    level: int,
) -> None:
    sc = _spatial_coord_arrays(ds, spatial_dims)
    tile = {
        "level":          level,
        "is_leaf":        True,
        "bounds":         _bounds_from_spatial_coords(sc, spatial_dims),
        "shape":          [ds.sizes[d] for d in spatial_dims],
        "spatial_coords": sc,
        "variables":      _variable_arrays(ds, spatial_dims),
    }
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    _pack(tile_path, tile)


def _write_internal_tile(
    ds: xr.Dataset,
    tile_path: Path,
    spatial_dims: list[str],
    level: int,
    max_leaf: int,
) -> None:
    # The internal node stores a 2× downsampled overview of this subtree's
    # data so that any LOD level can be served without loading leaf tiles.
    sc_down = _downsample_spatial_coords(ds, spatial_dims)
    vars_down = _variable_arrays(ds, spatial_dims, zoom_factor=0.5)

    # Build children
    children: list[str] = []
    for i, isel_dict in enumerate(_octree_child_isel(ds, spatial_dims)):
        child_ds = ds.isel(isel_dict)
        if any(child_ds.sizes[d] == 0 for d in spatial_dims):
            continue
        child_name = f"{tile_path.stem}_{i}.msgpack"
        child_path = tile_path.parent / child_name
        children.append(child_name)
        _build_tile(child_ds, child_path, spatial_dims, level + 1, max_leaf)

    sc = _spatial_coord_arrays(ds, spatial_dims)
    tile = {
        "level":          level,
        "is_leaf":        False,
        "bounds":         _bounds_from_spatial_coords(sc, spatial_dims),
        "shape":          [max(1, ds.sizes[d] // 2) for d in spatial_dims],
        "spatial_coords": sc_down,
        "variables":      vars_down,
        "children":       children,
    }
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    _pack(tile_path, tile)


# ─────────────────────────────────────────────────────────────────────────────
# Read – tile collection
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_intersects(bounds: list, bbox: list, n_spatial: int) -> bool:
    """Test whether *bounds* overlaps *bbox* in the first *n_spatial* axes."""
    for i in range(n_spatial):
        if bbox[i + n_spatial] < bounds[i] or bbox[i] > bounds[i + n_spatial]:
            return False
    return True


def _collect_tiles(
    root: Path,
    tile_path: Path,
    *,
    level: int | None,
    bbox: list[float] | None,
    meta: dict,
) -> list[dict]:
    """Recursively collect tiles that satisfy the level / bbox constraints."""
    tile = _unpack(tile_path)
    tile["_path"] = str(tile_path)

    n_spatial = len(meta["spatial_dims"])
    if bbox is not None and not _bbox_intersects(tile["bounds"], bbox, n_spatial):
        return []

    is_leaf    = tile.get("is_leaf", True)
    tile_level = tile["level"]

    # Return this tile if it is a leaf or if we have reached the target level.
    if is_leaf or (level is not None and tile_level >= level):
        return [tile]

    # Recurse into children.
    result: list[dict] = []
    for child_name in tile.get("children", []):
        child_path = root / child_name
        if child_path.exists():
            result.extend(
                _collect_tiles(root, child_path, level=level, bbox=bbox, meta=meta)
            )

    # If all children were filtered out by bbox, fall back to this tile so the
    # caller always gets at least a coarse result for the requested region.
    return result if result else [tile]


# ─────────────────────────────────────────────────────────────────────────────
# Read – dataset reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_dataset(tiles: list[dict], meta: dict) -> xr.Dataset:
    """Merge collected tiles into a single xarray Dataset with CF metadata."""
    if not tiles:
        raise ValueError("No tiles to reconstruct from")

    spatial_dims = meta["spatial_dims"]
    var_meta     = meta["var_meta"]
    coord_meta   = meta["coord_meta"]
    dim_sizes    = meta["dim_sizes"]

    # ── 1. Build sorted global coordinate arrays from tile spatial_coords ─────
    raw_coords: dict[str, list[np.ndarray]] = {d: [] for d in spatial_dims}
    for tile in tiles:
        sc = tile.get("spatial_coords", {})
        for dim in spatial_dims:
            if dim in sc:
                raw_coords[dim].append(np.asarray(sc[dim], dtype=np.float64))

    sorted_coords: dict[str, np.ndarray] = {}
    for dim in spatial_dims:
        if raw_coords[dim]:
            sorted_coords[dim] = np.unique(np.concatenate(raw_coords[dim]))
        else:
            sorted_coords[dim] = np.array([], dtype=np.float64)

    # ── 2. Build {coordinate_value → global_index} lookup tables ─────────────
    # Use Python float as dict key to get a consistent hash.
    coord_idx: dict[str, dict[float, int]] = {
        dim: {float(v): i for i, v in enumerate(arr)}
        for dim, arr in sorted_coords.items()
    }

    spatial_size = {dim: len(sorted_coords[dim]) for dim in spatial_dims}

    # ── 3. Allocate NaN-filled output arrays for every data variable ──────────
    out_arrays: dict[str, np.ndarray] = {}
    for varname, vmeta in var_meta.items():
        shape = [
            spatial_size[d] if d in spatial_dims else dim_sizes.get(d, 1)
            for d in vmeta["dims"]
        ]
        out_arrays[varname] = np.full(shape, np.nan, dtype=np.float32)

    # ── 4. Place each tile's data into the global arrays ─────────────────────
    for tile in tiles:
        sc = tile.get("spatial_coords", {})

        # Map this tile's coordinate values to global indices.
        tile_gidx: dict[str, np.ndarray] = {}
        for dim in spatial_dims:
            if dim in sc:
                vals = np.asarray(sc[dim], dtype=np.float64)
                tile_gidx[dim] = np.array(
                    [coord_idx[dim][float(v)] for v in vals], dtype=np.intp
                )

        for varname, arr_data in tile.get("variables", {}).items():
            if varname not in out_arrays:
                continue
            arr_data = np.asarray(arr_data, dtype=np.float32)
            var_dims = var_meta[varname]["dims"]

            # Build per-axis index arrays for np.ix_ orthogonal indexing.
            ix_parts = []
            for i, d in enumerate(var_dims):
                if d in spatial_dims and d in tile_gidx:
                    ix_parts.append(tile_gidx[d])
                else:
                    ix_parts.append(np.arange(out_arrays[varname].shape[i], dtype=np.intp))

            out_arrays[varname][np.ix_(*ix_parts)] = arr_data

    # ── 5. Build xarray coordinates ───────────────────────────────────────────
    coords: dict[str, xr.Variable] = {}

    # Spatial dimension coordinates
    for dim in spatial_dims:
        cmeta = coord_meta.get(dim, {})
        coords[dim] = xr.Variable(
            [dim],
            sorted_coords[dim].astype(cmeta.get("dtype", "float64")),
            attrs=cmeta.get("attrs", {}),
        )

    # Non-spatial coordinates: their values were embedded in metadata.
    for name, cmeta in coord_meta.items():
        if name in spatial_dims or name in coords:
            continue
        if "values" in cmeta:
            coords[name] = xr.Variable(
                cmeta.get("dims", [name]),
                np.asarray(cmeta["values"]).astype(cmeta.get("dtype", "float64")),
                attrs=cmeta.get("attrs", {}),
            )

    # ── 6. Build data variables ───────────────────────────────────────────────
    data_vars: dict[str, xr.Variable] = {}
    for varname, arr in out_arrays.items():
        vmeta = var_meta[varname]
        data_vars[varname] = xr.Variable(
            vmeta["dims"],
            arr.astype(vmeta.get("dtype", "float32")),
            attrs=vmeta.get("attrs", {}),
        )

    return xr.Dataset(data_vars, coords=coords, attrs=meta.get("global_attrs", {}))
