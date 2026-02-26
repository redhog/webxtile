# Gridtiles

Efficient storage and retrieval of 3-D geographical grids as a spatial octree with msgpack containers, designed for progressive web-frontend consumption.

Supports full roundtrip with [xarray](https://xarray.dev) Datasets following [CF Conventions](https://cfconventions.org/).

## Documentation

| Document | Contents |
|----------|----------|
| [User Guide](guide.md) | Installation, writing, reading, level-of-detail, examples |
| [API Reference](api.md) | `write_gridtiles`, `read_gridtiles`, xarray engine & accessor, CF detection |
| [Format Specification](format.md) | Tile file layout, msgpack schema, LOD data model, reconstruction |

## At a glance

```python
import gridtiles, xarray as xr

# Write
gridtiles.write_gridtiles(ds, "tiles/", max_leaf=32, crs="EPSG:3857")

# Read — full resolution
ds = gridtiles.read_gridtiles("tiles/")

# Read — low-fidelity overview (level 2 of the octree)
ds_lo = gridtiles.read_gridtiles("tiles/", level=2)

# Via xarray engine (after pip install)
ds = xr.open_dataset("tiles/", engine="gridtiles")
ds = xr.open_dataset("tiles/", engine="gridtiles", level=2)

# Via xarray accessor
ds.gridtiles.to_gridtiles("tiles/")
```
