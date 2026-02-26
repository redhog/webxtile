# WebXTile

Efficient storage and retrieval of 3-D geographical grids as a spatial octree with msgpack containers, designed for progressive web-frontend consumption.

Supports full roundtrip with [xarray](https://xarray.dev) Datasets following [CF Conventions](https://cfconventions.org/).

## Documentation

| Document | Contents |
|----------|----------|
| [User Guide](guide.md) | Installation, writing, reading, level-of-detail, examples, testing |
| [API Reference](api.md) | `write_webxtile`, `read_webxtile`, xarray engine & accessor, CF detection |
| [Format Specification](format.md) | Tile file layout, msgpack schema, LOD data model, reconstruction |

## At a glance

```python
import webxtile, xarray as xr

# Write
webxtile.write_webxtile(ds, "tiles/", max_leaf=32, crs="EPSG:3857")

# Read — full resolution
ds = webxtile.read_webxtile("tiles/")

# Read — low-fidelity overview (level 2 of the octree)
ds_lo = webxtile.read_webxtile("tiles/", level=2)

# Via xarray engine (after pip install)
ds = xr.open_dataset("tiles/", engine="webxtile")
ds = xr.open_dataset("tiles/", engine="webxtile", level=2)

# Via xarray accessor
ds.webxtile.to_webxtile("tiles/")
```
