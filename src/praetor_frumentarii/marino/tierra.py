"""Máscara de tierra para análisis marino."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy import ndimage

from ..core import como, resolucion


def mascara_tierra(
    referencia: xr.DataArray,
    fuente: Union[str, Path] = "global-land-mask",
    margen_m: float = 500.0,
) -> xr.DataArray:
    """True sobre tierra (más un margen costero) en la grilla de ``referencia``.

    ``fuente`` es 'global-land-mask' (GLOBE, cerca de 1 km: grueso en fiordos y
    canales) o la ruta a un archivo vectorial de polígonos de tierra (shapefile,
    GeoPackage, GeoJSON), que es lo recomendable en costas complejas.
    El margen evita la franja costera, donde abundan los falsos positivos.
    """
    import rioxarray  # noqa: F401

    forma = (referencia.sizes["y"], referencia.sizes["x"])
    if str(fuente) == "global-land-mask":
        from global_land_mask import globe

        xx, yy = np.meshgrid(referencia["x"].values, referencia["y"].values)
        t = Transformer.from_crs(referencia.rio.crs, 4326, always_xy=True)
        lon, lat = t.transform(xx, yy)
        tierra = np.asarray(globe.is_land(lat, lon), dtype=bool)
    else:
        import geopandas as gpd
        from rasterio import features

        ruta = Path(fuente)
        if not ruta.exists():
            raise FileNotFoundError("No existe el archivo de costa: %s" % ruta)
        oeste, sur, este, norte = referencia.rio.transform_bounds(4326)
        poligonos = gpd.read_file(ruta, bbox=(oeste, sur, este, norte)).to_crs(referencia.rio.crs)
        geoms = [g for g in poligonos.geometry if g is not None and not g.is_empty]
        tierra = features.rasterize(
            [(g, 1) for g in geoms], out_shape=forma, transform=referencia.rio.transform(),
            all_touched=True, dtype="uint8",
        ).astype(bool) if geoms else np.zeros(forma, dtype=bool)
    if margen_m > 0:
        pasos = int(np.ceil(margen_m / resolucion(referencia)))
        tierra = ndimage.binary_dilation(tierra, iterations=pasos)
    return como(tierra, referencia, "tierra")
