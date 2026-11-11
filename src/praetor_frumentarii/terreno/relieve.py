"""Pendiente y rugosidad desde un modelo de elevación (Copernicus DEM GLO-30)."""
from __future__ import annotations

from typing import Union

import numpy as np
import xarray as xr
from scipy import ndimage

from ..core import AOI, cargar_dem, como, resolucion


def pendiente(dem: xr.DataArray) -> xr.DataArray:
    """Pendiente en grados por diferencias centrales."""
    res = resolucion(dem)
    z = np.asarray(dem.values, dtype="float64")
    dz_dy, dz_dx = np.gradient(z, res, res)
    grados = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    return como(grados.astype("float32"), dem, "pendiente")


def rugosidad(dem: xr.DataArray, ventana: int = 3) -> xr.DataArray:
    """Desviación estándar local de la elevación (m) en una ventana cuadrada."""
    z = np.asarray(dem.values, dtype="float64")
    relleno = np.where(np.isfinite(z), z, np.nanmean(z) if np.isfinite(z).any() else 0.0)
    media = ndimage.uniform_filter(relleno, ventana, mode="nearest")
    media2 = ndimage.uniform_filter(relleno ** 2, ventana, mode="nearest")
    desv = np.sqrt(np.clip(media2 - media ** 2, 0, None))
    desv[~np.isfinite(z)] = np.nan
    return como(desv.astype("float32"), dem, "rugosidad")


def pendiente_rugosidad(
    aoi_o_dem: Union[AOI, str, xr.DataArray], resolucion_m: float = 30.0
) -> xr.Dataset:
    """Dataset con ``elevacion`` (m), ``pendiente`` (grados) y ``rugosidad`` (m).

    Acepta una AOI (descarga el DEM) o un DEM ya cargado.
    """
    import rioxarray  # noqa: F401

    dem = aoi_o_dem if isinstance(aoi_o_dem, xr.DataArray) else cargar_dem(aoi_o_dem, resolucion_m)
    ds = xr.Dataset(
        {"elevacion": dem.astype("float32"), "pendiente": pendiente(dem), "rugosidad": rugosidad(dem)}
    )
    return ds.rio.write_crs(dem.rio.crs)
