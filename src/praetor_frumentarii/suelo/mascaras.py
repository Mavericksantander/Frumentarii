"""Máscaras de vegetación y agua para dejar solo suelo y roca expuestos."""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..core import a_2d, como


def _normalizado(a: xr.DataArray, b: xr.DataArray) -> np.ndarray:
    x = np.asarray(a_2d(a).values, "float64")
    y = np.asarray(a_2d(b).values, "float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(x + y) > 1e-6, (x - y) / (x + y), np.nan)


def mascara_vegetacion_agua(
    ds: xr.Dataset, ndvi_max: float = 0.25, ndwi_max: float = 0.0, brillo_min: float = 0.04
) -> xr.DataArray:
    """True donde el píxel es suelo o roca expuestos.

    Descarta vegetación (NDVI > ``ndvi_max``), agua (NDWI > ``ndwi_max``),
    sombras profundas (reflectancia roja < ``brillo_min``) y píxeles sin dato.
    Para ASTER usa b3 (NIR) y b2 (rojo) para el NDVI, y detecta el agua por su
    oscuridad en el SWIR (b4 < 0,05). Ajuste tras la validación con datos reales
    (Atacama): ASTER L1T no tiene corrección atmosférica, el verde queda inflado
    por la dispersión y el NDWI marcaba todo el desierto como agua.
    """
    if "b6" in ds:
        rojo, nir = ds["b2"], ds["b3"]
        ndvi = _normalizado(nir, rojo)
        swir = np.asarray(a_2d(ds["b4"]).values, "float64")
        ndwi = np.where(np.isfinite(swir), np.where(swir < 0.05, 1.0, -1.0), np.nan)
    else:
        verde, rojo, nir = ds["verde"], ds["rojo"], ds["nir"]
        ndvi = _normalizado(nir, rojo)
        ndwi = _normalizado(verde, nir)
    r = np.asarray(a_2d(rojo).values, "float64")
    valido = (
        np.isfinite(ndvi) & np.isfinite(ndwi) & (ndvi <= ndvi_max) & (ndwi <= ndwi_max)
        & (r >= brillo_min)
    )
    m = como(valido, a_2d(rojo), "suelo_expuesto")
    m.attrs["fraccion"] = float(valido.mean())
    return m
