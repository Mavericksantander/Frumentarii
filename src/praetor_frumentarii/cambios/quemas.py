"""Áreas quemadas con Sentinel-2: NBR, dNBR y clases de severidad."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import xarray as xr

from ..core import a_2d, como

# Clases de severidad de uso habitual (USGS FIREMON), sobre dNBR sin escalar.
CLASES_SEVERIDAD: Dict[int, Tuple[str, str]] = {
    0: ("#2e7d32", "Sin quemar"),
    1: ("#ffee58", "Baja"),
    2: ("#ffa726", "Moderada baja"),
    3: ("#ef5350", "Moderada alta"),
    4: ("#6a1b9a", "Alta"),
}
LIMITES_SEVERIDAD = (0.10, 0.27, 0.44, 0.66)


def _indice(a: xr.DataArray, b: xr.DataArray) -> np.ndarray:
    a, b = np.asarray(a_2d(a).values, "float64"), np.asarray(a_2d(b).values, "float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(a + b) > 1e-6, (a - b) / (a + b), np.nan)


def nbr(ds: xr.Dataset) -> xr.DataArray:
    """Normalized Burn Ratio = (NIR - SWIR2) / (NIR + SWIR2)."""
    return como(_indice(ds["nir"], ds["swir2"]).astype("float32"), a_2d(ds["nir"]), "nbr")


def dnbr(antes: xr.Dataset, despues: xr.Dataset) -> xr.DataArray:
    """dNBR = NBR antes - NBR después. Positivo: pérdida de vegetación por fuego."""
    return como((nbr(antes).values - nbr(despues).values).astype("float32"),
                a_2d(antes["nir"]), "dnbr")


def severidad(d: xr.DataArray) -> xr.DataArray:
    """Clases 0 a 4 según ``LIMITES_SEVERIDAD``; 255 donde no hay dato."""
    v = np.asarray(d.values, "float64")
    clases = np.digitize(np.nan_to_num(v, nan=-9.0), LIMITES_SEVERIDAD).astype("uint8")
    clases[~np.isfinite(v)] = 255
    return como(clases, d, "severidad")


def area_quemada(antes: xr.Dataset, despues: xr.Dataset, umbral: float = 0.27) -> xr.Dataset:
    """Dataset con ``dnbr``, ``severidad`` y la máscara ``quemado``.

    Se excluye el agua (NDWI > 0 antes o después) porque el mar y los embalses
    producen dNBR espurio. El umbral 0,27 corresponde a severidad moderada baja.
    """
    import rioxarray  # noqa: F401

    d = dnbr(antes, despues)
    agua = (_indice(antes["verde"], antes["nir"]) > 0) | (_indice(despues["verde"], despues["nir"]) > 0)
    d = d.where(~agua)
    quemado = como(np.nan_to_num(d.values, nan=-9.0) >= umbral, d, "quemado")
    salida = xr.Dataset({"dnbr": d, "severidad": severidad(d), "quemado": quemado})
    return salida.rio.write_crs(a_2d(antes["nir"]).rio.crs)
