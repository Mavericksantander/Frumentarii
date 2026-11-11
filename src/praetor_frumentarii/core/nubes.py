"""Enmascarado de nubes para Sentinel-2 (SCL) y Landsat (QA_PIXEL)."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import xarray as xr

# Clases SCL de Sentinel-2 L2A que se descartan:
# 0 sin dato, 1 saturado/defectuoso, 3 sombra de nube, 8 y 9 nubes, 10 cirros, 11 nieve.
SCL_MALAS = (0, 1, 3, 8, 9, 10, 11)
SCL_AGUA = 6

# Bits de QA_PIXEL (Landsat Colección 2): 0 relleno, 1 nube dilatada, 2 cirros,
# 3 nube, 4 sombra de nube, 5 nieve.
QA_BITS_MALOS = (0, 1, 2, 3, 4, 5)


def mascara_scl(scl: xr.DataArray, clases_malas: Iterable[int] = SCL_MALAS) -> xr.DataArray:
    """True donde el píxel es utilizable según la capa SCL."""
    return ~scl.isin(list(clases_malas))


def mascara_qa_pixel(qa: xr.DataArray, bits_malos: Iterable[int] = QA_BITS_MALOS) -> xr.DataArray:
    """True donde el píxel es utilizable según QA_PIXEL de Landsat."""
    patron = 0
    for b in bits_malos:
        patron |= 1 << b
    return (qa.astype("uint16") & patron) == 0


def enmascarar_nubes(
    ds: xr.Dataset, metodo: str = "scl", banda: Optional[str] = None, quitar_banda: bool = True
) -> xr.Dataset:
    """Pone NaN en los píxeles nubosos de todas las variables del Dataset."""
    if metodo == "scl":
        banda = banda or "scl"
        buena = mascara_scl(ds[banda])
    elif metodo == "qa_pixel":
        banda = banda or "qa_pixel"
        buena = mascara_qa_pixel(ds[banda])
    else:
        raise ValueError("metodo debe ser 'scl' o 'qa_pixel'")
    resto = ds.drop_vars(banda) if quitar_banda else ds
    salida = resto.astype("float32").where(buena)
    salida.attrs = dict(ds.attrs)
    salida.attrs["fraccion_valida"] = float(buena.mean())
    return salida


def compuesto_mediana(ds: xr.Dataset) -> xr.Dataset:
    """Mediana temporal ignorando NaN: compuesto sin nubes si hay varias fechas."""
    if "time" not in ds.dims:
        return ds
    if ds.sizes["time"] == 1:
        return ds.isel(time=0, drop=True)
    return ds.median("time", skipna=True, keep_attrs=True)


def fraccion_valida(da: xr.DataArray) -> float:
    return float(np.isfinite(da.values).mean())
