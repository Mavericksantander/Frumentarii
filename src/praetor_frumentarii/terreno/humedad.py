"""Humedad superficial del suelo como proxy con Sentinel-1.

No es una medición de humedad: es la retrodispersión VV reescalada entre un
valor seco y uno húmedo. La vegetación densa, la rugosidad y la geometría de
observación también cambian la señal, por eso solo se usa como factor suave.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import xarray as xr

from ..core import AOI, a_decibeles, cargar_s1, obtener_logger
from ..core.stac import Fechas

log = obtener_logger("terreno.humedad")

DB_SECO = -17.0
DB_HUMEDO = -7.0


def indice_humedad(
    vv_db: xr.DataArray, db_seco: float = DB_SECO, db_humedo: float = DB_HUMEDO
) -> xr.DataArray:
    """Reescala VV (dB) a un índice 0 (seco) a 1 (muy húmedo)."""
    indice = ((vv_db - db_seco) / (db_humedo - db_seco)).clip(0, 1).astype("float32")
    indice.name = "humedad"
    return indice


def humedad_proxy(
    aoi: Union[AOI, str],
    fechas: Fechas,
    como: Optional[xr.DataArray] = None,
    limite: Optional[int] = 4,
) -> xr.DataArray:
    """Mediana temporal de VV en ``fechas`` convertida a índice 0 a 1."""
    import rioxarray  # noqa: F401

    ds = cargar_s1(aoi, fechas, polarizaciones=("vv",), resolucion=30.0, limite=limite)
    vv_db = a_decibeles(ds["vv"]).median("time", skipna=True)
    vv_db = vv_db.rio.write_crs(ds.rio.crs)
    if como is not None:
        vv_db = vv_db.rio.reproject_match(como, nodata=np.nan)
        vv_db = vv_db.assign_coords(x=como["x"], y=como["y"])
    indice = indice_humedad(vv_db)
    return indice.rio.write_crs(vv_db.rio.crs)
