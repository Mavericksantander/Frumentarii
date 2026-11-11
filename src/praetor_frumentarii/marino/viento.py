"""Filtro de viento para la detección de manchas (ERA5 opcional).

La detección de películas en SAR funciona con viento moderado: con menos de
unos 2 m/s el mar entero se ve oscuro, y con más de unos 10 m/s el oleaje
rompe la película y borra el contraste.

Fuentes de viento, en orden de preferencia:
1. Un valor dado por la persona usuaria (``--viento 6.5``).
2. Un archivo ERA5 propio (NetCDF del Copernicus Climate Data Store con
   ``u10`` y ``v10``); requiere un motor NetCDF para xarray (``pip install h5netcdf``).
3. ERA5 en Planetary Computer (colección ``era5-pds``): solo cubre 1979 a 2020
   y requiere ``pip install zarr adlfs``.
4. Si no hay nada de lo anterior, un estado del mar estimado desde el propio
   SAR (``estado_mar_por_sar``), que es solo orientativo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

from ..core import AOI, cargar_aoi, obtener_logger

log = obtener_logger("marino.viento")

VIENTO_MIN_MS = 2.0
VIENTO_MAX_MS = 10.0


def apto_por_viento(viento_ms: Optional[float], minimo: float = VIENTO_MIN_MS,
                    maximo: float = VIENTO_MAX_MS) -> Optional[bool]:
    """True/False si el viento permite detectar películas; None si no se conoce."""
    if viento_ms is None or not np.isfinite(viento_ms):
        return None
    return bool(minimo <= viento_ms <= maximo)


def estado_mar_por_sar(mar_mediana_db: float) -> Tuple[str, Optional[bool]]:
    """Estado del mar estimado desde la mediana de VV sobre el mar.

    Muy aproximado: VV también depende del ángulo de incidencia y de la
    dirección del viento. Devuelve (descripción, apto o None si es dudoso).
    """
    if not np.isfinite(mar_mediana_db):
        return ("sin datos de mar", None)
    if mar_mediana_db < -23.0:
        return ("mar en calma (VV %.1f dB): no apto, todo se ve oscuro" % mar_mediana_db, False)
    if mar_mediana_db > -7.0:
        return ("mar muy agitado (VV %.1f dB): el oleaje borra las películas" % mar_mediana_db, False)
    return ("viento probablemente moderado (VV %.1f dB)" % mar_mediana_db, True)


def _velocidad(u: xr.DataArray, v: xr.DataArray) -> float:
    return float(np.sqrt(u ** 2 + v ** 2).mean())


def viento_desde_archivo(ruta: Union[str, Path], aoi: Union[AOI, str], fecha: str) -> float:
    """Velocidad media del viento a 10 m (m/s) en la AOI desde un NetCDF ERA5."""
    aoi = cargar_aoi(aoi)
    oeste, sur, este, norte = aoi.bbox
    ds = xr.open_dataset(ruta)
    try:
        nombres = {"u": None, "v": None}
        for cand in ("u10", "10m_u_component_of_wind", "eastward_wind_at_10_metres"):
            if cand in ds:
                nombres["u"] = cand
        for cand in ("v10", "10m_v_component_of_wind", "northward_wind_at_10_metres"):
            if cand in ds:
                nombres["v"] = cand
        if None in nombres.values():
            raise ValueError("El archivo no tiene componentes u10 y v10 del viento")
        lat = "latitude" if "latitude" in ds.coords else "lat"
        lon = "longitude" if "longitude" in ds.coords else "lon"
        tiempo = "valid_time" if "valid_time" in ds.coords else "time"
        sub = ds.sel({tiempo: np.datetime64(fecha)}, method="nearest")
        lons = sub[lon].values
        oeste_, este_ = (oeste % 360, este % 360) if lons.max() > 180 else (oeste, este)
        # Se amplía medio grado: la grilla ERA5 es de 0,25 grados.
        sub = sub.where((sub[lat] >= sur - 0.5) & (sub[lat] <= norte + 0.5)
                        & (sub[lon] >= oeste_ - 0.5) & (sub[lon] <= este_ + 0.5), drop=True)
        return _velocidad(sub[nombres["u"]], sub[nombres["v"]])
    finally:
        ds.close()


def viento_era5(aoi: Union[AOI, str], fecha: str,
                archivo: Optional[Union[str, Path]] = None) -> Optional[float]:
    """Viento a 10 m (m/s) para la AOI y fecha, o None si no hay fuente disponible."""
    aoi = cargar_aoi(aoi)
    if archivo:
        return viento_desde_archivo(archivo, aoi, fecha)
    anio = int(str(fecha)[:4])
    if anio > 2020:
        log.info("  ERA5 en Planetary Computer llega solo hasta 2020; sin viento para %s", fecha)
        return None
    try:
        import adlfs  # noqa: F401
        import zarr  # noqa: F401
    except ImportError:
        log.info("  ERA5 opcional no instalado (pip install zarr adlfs); sin filtro de viento")
        return None
    try:
        from ..core.stac import abrir_catalogo

        catalogo = abrir_catalogo("planetary")
        item = catalogo.get_collection("era5-pds").get_item("era5-pds-%s-an" % str(fecha)[:7])
        componentes = []
        oeste, sur, este, norte = aoi.bbox
        for nombre in ("eastward_wind_at_10_metres", "northward_wind_at_10_metres"):
            activo = item.assets[nombre]
            ds = xr.open_dataset(activo.href, **activo.extra_fields["xarray:open_kwargs"])
            da = ds[nombre].sel(time=np.datetime64(fecha), method="nearest")
            da = da.sel(lat=slice(norte + 0.5, sur - 0.5),
                        lon=slice(oeste % 360 - 0.5, este % 360 + 0.5)).load()
            componentes.append(da)
        return _velocidad(componentes[0], componentes[1])
    except Exception as e:
        log.warning("  no se pudo leer ERA5 (%s: %s); sin filtro de viento", type(e).__name__, e)
        return None
