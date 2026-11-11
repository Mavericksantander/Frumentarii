"""Manchas oscuras en el mar con Sentinel-1.

Una película de hidrocarburo (derrame o filtración natural) amortigua las
ondas capilares y se ve más oscura que el mar vecino en banda C. Lo mismo
ocurre con zonas sin viento, sombras de viento tras la costa, floraciones de
algas, hielo graso, lluvia y surgencias: una mancha oscura es un indicio, no
una confirmación.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import ndimage

from ..core import como, resolucion


def fondo_local(vv_db: xr.DataArray, mar: np.ndarray, ventana_m: float = 4000.0) -> np.ndarray:
    """Media local del mar (dB) en una ventana grande, ignorando tierra y NaN."""
    v = np.asarray(vv_db.values, dtype="float64")
    valido = mar & np.isfinite(v)
    n = max(3, int(round(ventana_m / resolucion(vv_db))) | 1)
    suma = ndimage.uniform_filter(np.where(valido, v, 0.0), n, mode="nearest")
    peso = ndimage.uniform_filter(valido.astype("float64"), n, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(peso > 0.1, suma / peso, np.nan)


def manchas_oscuras(
    vv_db: xr.DataArray,
    tierra: Optional[xr.DataArray] = None,
    contraste_db: float = 4.0,
    ventana_m: float = 4000.0,
    area_min_ha: float = 10.0,
    fraccion_max: float = 0.3,
    borde_m: float = 800.0,
) -> xr.Dataset:
    """Detecta zonas del mar al menos ``contraste_db`` más oscuras que su entorno.

    Devuelve un Dataset con ``contraste`` (dB bajo el fondo local) y ``mancha``
    (máscara booleana). Se descartan manchas menores que ``area_min_ha`` y, si
    lo oscuro supera ``fraccion_max`` del mar, se asume calma generalizada y no
    se reporta nada (atributo ``descartada_por_calma``).

    ``borde_m`` excluye una franja junto a los píxeles sin dato. Ajuste tras la
    validación con datos reales (Magallanes, 2024): el borde de la franja de
    Sentinel-1 es más oscuro por ruido de borde y aparecía como una "mancha"
    recta y recurrente en todas las fechas.
    """
    import rioxarray  # noqa: F401

    v = np.asarray(vv_db.values, dtype="float64")
    mar = np.isfinite(v)
    if borde_m > 0 and not mar.all():
        pasos = int(np.ceil(borde_m / resolucion(vv_db)))
        mar &= ~ndimage.binary_dilation(~mar, iterations=pasos)
    if tierra is not None:
        mar &= ~np.asarray(tierra.values).astype(bool)
    suave = ndimage.uniform_filter(np.where(mar, v, 0.0), 3, mode="nearest")
    peso = ndimage.uniform_filter(mar.astype("float64"), 3, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        suave = np.where(peso > 0.3, suave / peso, np.nan)
    fondo = fondo_local(como(suave, vv_db), mar, ventana_m)
    contraste = np.where(mar, fondo - suave, np.nan)
    oscuro = mar & (np.nan_to_num(contraste, nan=-99.0) >= contraste_db)
    oscuro = ndimage.binary_opening(oscuro, structure=np.ones((2, 2), bool))
    px_min = max(1, int(np.ceil(area_min_ha * 1e4 / resolucion(vv_db) ** 2)))
    etiquetas, n = ndimage.label(oscuro)
    if n:
        tamanos = ndimage.sum(oscuro, etiquetas, index=np.arange(1, n + 1))
        oscuro = np.isin(etiquetas, np.flatnonzero(tamanos >= px_min) + 1)
    calma = bool(mar.sum() and oscuro.sum() / float(mar.sum()) > fraccion_max)
    if calma:
        oscuro = np.zeros_like(oscuro)
    salida = xr.Dataset({
        "contraste": como(contraste.astype("float32"), vv_db),
        "mancha": como(oscuro, vv_db),
    }).rio.write_crs(vv_db.rio.crs)
    salida.attrs.update(
        descartada_por_calma=calma,
        mar_px=int(mar.sum()),
        mar_mediana_db=float(np.nanmedian(v[mar])) if mar.any() else float("nan"),
    )
    return salida


def describir_manchas(resultado: xr.Dataset, fecha: str = ""):
    """Polígonos de cada mancha con área, contraste medio y elongación."""
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape
    from skimage.measure import regionprops

    mascara = np.asarray(resultado["mancha"].values).astype(bool)
    etiquetas, n = ndimage.label(mascara)
    res = resolucion(resultado["mancha"])
    filas = []
    transformada = resultado["mancha"].rio.transform()
    contraste = np.asarray(resultado["contraste"].values)
    geoms: Dict[int, Any] = {}
    for g, valor in features.shapes(etiquetas.astype("int32"), mask=mascara, transform=transformada):
        v = int(valor)
        geoms[v] = shape(g) if v not in geoms else geoms[v].union(shape(g))
    for prop in regionprops(etiquetas):
        menor = max(prop.minor_axis_length, 1.0)
        filas.append({
            "fecha": fecha,
            "area_ha": round(prop.area * res * res / 1e4, 2),
            "contraste_db": round(float(np.nanmean(contraste[etiquetas == prop.label])), 2),
            "elongacion": round(float(prop.major_axis_length / menor), 2),
            "geometry": geoms[prop.label],
        })
    gdf = gpd.GeoDataFrame(filas, geometry="geometry", crs=resultado.rio.crs) if filas else \
        gpd.GeoDataFrame({"fecha": [], "area_ha": [], "contraste_db": [], "elongacion": []},
                         geometry=[], crs=resultado.rio.crs)
    return gdf.sort_values("area_ha", ascending=False).reset_index(drop=True)
