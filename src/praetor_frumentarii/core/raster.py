"""Utilidades raster comunes: grillas, vectorización y realce para figuras."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr


def crear_raster(
    datos: np.ndarray,
    crs: Any = "EPSG:32718",
    x0: float = 600000.0,
    y0: float = 5300000.0,
    resolucion: float = 30.0,
    nombre: Optional[str] = None,
) -> xr.DataArray:
    """Envuelve un arreglo (y, x) como DataArray georreferenciado (norte arriba)."""
    import rioxarray  # noqa: F401

    alto, ancho = datos.shape[-2:]
    xs = x0 + (np.arange(ancho) + 0.5) * resolucion
    ys = y0 - (np.arange(alto) + 0.5) * resolucion
    da = xr.DataArray(datos, dims=("y", "x"), coords={"y": ys, "x": xs}, name=nombre)
    return da.rio.write_crs(crs)


def como(datos: np.ndarray, referencia: xr.DataArray, nombre: Optional[str] = None) -> xr.DataArray:
    """Nuevo DataArray con la misma grilla y CRS que ``referencia``."""
    import rioxarray  # noqa: F401

    da = xr.DataArray(
        datos, dims=("y", "x"), coords={"y": referencia["y"], "x": referencia["x"]}, name=nombre
    )
    return da.rio.write_crs(referencia.rio.crs)


def resolucion(da: xr.DataArray) -> float:
    """Tamaño de píxel en unidades del CRS (metros en UTM)."""
    return float(abs(da["x"].values[1] - da["x"].values[0]))


def a_2d(da: xr.DataArray) -> xr.DataArray:
    """Quita la dimensión de tiempo si tiene un solo elemento."""
    if "time" in da.dims:
        if da.sizes["time"] != 1:
            raise ValueError("Se esperaba una sola fecha; hay %d" % da.sizes["time"])
        da = da.isel(time=0, drop=True)
    return da


def vectorizar(
    mascara: xr.DataArray,
    area_min_ha: float = 0.5,
    atributos: Optional[Dict[str, Any]] = None,
):
    """Convierte una máscara booleana en polígonos (GeoDataFrame en el CRS del raster)."""
    import geopandas as gpd
    import rioxarray  # noqa: F401
    from rasterio import features
    from shapely.geometry import shape

    valores = np.asarray(mascara.values).astype("uint8")
    geoms = [
        shape(g)
        for g, v in features.shapes(valores, mask=valores == 1, transform=mascara.rio.transform())
        if v == 1
    ]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=mascara.rio.crs)
    gdf["area_ha"] = gdf.geometry.area / 1e4 if len(gdf) else []
    gdf = gdf[gdf["area_ha"] >= area_min_ha].sort_values("area_ha", ascending=False)
    gdf = gdf.reset_index(drop=True)
    for k, v in (atributos or {}).items():
        gdf[k] = v
    return gdf


def estirar(arreglo: np.ndarray, percentiles: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """Realce lineal a [0, 1] entre percentiles, ignorando NaN."""
    a = np.asarray(arreglo, dtype="float32")
    finitos = a[np.isfinite(a)]
    if finitos.size == 0:
        return np.zeros_like(a)
    bajo, alto = np.percentile(finitos, percentiles)
    if alto <= bajo:
        alto = bajo + 1e-6
    return np.clip((a - bajo) / (alto - bajo), 0, 1)


def componer_rgb(ds: xr.Dataset, bandas: Sequence[str] = ("rojo", "verde", "azul")) -> np.ndarray:
    """Arreglo (y, x, 3) realzado para mostrar; los NaN quedan en negro."""
    capas = [estirar(a_2d(ds[b]).values) for b in bandas]
    return np.nan_to_num(np.dstack(capas), nan=0.0)


def reproyectar(da: xr.DataArray, crs_destino: Any, remuestreo: str = "bilinear") -> xr.DataArray:
    """Reproyecta un DataArray 2D a otro CRS con rasterio.warp (grilla automática).

    Es independiente de rioxarray para controlar el relleno de los bordes.
    """
    import rioxarray  # noqa: F401
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    alto, ancho = da.shape
    oeste, sur, este, norte = da.rio.bounds()
    transformada, ancho2, alto2 = calculate_default_transform(
        da.rio.crs, crs_destino, ancho, alto, oeste, sur, este, norte
    )
    flotante = np.issubdtype(da.dtype, np.floating)
    origen = np.asarray(da.values)
    destino = np.full((alto2, ancho2), np.nan if flotante else 0, dtype=origen.dtype)
    reproject(
        origen, destino, src_transform=da.rio.transform(), src_crs=da.rio.crs,
        dst_transform=transformada, dst_crs=crs_destino,
        src_nodata=np.nan if flotante else None, dst_nodata=np.nan if flotante else 0,
        resampling=getattr(Resampling, remuestreo),
    )
    xs = transformada.c + (np.arange(ancho2) + 0.5) * transformada.a
    ys = transformada.f + (np.arange(alto2) + 0.5) * transformada.e
    salida = xr.DataArray(destino, dims=("y", "x"), coords={"y": ys, "x": xs}, name=da.name)
    return salida.rio.write_crs(crs_destino)


def leer_geotiff(ruta: Any, enmascarar: bool = False) -> xr.DataArray:
    """Lee un GeoTIFF completo a memoria y cierra el archivo.

    Siempre se cierra el descriptor: un ``open_rasterio`` que queda abierto
    hace que Python imprima "Error in sys.excepthook" al terminar (rasterio 1.5).
    Devuelve (band, y, x); usa ``.isel(band=0, drop=True)`` para una sola banda.
    """
    import rioxarray

    with rioxarray.open_rasterio(ruta, masked=enmascarar) as archivo:
        da = archivo.load()
    return da
