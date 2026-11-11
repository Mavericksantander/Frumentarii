"""Carga de ítems STAC como xarray con odc-stac, en UTM y con caché local."""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import xarray as xr

from . import cache as _cache
from .aoi import AOI, cargar_aoi
from .registro import obtener_logger

log = obtener_logger("core.carga")

_rio_configurado = False


def _configurar_rio() -> None:
    global _rio_configurado
    if _rio_configurado:
        return
    import odc.stac

    # Lecturas COG eficientes y acceso anónimo a los buckets públicos de AWS.
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    _rio_configurado = True


def cargar(
    items: Sequence[Any],
    aoi: Union[AOI, str],
    bandas: Sequence[str],
    resolucion: float = 10.0,
    crs: Optional[Any] = None,
    remuestreo: str = "bilinear",
    agrupar: Optional[str] = "solar_day",
    como: Optional[Union[xr.Dataset, xr.DataArray]] = None,
    usar_cache: bool = True,
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
) -> xr.Dataset:
    """Carga ``bandas`` de ``items`` recortadas a la AOI y reproyectadas a UTM.

    - ``crs``: por defecto la zona UTM de la AOI.
    - ``como``: otro raster cuya grilla se quiere copiar exactamente.
    - ``agrupar``: 'solar_day' fusiona las teselas del mismo día.
    Devuelve un Dataset con dimensiones (time, y, x) y CRS escrito con rioxarray.
    """
    import odc.stac
    import rioxarray  # noqa: F401

    aoi = cargar_aoi(aoi)
    items = list(items)
    if not items:
        raise ValueError("No hay escenas para cargar (la búsqueda volvió vacía)")
    crs = crs or aoi.crs_utm
    _configurar_rio()

    grilla = None
    if como is not None:
        grilla = (
            tuple(como.rio.transform())[:6],
            como.sizes["y"],
            como.sizes["x"],
            str(como.rio.crs),
        )
    clave = _cache.clave(
        sorted(it.id for it in items), list(bandas), aoi.bbox, resolucion, crs, remuestreo,
        agrupar, grilla, dtype,
    )
    if usar_cache:
        ds = _cache.leer(clave)
        if ds is not None:
            log.info("  desde caché local (%s): %s", clave[:8], ", ".join(bandas))
            return ds

    log.info(
        "  descargando %d escena(s), bandas [%s] a %g m ...",
        len(items), ", ".join(bandas), resolucion,
    )
    argumentos = dict(bands=list(bandas), resampling=remuestreo, groupby=agrupar)
    if dtype:
        argumentos["dtype"] = dtype
    if nodata is not None:
        argumentos["nodata"] = nodata
    if como is not None:
        from odc.geo.geobox import GeoBox

        argumentos["geobox"] = GeoBox(
            (como.sizes["y"], como.sizes["x"]), como.rio.transform(), str(como.rio.crs)
        )
    else:
        argumentos.update(crs=str(crs), resolution=resolucion, bbox=aoi.bbox)
    ds = odc.stac.load(items, **argumentos).load()
    if "spatial_ref" in ds.coords:
        crs_real = ds.odc.crs
        ds = ds.drop_vars("spatial_ref").rio.write_crs(str(crs_real))
    for v in ds.data_vars:
        ds[v].attrs.pop("grid_mapping", None)
        ds[v].encoding = {}
    log.info(
        "  cargado: %d fecha(s), %d x %d px", ds.sizes.get("time", 1), ds.sizes["y"], ds.sizes["x"]
    )
    if usar_cache:
        try:
            _cache.guardar(ds, clave)
        except Exception as e:
            log.warning("no se pudo escribir el caché (%s)", e)
    return ds


def a_decibeles(lineal: xr.DataArray, piso: float = 1e-5) -> xr.DataArray:
    """Convierte retrodispersión lineal a dB (los ceros y negativos quedan NaN)."""
    valido = lineal > 0
    return (10.0 * np.log10(lineal.where(valido).clip(min=piso))).astype("float32")
