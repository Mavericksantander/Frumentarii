"""Red vial desde OpenStreetMap (osmnx) y su rasterización por clase."""
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import xarray as xr

from ..core import AOI, cargar_aoi, como, obtener_logger

log = obtener_logger("terreno.vias")

SIN_VIA, SENDERO, HUELLA, CAMINO, PRINCIPAL = 0, 1, 2, 3, 4
CLASES_VIA: Dict[int, Tuple[str, str]] = {
    SENDERO: ("#8d6e63", "Sendero"),
    HUELLA: ("#ffb300", "Huella"),
    CAMINO: ("#fafafa", "Camino"),
    PRINCIPAL: ("#e53935", "Ruta principal"),
}
_SENDEROS = {"path", "footway", "bridleway", "steps", "cycleway", "pedestrian"}
_HUELLAS = {"track"}
_PRINCIPALES = {
    "motorway", "trunk", "primary", "secondary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
}
_IGNORAR = {"proposed", "construction", "abandoned", "razed", "platform", "corridor", "elevator"}


def clasificar_via(highway) -> int:
    """Clase de vía (1 a 4) desde la etiqueta ``highway`` de OSM; 0 si no aplica."""
    if isinstance(highway, (list, tuple)):
        highway = highway[0] if highway else None
    if not isinstance(highway, str) or highway in _IGNORAR:
        return SIN_VIA
    if highway in _SENDEROS:
        return SENDERO
    if highway in _HUELLAS:
        return HUELLA
    if highway in _PRINCIPALES:
        return PRINCIPAL
    return CAMINO


def red_vial(aoi: Union[AOI, str]):
    """Descarga las vías de OSM dentro de la AOI. GeoDataFrame en EPSG:4326."""
    import geopandas as gpd
    import osmnx as ox

    from ..core.cache import carpeta_cache

    ox.settings.cache_folder = str(carpeta_cache() / "osm")
    ox.settings.log_console = False
    aoi = cargar_aoi(aoi)
    log.info("descargando red vial de OpenStreetMap para %s ...", aoi.nombre)
    try:
        crudo = ox.features_from_polygon(aoi.geometria, {"highway": True})
    except Exception as e:  # osmnx lanza InsufficientResponseError si no hay nada
        if "No matching features" in str(e) or "InsufficientResponse" in type(e).__name__:
            log.warning("OSM no tiene vías en esta AOI")
            return gpd.GeoDataFrame({"highway": [], "clase": []}, geometry=[], crs=4326)
        raise
    lineas = crudo[crudo.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    columnas = [c for c in ("highway", "name", "surface", "ref") if c in lineas.columns]
    lineas = lineas[columnas + ["geometry"]].reset_index(drop=True)
    lineas["clase"] = lineas["highway"].map(clasificar_via).astype("int16")
    lineas = lineas[lineas["clase"] > 0].reset_index(drop=True)
    log.info("  %d tramos (%.0f km)", len(lineas),
             lineas.to_crs(aoi.crs_utm).length.sum() / 1000.0 if len(lineas) else 0.0)
    return lineas


def rasterizar_vias(vias, referencia: xr.DataArray) -> xr.DataArray:
    """Raster uint8 con la clase de vía más alta que toca cada píxel."""
    import rioxarray  # noqa: F401
    from rasterio import features

    forma = (referencia.sizes["y"], referencia.sizes["x"])
    salida = np.zeros(forma, dtype="uint8")
    if vias is not None and len(vias):
        en_crs = vias.to_crs(referencia.rio.crs)
        for clase in sorted(set(int(c) for c in en_crs["clase"])):
            geoms = [g for g in en_crs.geometry[en_crs["clase"] == clase] if g and not g.is_empty]
            if not geoms:
                continue
            quemado = features.rasterize(
                [(g, 1) for g in geoms], out_shape=forma, transform=referencia.rio.transform(),
                all_touched=True, dtype="uint8",
            )
            salida[quemado == 1] = clase
    return como(salida, referencia, "vias")
