"""Flujo completo de terreno: capas, costo, ruta y archivos de salida."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import xarray as xr

from ..core import (
    AOI, Panel, Progreso, cargar_aoi, como, guardar_geojson, guardar_geotiff,
    guardar_resumen_png, obtener_logger, registrar_corrida,
)
from .cobertura import CLASES_WORLDCOVER, cobertura as _cobertura
from .costo import fraccion_transitable, mapa_costo
from .humedad import humedad_proxy
from .relieve import pendiente_rugosidad
from .rutas import RutaImposible, ruta_menor_costo
from .vias import CLASES_VIA, rasterizar_vias, red_vial

log = obtener_logger("terreno")
MODULO = "terreno"

FUENTES = [
    "Copernicus DEM GLO-30 (© DLR e.V. 2010-2014 y © Airbus Defence and Space GmbH 2014-2018, "
    "provisto bajo COPERNICUS por la Unión Europea y la ESA)",
    "ESA WorldCover 10 m 2021 v200 (CC BY 4.0)",
    "© Colaboradores de OpenStreetMap (ODbL)",
]


def _sombreado(elevacion: xr.DataArray) -> np.ndarray:
    """Relieve sombreado simple (0 a 1) para usar de fondo en las figuras."""
    z = np.nan_to_num(np.asarray(elevacion.values, dtype="float64"))
    dy, dx = np.gradient(z)
    pend = np.pi / 2 - np.arctan(np.hypot(dx, dy) / 30.0 * 2)
    aspecto = np.arctan2(-dx, dy)
    az, alt = np.radians(315), np.radians(45)
    s = np.sin(alt) * np.sin(pend) + np.cos(alt) * np.cos(pend) * np.cos(az - aspecto)
    return np.clip((s - s.min()) / (np.ptp(s) + 1e-9), 0, 1)


def ejecutar(
    aoi: Union[AOI, str],
    modo: str = "pie",
    origen: Optional[Sequence[float]] = None,
    destino: Optional[Sequence[float]] = None,
    fechas_humedad: Optional[str] = None,
    usar_vias: bool = True,
    resolucion_m: float = 30.0,
    visor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Calcula el mapa de costo (y la ruta si hay origen y destino) y lo guarda.

    Devuelve un diccionario con las capas, la ruta y las rutas de archivo.
    """
    aoi = cargar_aoi(aoi)
    avisos: List[str] = []
    p = Progreso(6, log)

    def mostrar(panel: Panel) -> None:
        if visor is not None:
            visor.mostrar(panel)

    p.paso("relieve: Copernicus DEM GLO-30 para %s" % aoi.nombre)
    relieve = pendiente_rugosidad(aoi, resolucion_m)
    sombra = _sombreado(relieve["elevacion"])
    gris = np.dstack([sombra] * 3)
    mostrar(Panel(relieve["elevacion"], "Elevación (Copernicus DEM)", "terrain", etiqueta="m"))
    mostrar(Panel(relieve["pendiente"], "Pendiente", "magma", 0, 40, "grados"))

    p.paso("cobertura de suelo: ESA WorldCover")
    cob = _cobertura(aoi, relieve["elevacion"])
    mostrar(Panel(cob, "Cobertura (ESA WorldCover)", clases=CLASES_WORLDCOVER))

    p.paso("humedad del suelo (proxy Sentinel-1)" if fechas_humedad else
           "humedad del suelo: omitida (sin --humedad)")
    humedad = None
    if fechas_humedad:
        try:
            humedad = humedad_proxy(aoi, fechas_humedad, como=relieve["elevacion"])
            mostrar(Panel(humedad, "Humedad (proxy Sentinel-1 VV)", "YlGnBu", 0, 1, "índice"))
        except Exception as e:
            avisos.append("Humedad no disponible: %s" % e)
            log.warning(avisos[-1])

    p.paso("red vial de OpenStreetMap" if usar_vias else "red vial: omitida")
    vias_gdf, vias_raster = None, None
    if usar_vias:
        try:
            vias_gdf = red_vial(aoi)
            vias_raster = rasterizar_vias(vias_gdf, relieve["elevacion"])
        except Exception as e:
            avisos.append("Red vial no disponible (%s); el costo ignora caminos" % type(e).__name__)
            log.warning("%s: %s", avisos[-1], e)

    p.paso("mapa de costo para modo '%s'" % modo)
    costo = mapa_costo(relieve["pendiente"], cob, modo, relieve["rugosidad"], humedad, vias_raster)
    min_km = como(np.where(np.isfinite(costo.values), costo.values * 1000.0 / 60.0, np.nan),
                  costo, "min_por_km")
    panel_costo = Panel(min_km, "Costo de desplazamiento (%s)" % modo, "viridis_r",
                        vmax=float(np.nanpercentile(min_km.values, 97)) if np.isfinite(
                            min_km.values).any() else None,
                        etiqueta="minutos por km (sin color: intransitable)", fondo=gris,
                        referencia=costo)
    mostrar(panel_costo)

    p.paso("ruta de menor costo" if origen and destino else "ruta: omitida (sin origen y destino)")
    ruta = None
    if origen and destino:
        try:
            ruta = ruta_menor_costo(costo, origen, destino)
        except RutaImposible as e:
            avisos.append("Sin ruta: %s" % e)
            log.warning(avisos[-1])

    archivos = [
        guardar_geotiff(costo, MODULO, aoi, "costo_%s" % modo),
        guardar_geotiff(relieve["pendiente"], MODULO, aoi, "pendiente"),
        guardar_geotiff(relieve["rugosidad"], MODULO, aoi, "rugosidad"),
        guardar_geotiff(cob, MODULO, aoi, "cobertura"),
    ]
    if humedad is not None:
        archivos.append(guardar_geotiff(humedad, MODULO, aoi, "humedad_proxy"))
    if vias_gdf is not None:
        archivos.append(guardar_geojson(vias_gdf, MODULO, aoi, "vias"))
    if ruta is not None:
        archivos.append(guardar_geojson(ruta, MODULO, aoi, "ruta_%s" % modo))

    estilo_vias = {"color": "#222222", "linewidth": 0.5, "alpha": 0.8}
    vectores_costo = []
    if ruta is not None:
        vectores_costo = [(ruta, {"color": "white", "linewidth": 4}),
                          (ruta, {"color": "#e91e63", "linewidth": 2})]
    panel_costo.vectores = vectores_costo
    paneles = [
        Panel(relieve["elevacion"], "Elevación y red vial (OSM)", "terrain", etiqueta="m",
              vectores=[(vias_gdf, estilo_vias)] if vias_gdf is not None else []),
        Panel(relieve["pendiente"], "Pendiente", "magma", 0, 40, "grados"),
        Panel(cob, "Cobertura (ESA WorldCover 2021)", clases=CLASES_WORLDCOVER),
        panel_costo,
    ]
    if humedad is not None:
        paneles.insert(3, Panel(humedad, "Humedad (proxy Sentinel-1 VV)", "YlGnBu", 0, 1, "índice"))
    titulo = "Transitabilidad %s | %s" % (modo, aoi.nombre)
    if ruta is not None:
        titulo += " | ruta %.1f km, %.1f h" % (ruta["distancia_km"].iloc[0], ruta["tiempo_h"].iloc[0])
        mostrar(panel_costo)
    png = guardar_resumen_png(
        paneles, MODULO, aoi, titulo,
        pie="AOI aproximada. Tiempos estimados con parámetros genéricos: validar en terreno. "
            "Contiene datos Copernicus, ESA WorldCover y © OpenStreetMap.",
    )
    indicadores = {
        "modo": modo,
        "transitable_pct": round(100 * fraccion_transitable(costo), 1),
        "pendiente_media_grados": float(np.nanmean(relieve["pendiente"].values)),
        "vias_km": float(vias_gdf.to_crs(aoi.crs_utm).length.sum() / 1000) if vias_gdf is not None
        and len(vias_gdf) else 0.0,
    }
    if ruta is not None:
        indicadores["ruta_km"] = float(ruta["distancia_km"].iloc[0])
        indicadores["ruta_horas"] = float(ruta["tiempo_h"].iloc[0])
    finitos = costo.values[np.isfinite(costo.values)]
    capas_web = [
        {"archivo": "costo_%s.tif" % modo, "titulo": "Costo de desplazamiento (%s), s/m" % modo,
         "cmap": "viridis_r", "vmin": 0.0,
         "vmax": float(np.percentile(finitos, 97)) if finitos.size else 1.0, "visible": True},
        {"archivo": "pendiente.tif", "titulo": "Pendiente (grados)", "cmap": "magma",
         "vmin": 0.0, "vmax": 40.0, "visible": False},
        {"archivo": "cobertura.tif", "titulo": "Cobertura ESA WorldCover", "visible": False,
         "clases": {str(k): v[0] for k, v in CLASES_WORLDCOVER.items()}},
    ]
    if humedad is not None:
        capas_web.append({"archivo": "humedad_proxy.tif", "titulo": "Humedad (proxy S1)",
                          "cmap": "YlGnBu", "vmin": 0.0, "vmax": 1.0, "visible": False})
    if ruta is not None:
        capas_web.append({"archivo": "ruta_%s.geojson" % modo, "titulo": "Ruta de menor costo",
                          "color": "#ff2d78", "grosor": 4, "visible": True})
    registrar_corrida(MODULO, aoi, titulo, indicadores, archivos + [png], avisos,
                      {"modo": modo, "origen": origen, "destino": destino,
                       "fechas_humedad": fechas_humedad, "resolucion_m": resolucion_m}, FUENTES,
                      capas_web=capas_web)
    p.fin("terreno listo")
    return {"relieve": relieve, "cobertura": cob, "humedad": humedad, "vias": vias_gdf,
            "costo": costo, "ruta": ruta, "png": png, "archivos": archivos,
            "indicadores": indicadores, "avisos": avisos}
