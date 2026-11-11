"""Flujo completo de suelo: compuesto, máscaras, índices, Crosta y zonas anómalas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from ..core import (
    AOI, Panel, Progreso, cargar_aoi, cargar_aster, cargar_landsat, como, componer_rgb,
    compuesto_mediana, compuesto_s2, estirar, guardar_geojson, guardar_geotiff,
    guardar_resumen_png, obtener_logger, registrar_corrida, vectorizar,
)
from ..cambios.inundacion import limpiar
from .crosta import crosta
from .indices import COCIENTES_ASTER, COCIENTES_OPTICOS, anomalias, indices_alteracion
from .mascaras import mascara_vegetacion_agua

log = obtener_logger("suelo")
MODULO = "suelo"
FUENTES = {
    "s2": "Contiene datos Copernicus Sentinel-2 modificados (ESA)",
    "landsat": "Landsat 8/9 Colección 2 Nivel 2, cortesía del U.S. Geological Survey (dominio público)",
    "aster": "ASTER L1T, NASA/METI/AIST/Japan Space Systems y equipo científico ASTER EE.UU./Japón "
             "(LP DAAC, uso libre)",
}


def cargar_compuesto(aoi: AOI, fechas: str, sensor: str, nubes_max: float, visor: Any) -> xr.Dataset:
    import rioxarray  # noqa: F401

    if sensor == "s2":
        return compuesto_s2(aoi, fechas, nubes_max=nubes_max, resolucion=20.0, max_escenas=3,
                            visor=visor)
    if sensor == "landsat":
        ds = cargar_landsat(aoi, fechas, nubes_max=nubes_max, limite=4)
        fechas_usadas = ", ".join(str(t)[:10] for t in ds["time"].values)
        comp = compuesto_mediana(ds).rio.write_crs(ds.rio.crs)
        comp.attrs.update(ds.attrs, fechas_usadas=fechas_usadas)
        return comp
    if sensor == "aster":
        return cargar_aster(aoi, fechas, nubes_max=nubes_max)
    raise ValueError("sensor debe ser 's2', 'landsat' o 'aster'")


def ejecutar(
    aoi: Union[AOI, str],
    fechas: Optional[str] = None,
    sensor: str = "s2",
    nubes_max: float = 10.0,
    percentil: float = 97.0,
    area_min_ha: float = 2.0,
    ndvi_max: float = 0.25,
    visor: Any = None,
) -> Dict[str, Any]:
    """Índices de alteración y Crosta sobre suelo expuesto, con zonas anómalas."""
    aoi = cargar_aoi(aoi)
    if fechas is None:
        fechas = "2000-03-01/2008-03-31" if sensor == "aster" else "2024-01-01/2024-03-31"
    avisos: List[str] = []
    p = Progreso(4, log)
    p.paso("compuesto %s (%s)" % (sensor, fechas))
    ds = cargar_compuesto(aoi, fechas, sensor, nubes_max, visor)
    es_aster = "b6" in ds
    rgb = np.nan_to_num(np.dstack([estirar(ds[b].values) for b in ("b3", "b2", "b1")])) \
        if es_aster else componer_rgb(ds)
    ref = ds["b1"] if es_aster else ds["rojo"]

    p.paso("máscara de vegetación, agua y sombras")
    suelo = mascara_vegetacion_agua(ds, ndvi_max=ndvi_max)
    log.info("  suelo expuesto: %.0f%% de la AOI", 100 * suelo.attrs["fraccion"])
    if suelo.attrs["fraccion"] < 0.2:
        avisos.append("Solo %.0f%% de suelo expuesto: los índices cubren poca superficie"
                      % (100 * suelo.attrs["fraccion"]))

    p.paso("cocientes de bandas y Crosta (PCA dirigido)")
    indices = indices_alteracion(ds, suelo)
    c_hierro = crosta(ds, "hierro", suelo)
    c_arcillas = crosta(ds, "arcillas", suelo)
    log.info("  Crosta hierro: CP%s [%s]", c_hierro.attrs.get("componente"), c_hierro.attrs.get("pesos"))
    log.info("  Crosta arcillas: CP%s [%s]", c_arcillas.attrs.get("componente"),
             c_arcillas.attrs.get("pesos"))
    zonas = []
    for nombre, capa in (("hierro", c_hierro), ("arcillas", c_arcillas)):
        m = limpiar(anomalias(capa, percentil), 4)
        zonas.append(vectorizar(m, area_min_ha, {"tipo": "anomalia_%s" % nombre,
                                                 "percentil": percentil}))
    import geopandas as gpd
    import pandas as pd

    poligonos = gpd.GeoDataFrame(pd.concat(zonas, ignore_index=True), crs=zonas[0].crs) \
        if any(len(z) for z in zonas) else zonas[0]

    descripciones = COCIENTES_ASTER if es_aster else COCIENTES_OPTICOS
    paneles = [Panel(rgb, "%s: %s" % (ds.attrs.get("sensor", sensor),
                                     "falso color B3-B2-B1" if es_aster else "color real"),
                     referencia=ref)]
    estilo = {"hierro": ("YlOrRd", "#b71c1c"), "arcillas": ("PuBuGn", "#006064")}
    for nombre in list(indices.data_vars)[:3]:
        v = indices[nombre].values
        lim = np.nanpercentile(v, [2, 98]) if np.isfinite(v).any() else (0, 1)
        panel = Panel(indices[nombre], descripciones[nombre][2],
                      "YlOrRd" if "hierro" in nombre else "PuBuGn", float(lim[0]), float(lim[1]),
                      "cociente", fondo=rgb * 0.35, referencia=ref)
        paneles.append(panel)
        if visor is not None:
            visor.mostrar(panel)
    for nombre, capa in (("hierro", c_hierro), ("arcillas", c_arcillas)):
        sub = poligonos[poligonos["tipo"] == "anomalia_%s" % nombre] if len(poligonos) else poligonos
        panel = Panel(capa, "Crosta %s (CP%s)" % (nombre, capa.attrs.get("componente", "?")),
                      estilo[nombre][0], -1.0, 3.0, "puntaje z", fondo=rgb * 0.35, referencia=ref,
                      vectores=[(sub, {"facecolor": "none", "edgecolor": estilo[nombre][1],
                                       "linewidth": 0.7})])
        paneles.append(panel)
        if visor is not None:
            visor.mostrar(panel)

    p.paso("guardando resultados")
    archivos = [guardar_geotiff(indices[n], MODULO, aoi, "%s_%s" % (sensor, n)) for n in indices.data_vars]
    archivos += [guardar_geotiff(c_hierro, MODULO, aoi, "%s_crosta_hierro" % sensor),
                 guardar_geotiff(c_arcillas, MODULO, aoi, "%s_crosta_arcillas" % sensor),
                 guardar_geojson(poligonos, MODULO, aoi, "%s_anomalias" % sensor)]
    titulo = "Alteración mineral (%s) | %s | %d zonas anómalas" % (
        ds.attrs.get("sensor", sensor), aoi.nombre, len(poligonos))
    png = guardar_resumen_png(
        paneles, MODULO, aoi, titulo, nombre="resumen_%s" % sensor,
        pie="AOI aproximada. Indicadores espectrales, no identificación mineral: caminos, "
            "relaves y polvo dan respuestas similares. Validar con muestreo en terreno.",
    )
    indicadores: Dict[str, Any] = {
        "sensor": ds.attrs.get("sensor", sensor),
        "suelo_expuesto_pct": round(100 * suelo.attrs["fraccion"], 1),
        "zonas_hierro": int((poligonos["tipo"] == "anomalia_hierro").sum()) if len(poligonos) else 0,
        "zonas_arcillas": int((poligonos["tipo"] == "anomalia_arcillas").sum()) if len(poligonos) else 0,
        "area_anomala_ha": float(poligonos["area_ha"].sum()) if len(poligonos) else 0.0,
    }
    if es_aster:
        avisos.append("ASTER: reflectancia aparente sin corrección atmosférica y escenas "
                      "antiguas (2000 a 2008); comparar con Sentinel-2 antes de concluir")
    capas_web = [
        {"archivo": "%s_crosta_arcillas.tif" % sensor, "titulo": "Crosta arcillas (puntaje z)",
         "cmap": "PuBuGn", "vmin": 0.5, "vmax": 3.0, "visible": True},
        {"archivo": "%s_crosta_hierro.tif" % sensor, "titulo": "Crosta hierro (puntaje z)",
         "cmap": "YlOrRd", "vmin": 0.5, "vmax": 3.0, "visible": False},
        {"archivo": "%s_anomalias.geojson" % sensor, "titulo": "Zonas anómalas",
         "color": "#ffea00", "relleno": 0.0, "visible": True},
    ]
    registrar_corrida(
        MODULO, aoi, titulo, indicadores, archivos + [png], avisos,
        {"fechas": fechas, "sensor": sensor, "fechas_usadas": ds.attrs.get("fechas_usadas"),
         "percentil": percentil, "crosta_hierro": c_hierro.attrs.get("pesos"),
         "crosta_arcillas": c_arcillas.attrs.get("pesos")},
        [FUENTES[sensor]], analisis=sensor, png=png, capas_web=capas_web,
    )
    p.fin("suelo listo")
    return {"compuesto": ds, "suelo_expuesto": suelo, "indices": indices,
            "crosta_hierro": c_hierro, "crosta_arcillas": c_arcillas, "anomalias": poligonos,
            "png": png, "archivos": archivos, "indicadores": indicadores, "avisos": avisos}
