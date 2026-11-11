"""Flujo completo marino: manchas oscuras por fecha y recurrencia."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from ..core import (
    AOI, Panel, Progreso, a_2d, a_decibeles, buscar, cargar_aoi, cargar_s1, como,
    guardar_geojson, guardar_geotiff, guardar_resumen_png, obtener_logger, registrar_corrida,
)
from .manchas import describir_manchas, manchas_oscuras
from .recurrencia import recurrencia
from .tierra import mascara_tierra
from .viento import apto_por_viento, estado_mar_por_sar, viento_era5

log = obtener_logger("marino")
MODULO = "marino"
FUENTES = [
    "Contiene datos Copernicus Sentinel-1 modificados (ESA); corrección radiométrica de terreno "
    "(RTC) de Microsoft Planetary Computer",
    "Máscara de tierra: global-land-mask (GLOBE, NOAA, dominio público) o archivo costero propio",
]


def ejecutar(
    aoi: Union[AOI, str],
    fechas: str,
    max_escenas: int = 4,
    contraste_db: float = 4.0,
    area_min_ha: float = 10.0,
    costa: Union[str, Path] = "global-land-mask",
    margen_m: float = 500.0,
    viento_ms: Optional[float] = None,
    archivo_era5: Optional[Union[str, Path]] = None,
    resolucion_m: float = 40.0,
    visor: Any = None,
) -> Dict[str, Any]:
    """Detecta manchas oscuras en cada escena Sentinel-1 del período y su recurrencia."""
    import geopandas as gpd
    import pandas as pd

    aoi = cargar_aoi(aoi)
    avisos: List[str] = []
    items = buscar(aoi, "sentinel-1-rtc", fechas, catalogo="planetary")
    if not items:
        raise LookupError("Sin escenas Sentinel-1 RTC para %s en %s" % (aoi.nombre, fechas))
    # Una escena por día, las más recientes.
    por_dia: Dict[str, List[Any]] = {}
    for it in items:
        por_dia.setdefault(it.datetime.strftime("%Y-%m-%d"), []).append(it)
    dias = sorted(por_dia)[-max_escenas:]
    p = Progreso(len(dias) + 2, log)

    tierra: Optional[xr.DataArray] = None
    mascaras: List[xr.DataArray] = []
    validos: List[xr.DataArray] = []
    tablas = []
    detalle: List[Dict[str, Any]] = []
    ultima_db: Optional[xr.DataArray] = None
    for dia in dias:
        p.paso("escena %s: carga, máscara de tierra y detección" % dia)
        ds = cargar_s1(aoi, dia, ("vv",), resolucion_m, items=por_dia[dia])
        vv_db = a_decibeles(a_2d(ds["vv"]) if ds.sizes["time"] == 1 else ds["vv"].median("time"))
        vv_db = como(vv_db.values, a_2d(ds["vv"].isel(time=[0])), "vv_db")
        if tierra is None:
            tierra = mascara_tierra(vv_db, costa, margen_m)
        elif vv_db.shape != tierra.shape:
            avisos.append("La escena %s tiene otra grilla y se omite" % dia)
            continue
        r = manchas_oscuras(vv_db, tierra, contraste_db, area_min_ha=area_min_ha)
        viento = viento_ms if viento_ms is not None else viento_era5(aoi, dia, archivo_era5)
        apto = apto_por_viento(viento)
        if apto is None:
            estado, apto_sar = estado_mar_por_sar(r.attrs["mar_mediana_db"])
            origen = "estimado desde SAR"
        else:
            estado, apto_sar, origen = "viento %.1f m/s" % viento, apto, "ERA5 o usuario"
        log.info("  %s (%s)", estado, origen)
        usable = apto_sar is not False and not r.attrs["descartada_por_calma"]
        if r.attrs["descartada_por_calma"]:
            avisos.append("%s: zona oscura muy extensa (calma generalizada); escena descartada" % dia)
        elif apto_sar is False:
            avisos.append("%s: %s; escena descartada" % (dia, estado))
        mancha = r["mancha"] if usable else como(np.zeros(vv_db.shape, bool), vv_db, "mancha")
        tabla = describir_manchas(xr.Dataset({"mancha": mancha, "contraste": r["contraste"]})
                                  .rio.write_crs(vv_db.rio.crs), dia)
        log.info("  %d mancha(s), %.0f ha", len(tabla), float(tabla["area_ha"].sum()) if len(tabla) else 0)
        mar = np.isfinite(r["contraste"].values)
        if usable:
            mascaras.append(mancha)
            validos.append(como(mar, vv_db))
            tablas.append(tabla)
        detalle.append({"fecha": dia, "estado_mar": estado, "usable": usable,
                        "manchas": int(len(tabla)) if usable else 0})
        ultima_db = vv_db
        if visor is not None:
            visor.mostrar(Panel(vv_db.where(~tierra.values), "Sentinel-1 VV %s (%s)" % (dia, estado),
                                "gray", -25, -5, "dB",
                                vectores=[(tabla, {"facecolor": "none", "edgecolor": "#ff1744",
                                                   "linewidth": 1.2})]))
    if ultima_db is None or tierra is None:
        raise LookupError("No se pudo cargar ninguna escena para %s" % aoi.nombre)

    p.paso("recurrencia multifecha")
    if mascaras:
        rec = recurrencia(mascaras, validos)
        todas = gpd.GeoDataFrame(pd.concat(tablas, ignore_index=True), crs=tablas[0].crs) \
            if any(len(t) for t in tablas) else tablas[0]
    else:
        avisos.append("Ninguna escena fue apta: no hay detecciones que reportar")
        cero = como(np.zeros(ultima_db.shape, bool), ultima_db)
        rec = recurrencia([cero], [cero])
        todas = describir_manchas(xr.Dataset({"mancha": cero, "contraste": ultima_db})
                                  .rio.write_crs(ultima_db.rio.crs))

    p.paso("guardando resultados")
    archivos = [guardar_geotiff(rec["conteo"], MODULO, aoi, "recurrencia_conteo"),
                guardar_geotiff(rec["frecuencia"], MODULO, aoi, "recurrencia_frecuencia"),
                guardar_geotiff(ultima_db, MODULO, aoi, "vv_db_ultima"),
                guardar_geojson(todas, MODULO, aoi, "manchas")]
    n_usables = len(mascaras)
    recurrentes_ha = float((rec["conteo"].values >= 2).sum()) * resolucion_m ** 2 / 1e4
    ultimas = todas[todas["fecha"] == dias[-1]] if len(todas) else todas
    conteo_vis = rec["conteo"].where(rec["conteo"] > 0)
    titulo = "Manchas oscuras en el mar | %s | %d escena(s) aptas de %d" % (
        aoi.nombre, n_usables, len(dias))
    png = guardar_resumen_png(
        [Panel(ultima_db, "Sentinel-1 VV %s" % dias[-1], "gray", -25, -5, "dB"),
         Panel(ultima_db.where(~tierra.values), "Mar (tierra y costa enmascaradas) y manchas",
               "gray", -25, -5, "dB",
               vectores=[(ultimas, {"facecolor": "none", "edgecolor": "#ff1744", "linewidth": 1.2})]),
         Panel(conteo_vis, "Recurrencia: fechas con mancha (de %d)" % max(n_usables, 1), "inferno",
               0, max(n_usables, 1), "fechas", fondo=np.dstack([np.where(tierra.values, 0.25, 0.05)] * 3),
               referencia=rec["conteo"])],
        MODULO, aoi, titulo,
        pie="AOI aproximada. Una mancha oscura NO confirma un derrame: calma, algas, hielo o lluvia "
            "se ven igual. Validar con otra fuente. Contiene datos Copernicus Sentinel-1 modificados.",
    )
    indicadores = {
        "escenas": len(dias), "escenas_aptas": n_usables,
        "manchas_total": int(len(todas)),
        "area_manchas_ha": float(todas["area_ha"].sum()) if len(todas) else 0.0,
        "area_recurrente_ha": recurrentes_ha,
        "mar_mediana_db": float(np.nanmedian(ultima_db.values[~tierra.values]))
        if (~tierra.values).any() else None,
    }
    if costa == "global-land-mask":
        avisos.append("Máscara de tierra global de cerca de 1 km: en canales y fiordos conviene "
                      "pasar un archivo costero con --costa")
    registrar_corrida(
        MODULO, aoi, titulo, indicadores, archivos + [png], avisos,
        {"fechas": fechas, "contraste_db": contraste_db, "area_min_ha": area_min_ha,
         "margen_m": margen_m, "escenas": "; ".join("%s: %s" % (d["fecha"], d["estado_mar"])
                                                     for d in detalle)},
        FUENTES, png=png,
        capas_web=[
            {"archivo": "vv_db_ultima.tif", "titulo": "Sentinel-1 VV (dB), última escena",
             "cmap": "gray", "vmin": -25.0, "vmax": -5.0, "visible": True, "opacidad": 0.9},
            {"archivo": "recurrencia_frecuencia.tif", "titulo": "Frecuencia de mancha",
             "cmap": "inferno", "vmin": 0.0, "vmax": 1.0, "visible": False},
            {"archivo": "manchas.geojson", "titulo": "Manchas oscuras detectadas",
             "color": "#ff1744", "relleno": 0.25, "visible": True},
        ],
    )
    p.fin("marino listo")
    return {"recurrencia": rec, "manchas": todas, "tierra": tierra, "detalle": detalle,
            "png": png, "archivos": archivos, "indicadores": indicadores, "avisos": avisos}
