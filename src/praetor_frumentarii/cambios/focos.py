"""Focos de calor activos de NASA FIRMS (VIIRS y MODIS).

Dos formas de acceso:

- **Recientes, sin clave**: los CSV públicos de las últimas 24 h, 48 h o 7 días
  para Sudamérica. Es lo que usa el flujo por defecto.
- **Históricos, con clave**: la API de área de FIRMS, que pide una clave
  gratuita (MAP_KEY). Se lee de la variable de entorno ``FIRMS_MAP_KEY`` y
  nunca se escribe en los registros.

Un foco es un píxel con anomalía térmica (375 m en VIIRS, 1 km en MODIS): no
distingue un incendio forestal de una quema agrícola, una chimenea industrial
o un volcán, y no ve bajo nubes ni humo denso.
"""
from __future__ import annotations

import io
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core import AOI, cargar_aoi, obtener_logger

log = obtener_logger("cambios.focos")

BASE = "https://firms.modaps.eosdis.nasa.gov"
# sensor -> (carpeta, prefijo del CSV público, producto histórico de la API)
SENSORES: Dict[str, Tuple[str, str, str]] = {
    "viirs-noaa20": ("noaa-20-viirs-c2", "J1_VIIRS_C2", "VIIRS_NOAA20_SP"),
    "viirs-noaa21": ("noaa-21-viirs-c2", "J2_VIIRS_C2", "VIIRS_NOAA21_NRT"),
    "viirs-snpp": ("suomi-npp-viirs-c2", "SUOMI_VIIRS_C2", "VIIRS_SNPP_SP"),
    "modis": ("modis-c6.1", "MODIS_C6_1", "MODIS_SP"),
}
PERIODOS = ("24h", "48h", "7d")
SENSORES_POR_DEFECTO = ("viirs-noaa20", "viirs-noaa21", "modis")
DIAS_PERSISTENTE = 3
COLUMNAS = ["sensor", "fecha_utc", "frp", "confianza", "dia_noche", "satelite"]


def _vacio():
    import geopandas as gpd

    return gpd.GeoDataFrame({c: [] for c in COLUMNAS}, geometry=[], crs=4326)


def _confianza(valor: Any) -> str:
    """Unifica la confianza: VIIRS usa l/n/h; MODIS un porcentaje de 0 a 100."""
    texto = str(valor).strip().lower()
    if texto in ("l", "low"):
        return "baja"
    if texto in ("n", "nominal"):
        return "nominal"
    if texto in ("h", "high"):
        return "alta"
    try:
        numero = float(texto)
    except ValueError:
        return "nominal"
    return "baja" if numero < 30 else ("alta" if numero >= 80 else "nominal")


def leer_csv_firms(texto: str, sensor: str = ""):
    """Convierte un CSV de FIRMS en GeoDataFrame de puntos (EPSG:4326)."""
    import geopandas as gpd
    import pandas as pd

    if not texto.strip() or "latitude" not in texto.splitlines()[0]:
        # La API responde con una línea de texto cuando la clave o la consulta son inválidas.
        primera = texto.strip().splitlines()[0][:120] if texto.strip() else "respuesta vacía"
        raise ValueError("FIRMS no devolvió un CSV de focos: %s" % primera)
    tabla = pd.read_csv(io.StringIO(texto))
    if tabla.empty:
        return _vacio()
    hora = tabla["acq_time"].astype(int).astype(str).str.zfill(4)
    fecha = pd.to_datetime(tabla["acq_date"].astype(str) + " " + hora, format="%Y-%m-%d %H%M")
    salida = gpd.GeoDataFrame(
        {
            "sensor": sensor or tabla.get("instrument", pd.Series([""] * len(tabla))).astype(str),
            "fecha_utc": fecha.dt.strftime("%Y-%m-%d %H:%M"),
            "frp": pd.to_numeric(tabla["frp"], errors="coerce").fillna(0.0).round(2),
            "confianza": tabla["confidence"].map(_confianza),
            "dia_noche": tabla["daynight"].astype(str).map({"D": "dia", "N": "noche"}).fillna(""),
            "satelite": tabla["satellite"].astype(str),
        },
        geometry=gpd.points_from_xy(tabla["longitude"], tabla["latitude"]), crs=4326,
    )
    return salida


def filtrar_aoi(focos, aoi: Union[AOI, str]):
    """Deja solo los focos dentro de la geometría de la AOI."""
    aoi = cargar_aoi(aoi)
    if len(focos) == 0:
        return focos
    dentro = focos[focos.geometry.within(aoi.geometria)]
    return dentro.sort_values("fecha_utc").reset_index(drop=True)


def _descargar(url: str, tiempo: float = 60.0) -> str:
    pedido = urllib.request.Request(url, headers={"User-Agent": "frumentarii/0.1"})
    with urllib.request.urlopen(pedido, timeout=tiempo) as r:
        return r.read().decode("utf-8", "replace")


def focos_recientes(
    aoi: Union[AOI, str], periodo: str = "24h", sensores: Sequence[str] = SENSORES_POR_DEFECTO
):
    """Focos de las últimas 24 h, 48 h o 7 días dentro de la AOI. No requiere clave."""
    import geopandas as gpd
    import pandas as pd

    if periodo not in PERIODOS:
        raise ValueError("periodo debe ser uno de %s" % (PERIODOS,))
    aoi = cargar_aoi(aoi)
    partes = []
    for sensor in sensores:
        if sensor not in SENSORES:
            raise ValueError("sensor desconocido '%s'. Opciones: %s" % (sensor, sorted(SENSORES)))
        carpeta, prefijo, _ = SENSORES[sensor]
        url = "%s/data/active_fire/%s/csv/%s_South_America_%s.csv" % (BASE, carpeta, prefijo, periodo)
        try:
            todos = leer_csv_firms(_descargar(url), sensor)
        except (urllib.error.URLError, ValueError, OSError) as e:
            log.warning("  %s: no disponible (%s)", sensor, type(e).__name__)
            continue
        en_aoi = filtrar_aoi(todos, aoi)
        log.info("  %s: %d focos en Sudamérica, %d en la AOI", sensor, len(todos), len(en_aoi))
        if len(en_aoi):
            partes.append(en_aoi)
    if not partes:
        return _vacio()
    return gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=4326) \
        .sort_values("fecha_utc").reset_index(drop=True)


def focos_historicos(
    aoi: Union[AOI, str], fecha: str, dias: int = 1, sensores: Sequence[str] = ("viirs-snpp",),
    clave: Optional[str] = None,
):
    """Focos desde ``fecha`` (AAAA-MM-DD) durante ``dias`` días, por la API de FIRMS.

    Requiere una clave gratuita en ``FIRMS_MAP_KEY`` (se pide en
    https://firms.modaps.eosdis.nasa.gov/api/map_key/). La API entrega como
    máximo 5 días por consulta, así que se divide en tramos.
    """
    import datetime as dt

    import geopandas as gpd
    import pandas as pd

    clave = clave or os.environ.get("FIRMS_MAP_KEY")
    if not clave:
        raise LookupError(
            "Para focos históricos se necesita una clave gratuita de FIRMS en la variable de "
            "entorno FIRMS_MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/map_key/)"
        )
    aoi = cargar_aoi(aoi)
    caja = ",".join("%.4f" % v for v in aoi.bbox)
    inicio = dt.date.fromisoformat(fecha)
    partes = []
    for sensor in sensores:
        producto = SENSORES[sensor][2]
        for salto in range(0, int(dias), 5):
            tramo = min(5, int(dias) - salto)
            dia = (inicio + dt.timedelta(days=salto)).isoformat()
            url = "%s/api/area/csv/%s/%s/%s/%d/%s" % (BASE, clave, producto, caja, tramo, dia)
            try:
                partes.append(filtrar_aoi(leer_csv_firms(_descargar(url), sensor), aoi))
            except urllib.error.URLError as e:
                # No se registra la URL: contiene la clave.
                raise LookupError("La API de FIRMS falló para %s %s (%s)" % (producto, dia,
                                                                            type(e).__name__))
        log.info("  %s: %d focos entre %s y %d día(s) después", sensor,
                 sum(len(p) for p in partes), fecha, dias)
    partes = [p for p in partes if len(p)]
    if not partes:
        return _vacio()
    return gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=4326) \
        .sort_values("fecha_utc").reset_index(drop=True)


def agrupar_focos(focos, crs_metrico: Any, distancia_m: float = 750.0):
    """Une focos vecinos en frentes: polígonos con conteo, potencia y horario.

    Dos focos quedan en el mismo grupo si están a menos de ``2 * distancia_m``.
    Devuelve un GeoDataFrame en EPSG:4326 ordenado por potencia radiativa.

    ``dias_distintos`` cuenta en cuántos días UTC se detectó el grupo, y
    ``persistente`` vale True desde ``DIAS_PERSISTENTE``. Ajuste tras la
    validación con datos reales (Chile, noviembre de 2026): los grupos más
    intensos eran el volcán Nevados de Chillán y la fundición Caletones, que
    arden en el mismo lugar todos los días. Un incendio forestal se mueve o
    se apaga; una fuente que no cambia de sitio suele ser industria, un volcán
    o una antorcha de gas.
    """
    import geopandas as gpd

    columnas = ["focos", "frp_total", "primera_utc", "ultima_utc", "area_ha", "dias_distintos",
                "persistente"]
    if len(focos) == 0:
        return gpd.GeoDataFrame({c: [] for c in columnas}, geometry=[], crs=4326)
    metrico = focos.to_crs(crs_metrico)
    union = metrico.geometry.buffer(distancia_m).unary_union
    poligonos = list(union.geoms) if hasattr(union, "geoms") else [union]
    grupos = gpd.GeoDataFrame({"grupo": range(len(poligonos))}, geometry=poligonos, crs=crs_metrico)
    cruce = gpd.sjoin(metrico, grupos, predicate="within")
    filas = []
    for g, sub in cruce.groupby("grupo"):
        filas.append({
            "focos": int(len(sub)), "frp_total": round(float(sub["frp"].sum()), 1),
            "primera_utc": sub["fecha_utc"].min(), "ultima_utc": sub["fecha_utc"].max(),
            "area_ha": round(float(poligonos[int(g)].area) / 1e4, 1),
            "dias_distintos": int(sub["fecha_utc"].str[:10].nunique()),
            "persistente": bool(sub["fecha_utc"].str[:10].nunique() >= DIAS_PERSISTENTE),
            "geometry": poligonos[int(g)],
        })
    salida = gpd.GeoDataFrame(filas, geometry="geometry", crs=crs_metrico).to_crs(4326)
    return salida.sort_values("frp_total", ascending=False).reset_index(drop=True)


def resumen_focos(focos) -> Dict[str, Any]:
    """Indicadores simples de un conjunto de focos."""
    if len(focos) == 0:
        return {"focos": 0, "frp_total_mw": 0.0, "alta_confianza": 0, "ultima_deteccion_utc": None}
    return {
        "focos": int(len(focos)),
        "frp_total_mw": round(float(np.nansum(focos["frp"])), 1),
        "alta_confianza": int((focos["confianza"] == "alta").sum()),
        "ultima_deteccion_utc": str(focos["fecha_utc"].max()),
    }
