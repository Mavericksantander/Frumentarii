"""Exposición: qué infraestructura civil hay dentro o cerca de cada hallazgo.

Fuente: OpenStreetMap. Solo infraestructura civil de uso público (salud,
educación, bomberos, red eléctrica, rutas principales y localidades). OSM puede
estar incompleto: que no figure nada no garantiza que no haya nada.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core import obtener_logger

log = obtener_logger("inteligencia.exposicion")

ETIQUETAS_OSM = {
    "amenity": ["hospital", "clinic", "doctors", "school", "kindergarten", "college", "university",
                "fire_station"],
    "power": ["substation", "plant", "line"],
    "highway": ["motorway", "trunk", "primary", "secondary"],
    "place": ["city", "town", "village", "hamlet"],
}
SALUD = {"hospital", "clinic", "doctors"}
EDUCACION = {"school", "kindergarten", "college", "university"}
# Peso de cada elemento en el índice de exposición (por unidad o por km).
PESOS = {"salud": 3.0, "educacion": 2.0, "bomberos": 1.0, "subestaciones": 2.0,
         "localidades": 2.0, "rutas_km": 0.3, "lineas_km": 0.2}
CAMPOS = ["salud", "educacion", "bomberos", "subestaciones", "rutas_km", "lineas_km",
          "localidades", "nombres", "exposicion"]


def clasificar(fila: Dict[str, Any]) -> Optional[str]:
    """Categoría de exposición de un elemento OSM, o None si no interesa."""
    def valor(clave: str) -> str:
        v = fila.get(clave)
        return v if isinstance(v, str) else ""

    amenity, power, highway, place = valor("amenity"), valor("power"), valor("highway"), valor("place")
    if amenity in SALUD:
        return "salud"
    if amenity in EDUCACION:
        return "educacion"
    if amenity == "fire_station":
        return "bomberos"
    if power in ("substation", "plant"):
        return "subestaciones"
    if power == "line":
        return "lineas"
    if highway in ETIQUETAS_OSM["highway"]:
        return "rutas"
    if place in ETIQUETAS_OSM["place"]:
        return "localidades"
    return None


def descargar_infraestructura(geometria: Any):
    """Elementos OSM de interés dentro de ``geometria`` (EPSG:4326), ya clasificados."""
    import geopandas as gpd
    import osmnx as ox

    from ..core.cache import carpeta_cache

    ox.settings.cache_folder = str(carpeta_cache() / "osm")
    ox.settings.log_console = False
    # Overpass suele estar saturado: se falla rápido en vez de esperar turno varios minutos.
    ox.settings.overpass_rate_limit = False
    ox.settings.requests_timeout = 45
    try:
        crudo = ox.features_from_polygon(geometria, ETIQUETAS_OSM)
    except Exception as e:
        if "No matching features" in str(e) or "InsufficientResponse" in type(e).__name__:
            return gpd.GeoDataFrame({"clase": [], "nombre": []}, geometry=[], crs=4326)
        raise
    crudo = crudo.reset_index(drop=True)
    crudo["clase"] = [clasificar(f) for f in crudo.to_dict("records")]
    # Los nombres faltantes llegan como NA de pandas, que no se puede evaluar como booleano.
    crudo["nombre"] = [n if isinstance(n, str) else "" for n in crudo["name"]] \
        if "name" in crudo else ""
    return crudo[crudo["clase"].notna()][["clase", "nombre", "geometry"]].reset_index(drop=True)


def indice_exposicion(conteos: Dict[str, float]) -> float:
    """Índice 0 a 1 que satura: muchos elementos no suman indefinidamente."""
    suma = sum(PESOS[k] * float(conteos.get(k, 0.0)) for k in PESOS)
    return float(1.0 - np.exp(-suma / 5.0))


def exposicion_de(geometria_m: Any, infra_m, margen_m: float = 500.0,
                  radio_localidades_m: float = 5000.0) -> Dict[str, Any]:
    """Exposición de un hallazgo. Geometría e infraestructura en el mismo CRS métrico.

    Cuenta lo que cae dentro del hallazgo ampliado en ``margen_m`` y las
    localidades a menos de ``radio_localidades_m``.
    """
    vacio: Dict[str, Any] = {k: 0 for k in CAMPOS}
    vacio.update(rutas_km=0.0, lineas_km=0.0, nombres="", exposicion=0.0)
    if infra_m is None or len(infra_m) == 0:
        return vacio
    zona = geometria_m.buffer(margen_m)
    cerca = infra_m[infra_m.geometry.intersects(zona)]
    r = dict(vacio)
    for clase in ("salud", "educacion", "bomberos", "subestaciones"):
        r[clase] = int((cerca["clase"] == clase).sum())
    for clase, campo in (("rutas", "rutas_km"), ("lineas", "lineas_km")):
        lineas = cerca[cerca["clase"] == clase]
        r[campo] = round(float(lineas.geometry.intersection(zona).length.sum()) / 1000.0, 2)
    pueblos = infra_m[(infra_m["clase"] == "localidades")
                      & (infra_m.geometry.distance(geometria_m) <= radio_localidades_m)]
    r["localidades"] = int(len(pueblos))
    nombres = [n for n in dict.fromkeys(list(pueblos["nombre"]) + list(
        cerca[cerca["clase"].isin(["salud", "educacion", "subestaciones"])]["nombre"]))
               if isinstance(n, str) and n]
    r["nombres"] = ", ".join(nombres[:6])
    r["exposicion"] = round(indice_exposicion(r), 3)
    return r


def describir(r: Dict[str, Any]) -> str:
    """Frase corta con lo expuesto, para el parte."""
    partes: List[str] = []
    for campo, singular, plural in (("salud", "centro de salud", "centros de salud"),
                                    ("educacion", "establecimiento educacional",
                                     "establecimientos educacionales"),
                                    ("bomberos", "cuartel de bomberos", "cuarteles de bomberos"),
                                    ("subestaciones", "instalación eléctrica",
                                     "instalaciones eléctricas"),
                                    ("localidades", "localidad a menos de 5 km",
                                     "localidades a menos de 5 km")):
        n = int(r.get(campo, 0) or 0)
        if n:
            partes.append("%d %s" % (n, singular if n == 1 else plural))
    for campo, texto in (("rutas_km", "km de ruta principal"), ("lineas_km", "km de línea eléctrica")):
        km = float(r.get(campo, 0) or 0)
        if km >= 0.1:
            partes.append(("%.1f %s" % (km, texto)).replace(".", ","))
    return "; ".join(partes) if partes else "sin infraestructura registrada en OpenStreetMap"


def calcular_exposicion(hallazgos, infra=None, maximo_consultas: int = 10,
                        area_max_km2: float = 4000.0, tiempo_max_s: float = 150.0,
                        max_sueltos: int = 4):
    """Agrega las columnas de exposición a los hallazgos.

    Con ``infra`` se usa esa tabla. Sin ella se consulta OpenStreetMap solo
    para las alertas (el contexto no la necesita), agrupadas por AOI de origen:
    una consulta por grupo, o por hallazgo si el grupo es demasiado extenso,
    empezando por lo más severo.

    Ajuste tras la validación con datos reales: el servidor Overpass responde
    lento o con "502 at capacity" a menudo, y la primera evaluación quedó más
    de 10 minutos esperando. Ahora cada consulta falla a los 45 s y el total
    se corta en ``tiempo_max_s``. Lo que no se alcanza a consultar queda como
    "no evaluado" (``exposicion_evaluada = False``), nunca como "sin exposición".
    """
    import time

    from shapely.geometry import box

    from .fusion import CRS_METRICO

    h = hallazgos.copy()
    for campo in CAMPOS:
        h[campo] = "" if campo == "nombres" else 0.0
    h["exposicion_evaluada"] = False
    if len(h) == 0:
        return h
    metrico = h.to_crs(CRS_METRICO)

    def aplicar(indices: List[Any], tabla) -> None:
        tabla_m = tabla.to_crs(CRS_METRICO) if len(tabla) else tabla
        for i in indices:
            r = exposicion_de(metrico.geometry[i], tabla_m)
            for campo, valor in r.items():
                h.at[i, campo] = valor
            h.at[i, "exposicion_evaluada"] = True

    if infra is not None:
        aplicar(list(h.index), infra)
        return h

    alertas = h[h["categoria"] == "alerta"].sort_values("severidad", ascending=False)
    zonas: List[Tuple[str, List[Any], Any]] = []      # una consulta cubre muchos hallazgos
    sueltos: List[Tuple[str, List[Any], Any]] = []    # una consulta por hallazgo (zona extensa)
    for aoi, grupo in alertas.groupby("aoi", sort=False):
        oeste, sur, este, norte = grupo.total_bounds
        area_km2 = (este - oeste + 0.12) * 111.0 * (norte - sur + 0.12) * 111.0 * 0.8
        if area_km2 > area_max_km2:
            sueltos += [(str(aoi), [i], box(*h.geometry[i].bounds).buffer(0.06))
                        for i in grupo.index[:max_sueltos]]
        else:
            zonas.append((str(aoi), list(grupo.index),
                          box(oeste - 0.06, sur - 0.06, este + 0.06, norte + 0.06)))
    lotes = zonas + sueltos                           # primero las consultas que más rinden
    inicio = time.time()
    for n, (aoi, indices, geometria) in enumerate(lotes):
        if n >= maximo_consultas or time.time() - inicio > tiempo_max_s:
            log.info("  límite de consultas a OSM alcanzado: %d lote(s) quedan sin evaluar",
                     len(lotes) - n)
            break
        try:
            tabla = descargar_infraestructura(geometria)
            log.info("  OSM %s: %d elementos de infraestructura civil", aoi, len(tabla))
            aplicar(indices, tabla)
        except Exception as e:
            log.warning("  OSM no respondió para %s (%s): queda sin evaluar", aoi, type(e).__name__)
    return h
