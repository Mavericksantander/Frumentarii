"""Hallazgos normalizados: un formato común para lo que detecta cada módulo.

Cada módulo deja polígonos con atributos propios. Aquí se leen todas las
corridas, se agrupan los polígonos vecinos en *eventos* (un incendio es un
evento, no 68 polígonos) y se llevan a una tabla única con tipo, fecha,
magnitud, severidad y una confianza de partida.

La confianza inicial refleja cuánto se puede creer en una sola fuente; la
corroboración (``fusion``) la sube o la baja.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core import obtener_logger
from ..core.salidas import carpeta_base

log = obtener_logger("inteligencia.hallazgos")

COLUMNAS = [
    "id", "tipo", "categoria", "aoi", "modulo", "fecha", "magnitud", "unidad", "severidad",
    "confianza", "fuentes", "evidencia", "reservas", "partes",
]


@dataclass(frozen=True)
class Regla:
    """Cómo convertir un GeoJSON de un módulo en hallazgos."""

    archivo: str            # nombre del GeoJSON dentro de la carpeta de la corrida
    tipo: str
    categoria: str          # 'alerta' (posible emergencia) o 'contexto'
    campo_magnitud: str     # atributo que se suma por evento
    unidad: str
    referencia: float       # magnitud que corresponde a severidad 0,5
    confianza: float        # confianza de partida con una sola fuente
    union_m: float          # distancia para unir polígonos en un evento
    fuente: str
    campo_fecha: Optional[str] = None
    maximo: int = 40        # eventos que se conservan (los de mayor magnitud)


# (modulo, analisis) -> reglas. La referencia fija la escala de severidad por tipo.
REGLAS: Dict[tuple, List[Regla]] = {
    ("cambios", "quemas"): [
        Regla("area_quemada.geojson", "incendio", "alerta", "area_ha", "ha", 200.0, 0.60, 250.0,
              "dNBR Sentinel-2"),
    ],
    ("cambios", "focos"): [
        Regla("focos_frentes.geojson", "foco_calor", "alerta", "frp_total", "MW", 100.0, 0.45,
              0.0, "focos FIRMS (VIIRS/MODIS)", "ultima_utc", 60),
    ],
    ("cambios", "inundacion"): [
        Regla("inundacion.geojson", "inundacion", "alerta", "area_ha", "ha", 50.0, 0.50, 300.0,
              "agua nueva Sentinel-1/2"),
    ],
    ("cambios", "sar"): [
        Regla("sar_posible_dano_urbano.geojson", "posible_dano_urbano", "alerta", "area_ha", "ha",
              2.0, 0.25, 150.0, "proxy de coherencia Sentinel-1"),
        Regla("sar_cambio.geojson", "cambio_superficie", "contexto", "area_ha", "ha", 20.0, 0.30,
              200.0, "cambio de retrodispersión Sentinel-1", None, 20),
    ],
    ("marino", ""): [
        Regla("manchas.geojson", "mancha_marina", "alerta", "area_ha", "ha", 100.0, 0.25, 0.0,
              "mancha oscura Sentinel-1", "fecha"),
    ],
    ("suelo", "s2"): [
        Regla("s2_anomalias.geojson", "alteracion_mineral", "contexto", "area_ha", "ha", 50.0,
              0.40, 300.0, "Crosta Sentinel-2", None, 15),
    ],
}
ETIQUETAS = {
    "incendio": "Área quemada", "foco_calor": "Foco de calor activo",
    "fuente_termica_fija": "Fuente térmica fija", "inundacion": "Inundación",
    "posible_dano_urbano": "Posible daño en zona construida",
    "cambio_superficie": "Cambio de superficie (radar)", "mancha_marina": "Mancha oscura en el mar",
    "mancha_recurrente": "Mancha marina recurrente", "alteracion_mineral": "Alteración mineral",
}


def severidad_desde(magnitud: float, referencia: float) -> float:
    """Severidad 0 a 1, logarítmica: la referencia vale 0,5 y diez veces más, 1."""
    if not np.isfinite(magnitud) or magnitud <= 0 or referencia <= 0:
        return 0.0
    return float(np.clip(0.5 + 0.5 * np.log10(magnitud / referencia), 0.05, 1.0))


def _vacio():
    import geopandas as gpd

    return gpd.GeoDataFrame({c: [] for c in COLUMNAS}, geometry=[], crs=4326)


def _crs_metrico(gdf) -> Any:
    oeste, sur, este, norte = gdf.total_bounds
    zona = int(((oeste + este) / 2 + 180) // 6) % 60 + 1
    return "EPSG:%d" % ((32700 if (sur + norte) / 2 < 0 else 32600) + zona)


def agrupar_en_eventos(gdf, regla: Regla, fecha_corrida: str, aoi: str, modulo: str,
                       reservas: List[str]):
    """Une los polígonos de un GeoJSON en eventos y calcula sus atributos."""
    import geopandas as gpd

    if len(gdf) == 0:
        return _vacio()
    crs = _crs_metrico(gdf)
    metrico = gdf.to_crs(crs)
    metrico = metrico[~metrico.geometry.is_empty & metrico.geometry.notna()].reset_index(drop=True)
    if regla.union_m > 0:
        union = metrico.geometry.buffer(regla.union_m / 2.0).unary_union
        piezas = list(union.geoms) if hasattr(union, "geoms") else [union]
        zonas = gpd.GeoDataFrame({"evento": range(len(piezas))}, geometry=piezas, crs=crs)
        puntos = metrico.copy()
        puntos["geometry"] = metrico.geometry.representative_point()
        cruce = gpd.sjoin(puntos, zonas, predicate="within")
        metrico["evento"] = cruce["evento"].reindex(metrico.index).fillna(-1).astype(int)
    else:
        metrico["evento"] = np.arange(len(metrico))
    filas = []
    for _, sub in metrico[metrico["evento"] >= 0].groupby("evento"):
        geometria = sub.geometry.unary_union
        magnitud = float(sub[regla.campo_magnitud].sum()) if regla.campo_magnitud in sub else \
            float(geometria.area / 1e4)
        fecha = fecha_corrida
        if regla.campo_fecha and regla.campo_fecha in sub:
            fecha = str(sub[regla.campo_fecha].astype(str).max())[:16]
        fila = {
            "tipo": regla.tipo, "categoria": regla.categoria, "aoi": aoi, "modulo": modulo,
            "fecha": fecha, "magnitud": round(magnitud, 2), "unidad": regla.unidad,
            "severidad": round(severidad_desde(magnitud, regla.referencia), 3),
            "confianza": regla.confianza, "fuentes": regla.fuente,
            "evidencia": "", "reservas": " | ".join(reservas), "partes": int(len(sub)),
            "geometry": geometria,
        }
        # Atributos propios que usan las reglas de corroboración.
        for extra in ("persistente", "dias_distintos", "focos", "elongacion", "contraste_db"):
            if extra in sub:
                valores = sub[extra]
                fila[extra] = bool(valores.any()) if extra == "persistente" else float(valores.max())
        filas.append(fila)
    if not filas:
        return _vacio()
    eventos = gpd.GeoDataFrame(filas, geometry="geometry", crs=crs).to_crs(4326)
    return eventos.sort_values("magnitud", ascending=False).head(regla.maximo).reset_index(drop=True)


def _fecha_de(corrida: Dict[str, Any]) -> str:
    """Fecha del fenómeno observado (no la de la corrida)."""
    p = corrida.get("parametros", {})
    for clave in ("despues", "fechas"):
        valor = p.get(clave)
        if valor:
            return str(valor).split("/")[0]
    return str(corrida.get("generado", ""))[:10]


def recolectar(base: Optional[Path] = None):
    """Lee todas las corridas y devuelve los hallazgos como GeoDataFrame (EPSG:4326)."""
    import geopandas as gpd
    import pandas as pd

    base = Path(base) if base else carpeta_base()
    partes = []
    for ruta in sorted(base.glob("*/*/corrida*.json")):
        try:
            with open(ruta, encoding="utf-8") as f:
                corrida = json.load(f)
        except (OSError, ValueError):
            continue
        clave = (corrida.get("modulo", ""), corrida.get("analisis", ""))
        for regla in REGLAS.get(clave, []):
            archivo = ruta.parent / regla.archivo
            if not archivo.exists():
                continue
            try:
                gdf = gpd.read_file(archivo)
            except Exception as e:
                log.warning("no pude leer %s (%s)", archivo, type(e).__name__)
                continue
            # Los avisos de la corrida de focos hablan del conjunto, no de cada foco.
            reservas = [] if regla.tipo == "foco_calor" else list(corrida.get("avisos", []))
            eventos = agrupar_en_eventos(gdf, regla, _fecha_de(corrida), corrida.get("aoi", ""),
                                         clave[0], reservas)
            if len(eventos):
                log.info("  %s/%s %s: %d polígonos -> %d evento(s)", clave[0], corrida.get("aoi"),
                         regla.tipo, len(gdf), len(eventos))
                partes.append(eventos)
    if not partes:
        return _vacio()
    todos = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=4326)
    todos["id"] = ["H%03d" % (i + 1) for i in range(len(todos))]
    for columna in ("persistente",):
        if columna in todos:
            todos[columna] = todos[columna].fillna(False).astype(bool)
    return todos
