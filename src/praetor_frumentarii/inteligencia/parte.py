"""Evaluación completa y parte de situación."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .. import NOMBRE
from ..core import (
    AOI, Panel, Progreso, aoi_desde_bbox, cargar_aoi, carpeta_salida, crear_raster,
    guardar_geojson, guardar_resumen_png, obtener_logger, registrar_corrida,
)
from .exposicion import calcular_exposicion, describir
from .fusion import ajustar_por_instalaciones, corroborar
from .hallazgos import ETIQUETAS, recolectar
from .prioridad import priorizar, resumen

log = obtener_logger("inteligencia")
MODULO = "inteligencia"
FUENTES = [
    "Hallazgos de los módulos de PRÆTOR Frumentarii (ver las fuentes de cada corrida)",
    "Infraestructura civil: © Colaboradores de OpenStreetMap (ODbL)",
]


def _num(valor: float, decimales: int = 1) -> str:
    return ("%.*f" % (decimales, valor)).replace(".", ",")


def redactar_parte(hallazgos, zona: str, maximo: int = 15) -> str:
    """Texto Markdown del parte de situación."""
    ahora = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    alertas = hallazgos[hallazgos["categoria"] == "alerta"] if len(hallazgos) else hallazgos
    contexto = hallazgos[hallazgos["categoria"] != "alerta"] if len(hallazgos) else hallazgos
    L: List[str] = [
        "# Parte de situación: %s" % zona, "",
        "%s | generado %s | uso civil" % (NOMBRE, ahora), "",
        "> Evaluación automática a partir de imágenes públicas de 10 m a 1 km. Ordena qué revisar "
        "primero; no confirma daños. Todo hallazgo debe verificarse antes de decidir.", "",
        "## Resumen", "",
    ]
    if len(alertas) == 0:
        L += ["No hay alertas con las corridas disponibles.", ""]
    else:
        por_nivel = alertas["nivel"].value_counts()
        L += ["- %d alerta(s): %d de confianza alta, %d media y %d baja." % (
            len(alertas), por_nivel.get("alta", 0), por_nivel.get("media", 0),
            por_nivel.get("baja", 0)),
              "- %d hallazgo(s) de contexto (fuentes fijas, cambios sin clasificar, geología)."
              % len(contexto), ""]
        L += ["| N.º | Prioridad | Qué | Zona | Fecha | Magnitud | Confianza |",
              "|---|---|---|---|---|---|---|"]
        for n, (_, f) in enumerate(alertas.head(maximo).iterrows(), 1):
            L.append("| %d | %s | %s | %s | %s | %s %s | %s (%s) |" % (
                n, _num(f["prioridad"]), ETIQUETAS.get(f["tipo"], f["tipo"]), f["aoi"], f["fecha"],
                _num(f["magnitud"]), f["unidad"], f["nivel"], _num(f["confianza"], 2)))
        L.append("")
        L += ["## Alertas priorizadas", ""]
        for n, (_, f) in enumerate(alertas.head(maximo).iterrows(), 1):
            c = f.geometry.representative_point()
            L += [
                "### %d. %s en %s (prioridad %s)" % (
                    n, ETIQUETAS.get(f["tipo"], f["tipo"]), f["aoi"], _num(f["prioridad"])), "",
                "- **Dónde:** %s, %s (lat, lon); identificador %s." % (
                    _num(c.y, 4), _num(c.x, 4), f["id"]),
                "- **Cuándo:** %s." % f["fecha"],
                "- **Magnitud:** %s %s en %d parte(s)." % (_num(f["magnitud"]), f["unidad"],
                                                         int(f["partes"])),
                "- **Confianza:** %s (%s). Fuente: %s." % (f["nivel"], _num(f["confianza"], 2),
                                                           f["fuentes"]),
                "- **Por qué esa confianza:** %s." % (f["evidencia"] or "una sola fuente, sin "
                                                                       "corroboración disponible"),
                "- **Expuesto:** %s%s" % (
                    describir(f) if f.get("exposicion_evaluada") else "no evaluado",
                    (" (" + f["nombres"] + ")") if f.get("nombres") else ""),
            ]
            if f.get("reservas"):
                L.append("- **Reservas de la corrida:** %s." % f["reservas"])
            L += ["- **Verificar:** %s" % f["verificar"],
                  "- **Cálculo:** %s." % f["motivo"], ""]
    if len(contexto):
        L += ["## Contexto", "", "No son emergencias, pero ayudan a interpretar las alertas.", ""]
        for _, f in contexto.head(10).iterrows():
            c = f.geometry.representative_point()
            L.append("- %s en %s (%s, %s): %s %s. %s" % (
                ETIQUETAS.get(f["tipo"], f["tipo"]), f["aoi"], _num(c.y, 3), _num(c.x, 3),
                _num(f["magnitud"]), f["unidad"], f["evidencia"]))
        L.append("")
    L += ["## Lo que no se sabe", ""]
    if len(hallazgos):
        sin_expo = int((~alertas["exposicion_evaluada"].astype(bool)).sum()) if len(alertas) else 0
        no_eval = int(alertas["evidencia"].str.contains("no evaluable").sum()) if len(alertas) else 0
        L += [
            "- %d alerta(s) sin exposición evaluada (su prioridad usa un factor neutro)." % sin_expo,
            "- %d alerta(s) con corroboración no evaluable por falta de datos en la fecha." % no_eval,
        ]
    L += [
        "- Solo se evalúan las zonas y fechas que ya fueron procesadas: la ausencia de una alerta "
        "no significa que no haya ocurrido nada.",
        "- OpenStreetMap puede estar incompleto; \"sin infraestructura registrada\" no es lo mismo "
        "que \"sin infraestructura\".", "",
        "## Método", "",
        "1. Los polígonos de cada módulo se agrupan en eventos y se llevan a un formato común.",
        "2. La confianza parte de un valor por tipo de fuente y se ajusta por corroboración entre "
        "fuentes (cada ajuste aparece en \"Por qué esa confianza\").",
        "3. La exposición cuenta infraestructura civil de OpenStreetMap a menos de 500 m y "
        "localidades a menos de 5 km.",
        "4. Prioridad = 100 x severidad x confianza x (0,4 + 0,6 x exposición).", "",
    ]
    return "\n".join(L)


def evaluar(
    aoi: Optional[Union[AOI, str]] = None,
    base: Optional[Path] = None,
    con_exposicion: bool = True,
    infraestructura: Any = None,
    visor: Any = None,
) -> Dict[str, Any]:
    """Recolecta, corrobora, mide exposición, prioriza y escribe el parte.

    Sin ``aoi`` evalúa todas las corridas y escribe en ``salidas/inteligencia/general``.
    Con ``aoi`` se queda con los hallazgos que la intersectan.
    """
    from global_land_mask import globe

    p = Progreso(5, log)
    p.paso("recolectando hallazgos de todas las corridas")
    h = recolectar(base)
    zona = "general"
    if aoi is not None:
        zona_aoi = cargar_aoi(aoi)
        zona = zona_aoi.nombre
        h = h[h.geometry.intersects(zona_aoi.geometria)].reset_index(drop=True) if len(h) else h
    log.info("  %d hallazgo(s)", len(h))

    p.paso("corroboración entre fuentes")
    h = corroborar(h)
    if len(h) == 0:
        h["nivel"] = []

    p.paso("exposición de infraestructura civil (OpenStreetMap)" if con_exposicion
           else "exposición: omitida")
    if con_exposicion or infraestructura is not None:
        h = calcular_exposicion(h, infraestructura)
    else:
        h = calcular_exposicion(h.iloc[0:0], None) if len(h) == 0 else h.assign(
            exposicion=0.0, exposicion_evaluada=False, nombres="", salud=0, educacion=0,
            bomberos=0, subestaciones=0, rutas_km=0.0, lineas_km=0.0, localidades=0)

    p.paso("priorización")
    h = priorizar(ajustar_por_instalaciones(h))
    indicadores = resumen(h)

    p.paso("parte de situación y archivos")
    if aoi is not None:
        destino_aoi = cargar_aoi(aoi)
    elif len(h):
        o, s, e, n = h.total_bounds
        destino_aoi = aoi_desde_bbox("general", [o - 0.3, s - 0.3, e + 0.3, n + 0.3],
                                     "Todas las zonas con corridas")
    else:
        destino_aoi = aoi_desde_bbox("general", [-76.0, -56.0, -66.4, -17.5],
                                     "Todas las zonas con corridas")
    carpeta = carpeta_salida(MODULO, destino_aoi)
    texto = redactar_parte(h, zona)
    ruta_parte = carpeta / "parte.md"
    ruta_parte.write_text(texto, encoding="utf-8")
    log.info("  parte    -> %s", ruta_parte)
    archivos = [guardar_geojson(h, MODULO, destino_aoi, "hallazgos"), ruta_parte]

    # Figura: ubicación de los hallazgos, tamaño y color según prioridad.
    oeste, sur, este, norte = destino_aoi.bbox
    paso = max(este - oeste, norte - sur) / 500.0
    nx, ny = max(2, int((este - oeste) / paso)), max(2, int((norte - sur) / paso))
    lat = norte - (np.arange(ny) + 0.5) * paso
    lon = oeste + (np.arange(nx) + 0.5) * paso
    tierra = globe.is_land(*np.meshgrid(lat, lon, indexing="ij"))
    fondo = crear_raster(np.where(tierra, 0.32, 0.12).astype("float32"), "EPSG:4326", oeste,
                         norte, paso, "fondo")
    capas = []
    if len(h):
        puntos = h.copy()
        puntos["geometry"] = h.geometry.representative_point()
        for categoria, nivel, color in (("contexto", None, "#9e9e9e"), ("alerta", "baja", "#ffd54f"),
                                        ("alerta", "media", "#ff9100"), ("alerta", "alta", "#ff1744")):
            sub = puntos[(puntos["categoria"] == categoria)
                         & ((puntos["nivel"] == nivel) if nivel else True)]
            if len(sub):
                capas.append((sub, {"color": color, "edgecolor": "black", "linewidth": 0.4,
                                    "markersize": list(12 + sub["prioridad"] * 3), "alpha": 0.9}))
    panel = Panel(np.dstack([fondo.values] * 3),
                  "Hallazgos: tamaño = prioridad; rojo alta, naranja media, amarillo baja confianza; "
                  "gris contexto", referencia=fondo, vectores=capas)
    if visor is not None:
        visor.mostrar(panel)
    titulo = "Evaluación de situación | %s | %d alertas, %d de confianza alta" % (
        zona, indicadores["alertas"], indicadores["confianza_alta"])
    png = guardar_resumen_png(
        [panel], MODULO, destino_aoi, titulo,
        pie="La prioridad ordena qué verificar primero; no confirma daños. Detalle y cálculo en "
            "parte.md. Infraestructura: © Colaboradores de OpenStreetMap.")
    avisos = []
    if len(h) and con_exposicion:
        sin = int((~h[h["categoria"] == "alerta"]["exposicion_evaluada"].astype(bool)).sum())
        if sin:
            avisos.append("%d alerta(s) sin exposición evaluada: su prioridad usa un factor neutro"
                          % sin)
    registrar_corrida(
        MODULO, destino_aoi, titulo, indicadores, archivos + [png], avisos,
        {"zona": zona, "con_exposicion": con_exposicion}, FUENTES, png=png,
        capas_web=[{"archivo": "hallazgos.geojson", "titulo": "Hallazgos evaluados",
                    "color": "#ff1744", "relleno": 0.2, "visible": True}],
    )
    p.fin("evaluación lista")
    return {"hallazgos": h, "parte": ruta_parte, "png": png, "archivos": archivos,
            "indicadores": indicadores, "avisos": avisos}
