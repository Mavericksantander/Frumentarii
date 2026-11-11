"""Priorización: ordena los hallazgos para decidir qué verificar primero.

prioridad = 100 x severidad x confianza x (0,4 + 0,6 x exposición)

Es una ayuda para ordenar la revisión, no una predicción ni una medida de
daño. Los tres factores quedan a la vista en ``motivo`` para poder discutirlos.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

# Qué convendría comprobar en terreno o con otra fuente, según el tipo.
VERIFICAR: Dict[str, str] = {
    "incendio": "Confirmar perímetro y estado (activo o extinguido) con CONAF o brigadas; revisar "
                "viviendas e infraestructura dentro del perímetro.",
    "foco_calor": "Confirmar si es incendio, quema agrícola o industria con observación directa o "
                  "la siguiente pasada del satélite.",
    "fuente_termica_fija": "Identificar la instalación o volcán; descartar como incendio.",
    "inundacion": "Confirmar anegamiento y estado de caminos y puentes; comparar con registros "
                  "de caudal (DGA).",
    "posible_dano_urbano": "Inspección visual o imagen de alta resolución; descartar obras y "
                           "demoliciones programadas.",
    "cambio_superficie": "Revisar si corresponde a remoción en masa, cosecha, faenas o humedad.",
    "mancha_marina": "Buscar otra fuente (imagen óptica, reporte de navegación, autoridad "
                     "marítima) antes de tratarla como derrame.",
    "mancha_recurrente": "Revisar batimetría, descargas costeras y viento habitual; muestreo de "
                         "agua si persiste.",
    "alteracion_mineral": "Contrastar con cartografía geológica y faenas existentes; muestreo en "
                          "terreno.",
}
FACTOR_SIN_EXPOSICION = 0.7     # neutro cuando no se pudo evaluar la exposición
FACTOR_CONTEXTO = 0.3           # los hallazgos de contexto no compiten con las alertas


def priorizar(hallazgos):
    """Agrega ``prioridad`` (0 a 100), ``motivo`` y ``verificar``; ordena de mayor a menor."""
    h = hallazgos.copy()
    if len(h) == 0:
        for columna in ("prioridad", "motivo", "verificar"):
            h[columna] = []
        return h
    prioridades, motivos = [], []
    for _, f in h.iterrows():
        evaluada = bool(f.get("exposicion_evaluada", False))
        exposicion = float(f.get("exposicion", 0.0) or 0.0)
        factor = (0.4 + 0.6 * exposicion) if evaluada else FACTOR_SIN_EXPOSICION
        valor = 100.0 * float(f["severidad"]) * float(f["confianza"]) * factor
        if f["categoria"] != "alerta":
            valor *= FACTOR_CONTEXTO
        prioridades.append(round(float(np.clip(valor, 0, 100)), 1))
        motivos.append("severidad %.2f x confianza %.2f x exposición %s%s" % (
            f["severidad"], f["confianza"],
            "%.2f" % exposicion if evaluada else "no evaluada (factor neutro 0,7)",
            "" if f["categoria"] == "alerta" else " x 0,3 por ser contexto"))
    h["prioridad"] = prioridades
    h["motivo"] = motivos
    h["verificar"] = h["tipo"].map(VERIFICAR).fillna("")
    return h.sort_values("prioridad", ascending=False).reset_index(drop=True)


def resumen(hallazgos) -> Dict[str, Any]:
    """Indicadores globales para el tablero."""
    if len(hallazgos) == 0:
        return {"hallazgos": 0, "alertas": 0, "confianza_alta": 0, "prioridad_maxima": 0.0}
    alertas = hallazgos[hallazgos["categoria"] == "alerta"]
    return {
        "hallazgos": int(len(hallazgos)),
        "alertas": int(len(alertas)),
        "confianza_alta": int((alertas["nivel"] == "alta").sum()) if "nivel" in alertas else 0,
        "confianza_media": int((alertas["nivel"] == "media").sum()) if "nivel" in alertas else 0,
        "prioridad_maxima": float(alertas["prioridad"].max()) if len(alertas) else 0.0,
        "con_exposicion": int(alertas["exposicion_evaluada"].sum())
        if "exposicion_evaluada" in alertas else 0,
    }
