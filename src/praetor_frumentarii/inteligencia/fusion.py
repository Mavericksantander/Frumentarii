"""Corroboración entre fuentes: sube o baja la confianza de cada hallazgo.

Un hallazgo visto por una sola fuente es un indicio. Si otra fuente
*independiente* muestra algo compatible en el mismo lugar y fecha, la confianza
sube; si la otra fuente tenía datos y no vio nada, baja; si no tenía datos para
esa fecha, se dice que no fue evaluable (no cuenta ni a favor ni en contra).

Cada ajuste queda escrito en ``evidencia`` para poder auditar la confianza.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, List, Optional

import numpy as np

CRS_METRICO = "EPSG:32719"   # suficiente para medir cercanía (cientos de metros) en Chile
NIVELES = ((0.7, "alta"), (0.4, "media"), (0.0, "baja"))


def nivel_confianza(valor: float) -> str:
    for umbral, nombre in NIVELES:
        if valor >= umbral:
            return nombre
    return "baja"


def _fecha(texto: Any) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(str(texto)[:10])
    except ValueError:
        return None


def dias_entre(a: Any, b: Any) -> Optional[int]:
    fa, fb = _fecha(a), _fecha(b)
    if fa is None or fb is None:
        return None
    return abs((fa - fb).days)


def _anotar(h, indice: Any, texto: str, delta: float = 0.0) -> None:
    previo = h.at[indice, "evidencia"]
    marca = " (%+.2f)" % delta if delta else ""
    h.at[indice, "evidencia"] = (previo + " | " if previo else "") + texto + marca
    h.at[indice, "confianza"] = float(np.clip(h.at[indice, "confianza"] + delta, 0.05, 0.95))


def corroborar(hallazgos, ventana_dias: int = 10, radio_m: float = 1000.0):
    """Aplica las reglas de corroboración y devuelve una copia actualizada."""
    h = hallazgos.copy()
    if len(h) == 0:
        return h
    h["evidencia"] = h["evidencia"].fillna("").astype(str)
    metrico = h.to_crs(CRS_METRICO).geometry
    es = lambda tipo: h["tipo"] == tipo  # noqa: E731

    # 1. Focos persistentes: fuente fija, no incendio.
    if "persistente" in h:
        for i in h.index[es("foco_calor") & h["persistente"].fillna(False).astype(bool)]:
            dias = int(h.at[i, "dias_distintos"]) if "dias_distintos" in h else 3
            h.at[i, "tipo"], h.at[i, "categoria"] = "fuente_termica_fija", "contexto"
            _anotar(h, i, "arde en el mismo lugar %d días distintos: probable industria, volcán o "
                          "antorcha, no un incendio" % dias, -0.35)

    # 2. Focos no persistentes: más detecciones, más creíble.
    for i in h.index[es("foco_calor")]:
        n = int(h.at[i, "focos"]) if "focos" in h and np.isfinite(h.at[i, "focos"]) else 1
        if n >= 3:
            _anotar(h, i, "%d detecciones térmicas agrupadas" % n, 0.15)
        elif n == 1:
            _anotar(h, i, "una sola detección térmica", -0.10)

    # 3. Área quemada frente a focos de calor (fuentes independientes: óptico y térmico).
    focos = h.index[h["tipo"].isin(["foco_calor", "fuente_termica_fija"])]
    for i in h.index[es("incendio")]:
        en_ventana = [j for j in focos
                      if (dias_entre(h.at[i, "fecha"], h.at[j, "fecha"]) or 10 ** 6) <= ventana_dias]
        if not en_ventana:
            _anotar(h, i, "sin datos de focos FIRMS para esa fecha: corroboración térmica no "
                          "evaluable (requiere consulta histórica)")
            continue
        cerca = [j for j in en_ventana if metrico[i].distance(metrico[j]) <= radio_m
                 and h.at[j, "tipo"] == "foco_calor"]
        if cerca:
            _anotar(h, i, "%d grupo(s) de focos FIRMS a menos de %.0f m en ±%d días"
                    % (len(cerca), radio_m, ventana_dias), 0.30)
            for j in cerca:
                _anotar(h, j, "coincide con área quemada %s (dNBR)" % h.at[i, "id"], 0.30)
        else:
            _anotar(h, i, "había datos de focos FIRMS para esa fecha y ninguno cerca", -0.10)

    # 4. Inundación: apoyo parcial del cambio de retrodispersión y reserva por agua agitada.
    cambios = h.index[es("cambio_superficie")]
    for i in h.index[es("inundacion")]:
        apoyo = [j for j in cambios if metrico[i].intersects(metrico[j])
                 and (dias_entre(h.at[i, "fecha"], h.at[j, "fecha"]) or 0) <= ventana_dias]
        if apoyo:
            _anotar(h, i, "coincide con cambio fuerte de retrodispersión (apoyo parcial: es el "
                          "mismo sensor)", 0.10)
        if "agitada" in str(h.at[i, "reservas"]).lower():
            _anotar(h, i, "agua agitada en la escena posterior: más riesgo de falso positivo", -0.10)

    # 5. Manchas marinas: la recurrencia en el mismo lugar cambia la interpretación.
    manchas = list(h.index[es("mancha_marina")])
    for i in manchas:
        otras = {str(h.at[j, "fecha"])[:10] for j in manchas if j != i
                 and str(h.at[j, "fecha"])[:10] != str(h.at[i, "fecha"])[:10]
                 and metrico[i].distance(metrico[j]) <= 500.0}
        if otras:
            h.at[i, "tipo"] = "mancha_recurrente"
            _anotar(h, i, "se repite en el mismo lugar en %d fecha(s) más: fuente fija (filtración, "
                          "descarga) o falso positivo persistente (sombra de viento, bajo fondo)"
                    % len(otras), 0.15)
        else:
            _anotar(h, i, "una sola fecha: sin otra fuente no se distingue de calma, algas o lluvia")

    # 6. Daño urbano: recordar siempre el límite del método.
    for i in h.index[es("posible_dano_urbano")]:
        _anotar(h, i, "indicio por correlación de amplitud, no coherencia interferométrica: "
                      "obras y cambios de humedad dan lo mismo")

    h["confianza"] = h["confianza"].round(2)
    h["nivel"] = h["confianza"].map(nivel_confianza)
    return h


# Palabras que delatan una instalación capaz de dar una anomalía térmica sin incendio.
PISTAS_INSTALACION = ("pfv", "fotovolt", "solar", "termosolar", "fundici", "refiner", "termoel",
                      "central ", "planta ", "siderurg", "cement", "celulosa", "geotér")


def ajustar_por_instalaciones(hallazgos):
    """Reclasifica focos de calor que coinciden con una instalación identificada.

    Se aplica después de calcular la exposición. Ajuste tras la validación con
    datos reales (noviembre de 2026): el foco más intenso del país (887 MW)
    caía sobre la planta fotovoltaica Campos del Sol. Los paneles y las plantas
    termosolares reflejan el sol hacia el sensor y VIIRS los marca como focos;
    fundiciones y centrales térmicas emiten calor real pero no son incendios.
    """
    h = hallazgos.copy()
    if len(h) == 0 or "nombres" not in h:
        return h
    for i in h.index[h["tipo"] == "foco_calor"]:
        nombres = str(h.at[i, "nombres"])
        pista = next((p for p in PISTAS_INSTALACION if p in nombres.lower()), None)
        hay_planta = float(h.at[i, "subestaciones"] or 0) > 0 if "subestaciones" in h else False
        if pista and hay_planta:
            h.at[i, "tipo"], h.at[i, "categoria"] = "fuente_termica_fija", "contexto"
            _anotar(h, i, "coincide con una instalación registrada en OpenStreetMap (%s): probable "
                          "reflejo solar o calor industrial, no un incendio" % nombres, -0.30)
    h["confianza"] = h["confianza"].round(2)
    h["nivel"] = h["confianza"].map(nivel_confianza)
    return h
