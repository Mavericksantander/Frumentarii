"""Mapa de costo de desplazamiento para rescate y logística civil.

El costo se expresa en segundos por metro (el inverso de la velocidad
estimada), así la suma a lo largo de una ruta es directamente un tiempo de
viaje aproximado. Las celdas intransitables valen infinito.

Los parámetros son órdenes de magnitud razonables, no una calibración: se
pueden ajustar creando otro ``ModoTransito``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import xarray as xr

from ..core import como
from .vias import CAMINO, HUELLA, PRINCIPAL, SENDERO

INF = float("inf")


@dataclass(frozen=True)
class ModoTransito:
    nombre: str
    velocidad_kmh: float                 # fuera de camino, en plano y suelo firme
    pendiente_max: float                 # grados; más que esto es intransitable
    pendiente_max_via: float             # límite sobre una vía existente
    factor_cobertura: Dict[int, float] = field(default_factory=dict)  # divisor de velocidad
    velocidad_via_kmh: Dict[int, float] = field(default_factory=dict)  # por clase de vía
    sensibilidad_humedad: float = 0.5    # cuánto frena el suelo húmedo fuera de camino
    rugosidad_ref: float = 10.0          # m de desviación que duplican el costo


MODOS: Dict[str, ModoTransito] = {
    "pie": ModoTransito(
        "pie", 5.0, 40.0, 45.0,
        {10: 1.8, 20: 1.4, 30: 1.1, 40: 1.2, 50: 1.0, 60: 1.0, 70: 2.5, 80: INF, 90: 3.0,
         95: INF, 100: 1.3},
        {SENDERO: 5.0, HUELLA: 5.0, CAMINO: 5.0, PRINCIPAL: 5.0},
        0.5, 12.0,
    ),
    "4x4": ModoTransito(
        "4x4", 15.0, 22.0, 30.0,
        {10: 6.0, 20: 2.5, 30: 1.5, 40: 1.8, 50: 1.0, 60: 1.2, 70: INF, 80: INF, 90: INF,
         95: INF, 100: 2.0},
        {HUELLA: 20.0, CAMINO: 40.0, PRINCIPAL: 70.0},
        1.5, 6.0,
    ),
    "camion": ModoTransito(
        "camion", 8.0, 10.0, 15.0,
        {10: INF, 20: 8.0, 30: 4.0, 40: 5.0, 50: 1.2, 60: 3.0, 70: INF, 80: INF, 90: INF,
         95: INF, 100: INF},
        {HUELLA: 10.0, CAMINO: 30.0, PRINCIPAL: 60.0},
        3.0, 4.0,
    ),
}


def _factor_pendiente(grados: np.ndarray, modo: ModoTransito) -> np.ndarray:
    """Fracción de la velocidad en plano que se conserva (0 a 1)."""
    if modo.nombre == "pie":
        # Función de senderismo de Tobler, normalizada a 1 en terreno plano.
        return np.exp(-3.5 * np.abs(np.tan(np.radians(grados)) + 0.05)) / np.exp(-3.5 * 0.05)
    return np.clip(1.0 - (grados / modo.pendiente_max) ** 2 * 0.8, 0.05, 1.0)


def mapa_costo(
    pendiente: xr.DataArray,
    cobertura: Optional[xr.DataArray] = None,
    modo: str = "pie",
    rugosidad: Optional[xr.DataArray] = None,
    humedad: Optional[xr.DataArray] = None,
    vias: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Costo de desplazamiento en s/m para ``modo`` ('pie', '4x4' o 'camion').

    Todas las capas deben compartir grilla. ``cobertura`` usa clases WorldCover,
    ``humedad`` es el índice 0 a 1 y ``vias`` el raster de clases de vía.
    Sobre una vía transitable para el modo se usa la velocidad de la vía y se
    ignoran cobertura y humedad (un puente cruza el agua).
    """
    if modo not in MODOS:
        raise ValueError("modo debe ser uno de %s" % sorted(MODOS))
    m = MODOS[modo]
    s = np.asarray(pendiente.values, dtype="float64")
    f_pend = _factor_pendiente(s, m)
    velocidad = m.velocidad_kmh * f_pend

    if cobertura is not None:
        clases = np.asarray(cobertura.values)
        divisor = np.ones(s.shape, dtype="float64")
        for valor, factor in m.factor_cobertura.items():
            divisor[clases == valor] = factor
        velocidad = velocidad / divisor
    if rugosidad is not None:
        r = np.nan_to_num(np.asarray(rugosidad.values, dtype="float64"), nan=0.0)
        velocidad = velocidad / (1.0 + r / m.rugosidad_ref)
    if humedad is not None:
        h = np.nan_to_num(np.asarray(humedad.values, dtype="float64"), nan=0.0)
        velocidad = velocidad / (1.0 + m.sensibilidad_humedad * h)
    velocidad = np.where(s > m.pendiente_max, 0.0, velocidad)

    if vias is not None:
        clase_via = np.asarray(vias.values)
        for clase, v_via in m.velocidad_via_kmh.items():
            en_via = (clase_via == clase) & (s <= m.pendiente_max_via)
            # En vía la pendiente frena la mitad que fuera de ella.
            v = v_via * (0.5 + 0.5 * f_pend)
            velocidad = np.where(en_via, np.maximum(v, velocidad), velocidad)

    velocidad = np.where(np.isfinite(s), velocidad, 0.0)
    with np.errstate(divide="ignore"):
        costo = np.where(velocidad > 0, 3.6 / velocidad, INF)  # km/h -> s/m
    salida = como(costo.astype("float32"), pendiente, "costo")
    salida.attrs.update(unidades="s/m", modo=modo)
    return salida


def fraccion_transitable(costo: xr.DataArray) -> float:
    return float(np.isfinite(costo.values).mean())
