"""Marino: manchas oscuras en el mar con Sentinel-1 (derrames y filtraciones naturales)."""
from __future__ import annotations

from .flujo import ejecutar
from .manchas import describir_manchas, fondo_local, manchas_oscuras
from .recurrencia import recurrencia
from .tierra import mascara_tierra
from .viento import apto_por_viento, estado_mar_por_sar, viento_desde_archivo, viento_era5

__all__ = [
    "ejecutar", "describir_manchas", "fondo_local", "manchas_oscuras", "recurrencia",
    "mascara_tierra", "apto_por_viento", "estado_mar_por_sar", "viento_desde_archivo",
    "viento_era5",
]
