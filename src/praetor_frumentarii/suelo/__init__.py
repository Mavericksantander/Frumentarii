"""Suelo: índices de alteración mineral con Sentinel-2, Landsat y ASTER."""
from __future__ import annotations

from .crosta import CONFIGURACION, crosta, elegir_componente
from .flujo import ejecutar
from .indices import COCIENTES_ASTER, COCIENTES_OPTICOS, anomalias, cociente, indices_alteracion
from .mascaras import mascara_vegetacion_agua

__all__ = [
    "CONFIGURACION", "crosta", "elegir_componente", "ejecutar", "COCIENTES_ASTER",
    "COCIENTES_OPTICOS", "anomalias", "cociente", "indices_alteracion",
    "mascara_vegetacion_agua",
]
