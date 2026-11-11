"""Inteligencia: fusiona los hallazgos de los módulos en una evaluación priorizada.

Uso exclusivamente civil. Evalúa fenómenos (fuego, agua, cambios del terreno,
manchas en el mar) y la infraestructura civil expuesta; no identifica ni sigue
personas, vehículos ni instalaciones militares.
"""
from __future__ import annotations

from .exposicion import calcular_exposicion, clasificar, describir, exposicion_de, indice_exposicion
from .fusion import ajustar_por_instalaciones, corroborar, dias_entre, nivel_confianza
from .hallazgos import ETIQUETAS, REGLAS, Regla, agrupar_en_eventos, recolectar, severidad_desde
from .parte import evaluar, redactar_parte
from .prioridad import VERIFICAR, priorizar, resumen

__all__ = [
    "calcular_exposicion", "clasificar", "describir", "exposicion_de", "indice_exposicion",
    "ajustar_por_instalaciones", "corroborar", "dias_entre", "nivel_confianza", "ETIQUETAS", "REGLAS", "Regla",
    "agrupar_en_eventos", "recolectar", "severidad_desde", "evaluar", "redactar_parte",
    "VERIFICAR", "priorizar", "resumen",
]
