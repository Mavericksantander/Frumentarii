"""Viewer: observación en vivo (matplotlib) y tablero web local (folium)."""
from __future__ import annotations

from .servidor import construir_comando, crear_servidor
from .vivo import VisorVivo
from .web import generar_tablero, leer_corridas, mapa_web, raster_a_png, servir

__all__ = ["construir_comando", "crear_servidor", "VisorVivo", "generar_tablero", "leer_corridas", "mapa_web", "raster_a_png", "servir"]
