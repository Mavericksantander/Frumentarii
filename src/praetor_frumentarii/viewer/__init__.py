"""Viewer: observación en vivo (matplotlib) y tablero web local (folium)."""
from __future__ import annotations

from .vivo import VisorVivo
from .web import generar_tablero, leer_corridas, mapa_web, raster_a_png, servir

__all__ = ["VisorVivo", "generar_tablero", "leer_corridas", "mapa_web", "raster_a_png", "servir"]
