"""Logging claro de progreso para todos los módulos."""
from __future__ import annotations

import logging
import sys
import time
from typing import Optional

_RAIZ = "frumentarii"
_configurado = False


def obtener_logger(nombre: str = "") -> logging.Logger:
    """Devuelve un logger hijo de ``frumentarii`` que escribe a la terminal."""
    global _configurado
    raiz = logging.getLogger(_RAIZ)
    if not _configurado:
        manejador = logging.StreamHandler(sys.stderr)
        manejador.setFormatter(
            logging.Formatter("%(asctime)s | %(name)-22s | %(message)s", "%H:%M:%S")
        )
        raiz.addHandler(manejador)
        raiz.setLevel(logging.INFO)
        raiz.propagate = False
        _configurado = True
    return raiz.getChild(nombre) if nombre else raiz


def fijar_nivel(nivel: str) -> None:
    obtener_logger().setLevel(getattr(logging, nivel.upper(), logging.INFO))


class Progreso:
    """Contador de pasos: ``[2/5] cargando escena (paso anterior: 3.1 s)``."""

    def __init__(self, total: int, logger: Optional[logging.Logger] = None) -> None:
        self.total = int(total)
        self.actual = 0
        self.log = logger or obtener_logger()
        self._t0 = time.time()
        self._t_paso = self._t0

    def paso(self, mensaje: str) -> None:
        ahora = time.time()
        self.actual += 1
        previo = "" if self.actual == 1 else "  (paso anterior: %.1f s)" % (ahora - self._t_paso)
        self._t_paso = ahora
        self.log.info("[%d/%d] %s%s", self.actual, self.total, mensaje, previo)

    def fin(self, mensaje: str = "listo") -> None:
        self.log.info("%s en %.1f s", mensaje, time.time() - self._t0)
