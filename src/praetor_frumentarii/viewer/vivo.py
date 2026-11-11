"""Observación en vivo: una ventana matplotlib que muestra cada escena procesada."""
from __future__ import annotations

import time
from typing import Any, Optional

from .. import NOMBRE
from ..core import obtener_logger
from ..core.salidas import Panel, dibujar_panel

log = obtener_logger("viewer.vivo")


class VisorVivo:
    """Muestra cada ``Panel`` durante ``pausa_s`` segundos (10 por defecto).

    Los módulos reciben el visor por el argumento ``visor=`` y llaman a
    ``visor.mostrar(panel)`` después de procesar cada escena. Si no hay
    pantalla disponible (backend Agg) solo se registra el avance.
    """

    def __init__(self, pausa_s: float = 10.0, titulo: str = NOMBRE) -> None:
        self.pausa_s = max(0.0, float(pausa_s))
        self.titulo = titulo
        self.cuadros = 0
        self._fig: Optional[Any] = None
        self._plt: Optional[Any] = None

    def _abrir(self) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.ion()
        self._fig = plt.figure(figsize=(9, 8))
        try:
            self._fig.canvas.manager.set_window_title("%s | observación en vivo" % self.titulo)
        except Exception:
            pass
        log.info("ventana en vivo abierta: cada escena se muestra %.0f s", self.pausa_s)

    def mostrar(self, panel: Panel) -> None:
        if self._fig is None:
            self._abrir()
        assert self._fig is not None and self._plt is not None
        self.cuadros += 1
        self._fig.clf()
        dibujar_panel(self._fig.add_subplot(1, 1, 1), panel)
        self._fig.suptitle(
            "%s | escena %d | %s" % (self.titulo, self.cuadros, time.strftime("%H:%M:%S")),
            fontsize=10,
        )
        log.info("  en vivo [%d]: %s (%.0f s)", self.cuadros, panel.titulo, self.pausa_s)
        self._fig.canvas.draw_idle()
        if self.pausa_s > 0:
            # plt.pause mantiene la ventana viva y atiende sus eventos.
            self._plt.pause(self.pausa_s)
        else:
            self._fig.canvas.flush_events()

    def cerrar(self) -> None:
        if self._fig is not None and self._plt is not None:
            self._plt.close(self._fig)
            self._fig = None
