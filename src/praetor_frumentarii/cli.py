"""Punto de entrada: ``frumentarii <modulo> <comando> [opciones]``."""
from __future__ import annotations

import importlib
import sys
from typing import List, Optional

from . import NOMBRE, __version__

MODULOS = {
    "terreno": "transitabilidad para rescate y logística: costo de desplazamiento y rutas",
    "cambios": "cambios por desastres: inundaciones, áreas quemadas, daño tras terremotos",
    "marino": "manchas oscuras en el mar con Sentinel-1 (derrames y filtraciones naturales)",
    "suelo": "índices de alteración mineral con Sentinel-2, Landsat y ASTER",
    "viewer": "tablero web local y visor de resultados",
}


def _ayuda() -> str:
    lineas = [
        "%s %s" % (NOMBRE, __version__),
        "Análisis geoespacial con imágenes satelitales abiertas, para usos civiles.",
        "",
        "Uso: frumentarii <modulo> <comando> [opciones]",
        "",
        "Módulos:",
    ]
    lineas += ["  %-8s %s" % (k, v) for k, v in MODULOS.items()]
    lineas += ["", "Ayuda de un módulo: frumentarii <modulo> --help",
               "AOIs de ejemplo:    frumentarii aois"]
    return "\n".join(lineas)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_ayuda())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    if argv[0] == "aois":
        from .core import cargar_aoi, listar_aois

        for nombre in listar_aois():
            a = cargar_aoi(nombre)
            print("%-16s %s  [aproximada, %.0f km²]" % (nombre, a.descripcion, a.area_km2))
        return 0
    if argv[0] not in MODULOS:
        print("Módulo desconocido: %s\n\n%s" % (argv[0], _ayuda()), file=sys.stderr)
        return 2
    modulo = importlib.import_module("praetor_frumentarii.%s.cli" % argv[0])
    return int(modulo.main(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
