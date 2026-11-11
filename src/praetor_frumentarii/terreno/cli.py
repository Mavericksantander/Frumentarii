"""CLI de terreno: ``frumentarii terreno costo|ruta``."""
from __future__ import annotations

import argparse
from typing import List, Optional

from .. import cli_comun as comun
from .costo import MODOS


def _correr(args: argparse.Namespace) -> None:
    from .flujo import ejecutar

    comun.saludo("terreno")
    visor = comun.crear_visor(args)
    try:
        resultado = ejecutar(
            args.aoi, modo=args.modo, origen=getattr(args, "origen", None),
            destino=getattr(args, "destino", None), fechas_humedad=args.humedad,
            usar_vias=not args.sin_vias, resolucion_m=args.resolucion, visor=visor,
        )
    finally:
        comun.cerrar_visor(visor)
    comun.fin(resultado)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frumentarii terreno",
        description="Transitabilidad para rescate y logística: mapa de costo y rutas.",
    )
    sub = parser.add_subparsers(dest="comando")
    for nombre, ayuda in (("costo", "mapa de costo de desplazamiento"),
                          ("ruta", "mapa de costo y ruta de menor costo entre dos puntos")):
        s = sub.add_parser(nombre, help=ayuda)
        comun.agregar_comunes(s, "chiloe")
        s.add_argument("--modo", choices=sorted(MODOS), default="pie")
        s.add_argument("--humedad", metavar="FECHAS",
                       help="rango 'AAAA-MM-DD/AAAA-MM-DD' para el proxy de humedad Sentinel-1")
        s.add_argument("--sin-vias", action="store_true", help="no usar la red vial de OSM")
        s.add_argument("--resolucion", type=float, default=30.0, help="metros por píxel")
        if nombre == "ruta":
            s.add_argument("--origen", type=comun.punto, required=True, metavar="LON,LAT")
            s.add_argument("--destino", type=comun.punto, required=True, metavar="LON,LAT")
        s.set_defaults(funcion=_correr)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return comun.ejecutar_parser(crear_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
