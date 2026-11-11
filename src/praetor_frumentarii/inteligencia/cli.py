"""CLI de inteligencia: ``frumentarii inteligencia evaluar``."""
from __future__ import annotations

import argparse
from typing import List, Optional

from .. import cli_comun as comun


def _evaluar(args: argparse.Namespace) -> None:
    from .parte import evaluar

    comun.saludo("inteligencia")
    visor = comun.crear_visor(args)
    try:
        r = evaluar(args.aoi, con_exposicion=not args.sin_exposicion, visor=visor)
    finally:
        comun.cerrar_visor(visor)
    comun.fin(r)
    comun.log.info("Parte de situación: %s", r["parte"])


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frumentarii inteligencia",
        description="Fusiona los hallazgos de todas las corridas, mide la exposición de "
                    "infraestructura civil, los prioriza y escribe un parte de situación.",
    )
    sub = parser.add_subparsers(dest="comando")
    s = sub.add_parser("evaluar", help="evaluación priorizada y parte de situación")
    s.add_argument("--aoi", default=None,
                   help="limitar a una AOI (nombre, GeoJSON o bbox); por defecto, todas las corridas")
    s.add_argument("--sin-exposicion", action="store_true",
                   help="no consultar OpenStreetMap (más rápido, funciona sin red)")
    s.add_argument("--en-vivo", action="store_true", help="mostrar el mapa en una ventana")
    s.add_argument("--pausa", type=float, default=10.0, help="segundos de la ventana en vivo")
    s.set_defaults(funcion=_evaluar)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return comun.ejecutar_parser(crear_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
