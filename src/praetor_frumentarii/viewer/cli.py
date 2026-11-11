"""CLI del visor: ``frumentarii viewer tablero|servir``."""
from __future__ import annotations

import argparse
from typing import List, Optional

from .. import cli_comun as comun


def _tablero(args: argparse.Namespace) -> None:
    from .web import generar_tablero

    comun.saludo("viewer")
    ruta = generar_tablero()
    if args.abrir:
        import webbrowser

        webbrowser.open(ruta.as_uri())


def _servir(args: argparse.Namespace) -> None:
    from .web import servir

    comun.saludo("viewer")
    servir(puerto=args.puerto, abrir=not args.sin_abrir)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frumentarii viewer",
        description="Tablero web local con las capas y los indicadores de todas las corridas.",
    )
    sub = parser.add_subparsers(dest="comando")
    t = sub.add_parser("tablero", help="genera salidas/tablero.html")
    t.add_argument("--abrir", action="store_true", help="abre el tablero en el navegador")
    t.set_defaults(funcion=_tablero)
    s = sub.add_parser("servir", help="sirve el tablero en http://127.0.0.1:<puerto>")
    s.add_argument("--puerto", type=int, default=8765)
    s.add_argument("--sin-abrir", action="store_true", help="no abrir el navegador")
    s.set_defaults(funcion=_servir)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return comun.ejecutar_parser(crear_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
