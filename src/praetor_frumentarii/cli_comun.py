"""Piezas compartidas por las líneas de comando de cada módulo."""
from __future__ import annotations

import argparse
from typing import Any, List, Optional, Tuple

from . import NOMBRE
from .core import listar_aois, obtener_logger

log = obtener_logger("cli")


def punto(texto: str) -> Tuple[float, float]:
    """Convierte 'lon,lat' en una tupla."""
    try:
        lon, lat = [float(v) for v in texto.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("se espera 'lon,lat', por ejemplo -73.76,-42.48")
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise argparse.ArgumentTypeError("coordenadas fuera de rango (¿lat y lon invertidas?)")
    return (lon, lat)


def agregar_comunes(parser: argparse.ArgumentParser, aoi_por_defecto: str) -> None:
    parser.add_argument(
        "--aoi", default=aoi_por_defecto,
        help="nombre de AOI de ejemplo (%s), ruta a un GeoJSON o bbox 'O,S,E,N' "
             "(por defecto: %s)" % (", ".join(listar_aois()), aoi_por_defecto),
    )
    parser.add_argument("--en-vivo", action="store_true",
                        help="abre una ventana que muestra cada escena a medida que se procesa")
    parser.add_argument("--pausa", type=float, default=10.0,
                        help="segundos que se muestra cada escena con --en-vivo (por defecto 10)")


def crear_visor(args: argparse.Namespace) -> Optional[Any]:
    if not getattr(args, "en_vivo", False):
        return None
    from .viewer import VisorVivo

    return VisorVivo(pausa_s=args.pausa)


def cerrar_visor(visor: Optional[Any]) -> None:
    if visor is not None:
        visor.cerrar()


def saludo(modulo: str) -> None:
    log.info("%s | módulo %s | uso exclusivamente civil", NOMBRE, modulo)


def fin(resultado: dict) -> None:
    for aviso in resultado.get("avisos", []):
        log.warning("AVISO: %s", aviso)
    log.info("PNG resumen: %s", resultado.get("png"))


def unir_negativos(argv: List[str]) -> List[str]:
    """Permite ``--origen -73.7,-42.4``: argparse confunde el valor con una opción.

    Une ``--opcion`` y un valor que empieza con ``-`` y dígito en ``--opcion=valor``.
    """
    salida: List[str] = []
    i = 0
    while i < len(argv):
        actual = argv[i]
        siguiente = argv[i + 1] if i + 1 < len(argv) else ""
        if (actual.startswith("--") and "=" not in actual and len(siguiente) > 1
                and siguiente[0] == "-" and (siguiente[1].isdigit() or siguiente[1] == ".")):
            salida.append("%s=%s" % (actual, siguiente))
            i += 2
        else:
            salida.append(actual)
            i += 1
    return salida


def ejecutar_parser(parser: argparse.ArgumentParser, argv: Optional[List[str]]) -> int:
    import sys

    args = parser.parse_args(unir_negativos(list(sys.argv[1:] if argv is None else argv)))
    if not hasattr(args, "funcion"):
        parser.print_help()
        return 2
    try:
        args.funcion(args)
    except (LookupError, FileNotFoundError, ValueError) as e:
        log.error("ERROR: %s", e)
        return 1
    return 0
