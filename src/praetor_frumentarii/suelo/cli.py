"""CLI de suelo: ``frumentarii suelo alteracion``."""
from __future__ import annotations

import argparse
from typing import List, Optional

from .. import cli_comun as comun


def _correr(args: argparse.Namespace) -> None:
    from .flujo import ejecutar

    comun.saludo("suelo")
    visor = comun.crear_visor(args)
    try:
        r = ejecutar(args.aoi, args.fechas, sensor=args.sensor, nubes_max=args.nubes,
                     percentil=args.percentil, area_min_ha=args.area_min, visor=visor)
    finally:
        comun.cerrar_visor(visor)
    comun.fin(r)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frumentarii suelo",
        description="Índices de alteración mineral (óxidos de hierro, arcillas, Crosta/PCA).",
    )
    sub = parser.add_subparsers(dest="comando")
    s = sub.add_parser("alteracion", help="cocientes de bandas, Crosta y zonas anómalas")
    comun.agregar_comunes(s, "atacama")
    s.add_argument("--sensor", choices=("s2", "landsat", "aster"), default="s2")
    s.add_argument("--fechas", default=None, metavar="FECHAS",
                   help="'AAAA-MM-DD/AAAA-MM-DD' (por defecto: verano 2024; ASTER: 2000 a 2008)")
    s.add_argument("--nubes", type=float, default=10.0, help="nubosidad máxima en %%")
    s.add_argument("--percentil", type=float, default=97.0,
                   help="percentil sobre el cual una zona se marca como anómala (97)")
    s.add_argument("--area-min", type=float, default=2.0, help="área mínima en hectáreas (2)")
    s.set_defaults(funcion=_correr)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return comun.ejecutar_parser(crear_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
