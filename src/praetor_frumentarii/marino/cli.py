"""CLI marino: ``frumentarii marino manchas``."""
from __future__ import annotations

import argparse
from typing import List, Optional

from .. import cli_comun as comun


def _correr(args: argparse.Namespace) -> None:
    from .flujo import ejecutar

    comun.saludo("marino")
    visor = comun.crear_visor(args)
    try:
        r = ejecutar(args.aoi, args.fechas, max_escenas=args.escenas, contraste_db=args.contraste,
                     area_min_ha=args.area_min, costa=args.costa, margen_m=args.margen,
                     viento_ms=args.viento, archivo_era5=args.era5, visor=visor)
    finally:
        comun.cerrar_visor(visor)
    comun.fin(r)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frumentarii marino",
        description="Manchas oscuras en el mar con Sentinel-1: indicios de derrames o filtraciones.",
    )
    sub = parser.add_subparsers(dest="comando")
    s = sub.add_parser("manchas", help="detección por escena y recurrencia multifecha")
    comun.agregar_comunes(s, "magallanes")
    s.add_argument("--fechas", required=True, metavar="FECHAS", help="'AAAA-MM-DD/AAAA-MM-DD'")
    s.add_argument("--escenas", type=int, default=4, help="máximo de fechas a procesar (4)")
    s.add_argument("--contraste", type=float, default=4.0,
                   help="dB bajo el fondo local para marcar una mancha (4)")
    s.add_argument("--area-min", type=float, default=10.0, help="área mínima en hectáreas (10)")
    s.add_argument("--costa", default="global-land-mask",
                   help="'global-land-mask' o ruta a un shapefile/GeoPackage de tierra")
    s.add_argument("--margen", type=float, default=500.0, help="margen costero en metros (500)")
    s.add_argument("--viento", type=float, default=None, help="viento conocido en m/s")
    s.add_argument("--era5", default=None, help="NetCDF ERA5 propio con u10 y v10")
    s.set_defaults(funcion=_correr)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return comun.ejecutar_parser(crear_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
