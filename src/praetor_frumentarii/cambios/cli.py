"""CLI de cambios: ``frumentarii cambios quemas|inundacion|sar``."""
from __future__ import annotations

import argparse
from typing import List, Optional

from .. import cli_comun as comun


def _correr(args: argparse.Namespace) -> None:
    from . import flujo

    comun.saludo("cambios")
    visor = comun.crear_visor(args)
    try:
        if args.comando == "quemas":
            r = flujo.ejecutar_quemas(args.aoi, args.antes, args.despues, umbral=args.umbral,
                                      nubes_max=args.nubes, visor=visor)
        elif args.comando == "inundacion":
            r = flujo.ejecutar_inundacion(args.aoi, args.antes, args.despues, metodo=args.metodo,
                                          umbral_db=args.umbral_db, nubes_max=args.nubes,
                                          visor=visor)
        else:
            r = flujo.ejecutar_sar(args.aoi, args.antes, args.despues, umbral_db=args.umbral_db,
                                   visor=visor)
    finally:
        comun.cerrar_visor(visor)
    comun.fin(r)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frumentarii cambios",
        description="Detección de cambios para desastres comparando un período antes y otro después.",
    )
    sub = parser.add_subparsers(dest="comando")
    definiciones = (
        ("quemas", "área quemada con dNBR de Sentinel-2", "valparaiso_vina"),
        ("inundacion", "agua nueva con umbral SAR y/o NDWI", "biobio"),
        ("sar", "cambio de retrodispersión y proxy de coherencia Sentinel-1", "biobio"),
    )
    for nombre, ayuda, aoi in definiciones:
        s = sub.add_parser(nombre, help=ayuda)
        comun.agregar_comunes(s, aoi)
        s.add_argument("--antes", required=True, metavar="FECHAS",
                       help="período previo 'AAAA-MM-DD/AAAA-MM-DD'")
        s.add_argument("--despues", required=True, metavar="FECHAS",
                       help="período posterior 'AAAA-MM-DD/AAAA-MM-DD'")
        if nombre == "quemas":
            s.add_argument("--umbral", type=float, default=0.27, help="dNBR mínimo (0.27)")
        if nombre == "inundacion":
            s.add_argument("--metodo", choices=("sar", "ndwi", "ambos"), default="sar")
            s.add_argument("--umbral-db", type=float, default=None,
                           help="umbral VV en dB; por defecto se calcula con Otsu")
        if nombre == "sar":
            s.add_argument("--umbral-db", type=float, default=3.0, help="cambio mínimo en dB (3)")
        if nombre != "sar":
            s.add_argument("--nubes", type=float, default=40.0, help="nubosidad máxima en %%")
        s.set_defaults(funcion=_correr)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return comun.ejecutar_parser(crear_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
