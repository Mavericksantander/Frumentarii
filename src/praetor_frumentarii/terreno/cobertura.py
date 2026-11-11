"""Cobertura de suelo ESA WorldCover."""
from __future__ import annotations

from typing import Dict, Tuple, Union

import xarray as xr

from ..core import AOI, cargar_worldcover

# valor -> (color oficial, etiqueta en español)
CLASES_WORLDCOVER: Dict[int, Tuple[str, str]] = {
    10: ("#006400", "Bosque"),
    20: ("#ffbb22", "Matorral"),
    30: ("#ffff4c", "Pradera"),
    40: ("#f096ff", "Cultivo"),
    50: ("#fa0000", "Construido"),
    60: ("#b4b4b4", "Suelo desnudo"),
    70: ("#f0f0f0", "Nieve y hielo"),
    80: ("#0064c8", "Agua"),
    90: ("#0096a0", "Humedal"),
    95: ("#00cf75", "Manglar"),
    100: ("#fae6a0", "Musgo y liquen"),
}
AGUA = 80


def cobertura(aoi: Union[AOI, str], como: xr.DataArray, anio: int = 2021) -> xr.DataArray:
    """Clases WorldCover en la grilla de ``como`` (por ejemplo, la del DEM)."""
    return cargar_worldcover(aoi, como, anio)
