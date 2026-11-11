"""Ruta de menor costo entre dos puntos sobre el mapa de costo."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy import ndimage

from ..core import obtener_logger, resolucion

log = obtener_logger("terreno.rutas")


class RutaImposible(RuntimeError):
    """No existe un camino transitable entre origen y destino."""


def _a_celda(costo: xr.DataArray, lon: float, lat: float) -> Tuple[int, int]:
    t = Transformer.from_crs(4326, costo.rio.crs, always_xy=True)
    x, y = t.transform(lon, lat)
    xs, ys = costo["x"].values, costo["y"].values
    res = resolucion(costo)
    if not (xs.min() - res <= x <= xs.max() + res and ys.min() - res <= y <= ys.max() + res):
        raise ValueError("El punto (%.5f, %.5f) queda fuera del mapa de costo" % (lon, lat))
    return int(np.abs(ys - y).argmin()), int(np.abs(xs - x).argmin())


def _ajustar_a_transitable(
    transitable: np.ndarray, celda: Tuple[int, int], res: float, radio_max_m: float, nombre: str
) -> Tuple[int, int]:
    if transitable[celda]:
        return celda
    distancia, indices = ndimage.distance_transform_edt(~transitable, return_indices=True)
    d_m = float(distancia[celda]) * res
    if not np.isfinite(d_m) or d_m > radio_max_m:
        raise RutaImposible(
            "El %s cae en terreno intransitable y no hay celda transitable a menos de %.0f m"
            % (nombre, radio_max_m)
        )
    nueva = (int(indices[0][celda]), int(indices[1][celda]))
    log.warning("el %s cae en celda intransitable; se mueve %.0f m a la más cercana", nombre, d_m)
    return nueva


def ruta_menor_costo(
    costo: xr.DataArray,
    origen: Sequence[float],
    destino: Sequence[float],
    radio_ajuste_m: float = 500.0,
):
    """Ruta óptima entre ``origen`` y ``destino`` (lon, lat en grados).

    Devuelve un GeoDataFrame (EPSG:4326) con una línea y los atributos
    ``tiempo_h``, ``distancia_km`` y ``modo``. Usa ``skimage.graph.MCP_Geometric``
    con 8 vecinos; el costo en s/m se integra como tiempo.
    """
    import geopandas as gpd
    import rioxarray  # noqa: F401
    from shapely.geometry import LineString
    from skimage.graph import MCP_Geometric

    res = resolucion(costo)
    valores = np.asarray(costo.values, dtype="float64")
    transitable = np.isfinite(valores) & (valores > 0)
    if not transitable.any():
        raise RutaImposible("El mapa de costo no tiene celdas transitables")
    ini = _ajustar_a_transitable(transitable, _a_celda(costo, *origen), res, radio_ajuste_m, "origen")
    fin = _ajustar_a_transitable(transitable, _a_celda(costo, *destino), res, radio_ajuste_m, "destino")

    # MCP trata los costos negativos como barreras.
    mcp = MCP_Geometric(np.where(transitable, valores, -1.0), sampling=(res, res))
    acumulado, _ = mcp.find_costs([ini], [fin])
    total_s = float(acumulado[fin])
    if not np.isfinite(total_s):
        raise RutaImposible("No hay conexión transitable entre el origen y el destino")
    celdas = mcp.traceback(fin)
    xs, ys = costo["x"].values, costo["y"].values
    coords = [(float(xs[c]), float(ys[f])) for f, c in celdas]
    if len(coords) == 1:
        coords = coords * 2
    linea = LineString(coords)
    gdf = gpd.GeoDataFrame(
        {
            "modo": [costo.attrs.get("modo", "")],
            "tiempo_h": [round(total_s / 3600.0, 3)],
            "distancia_km": [round(linea.length / 1000.0, 3)],
        },
        geometry=[linea], crs=costo.rio.crs,
    ).to_crs(4326)
    log.info("ruta: %.1f km, %.2f h estimadas (%s)", linea.length / 1000.0, total_s / 3600.0,
             costo.attrs.get("modo", "?"))
    return gdf
