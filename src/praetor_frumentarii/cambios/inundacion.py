"""Inundación: agua nueva entre dos fechas con NDWI (óptico) y umbral SAR."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr
from scipy import ndimage

from ..core import a_2d, como, obtener_logger

log = obtener_logger("cambios.inundacion")

# Rango físico razonable para el umbral agua/tierra en VV (dB).
RANGO_UMBRAL_DB: Tuple[float, float] = (-24.0, -10.5)
UMBRAL_DB_POR_DEFECTO = -18.0


def ndwi(ds: xr.Dataset) -> xr.DataArray:
    """NDWI de McFeeters = (verde - NIR) / (verde + NIR). Agua: valores positivos."""
    v = np.asarray(a_2d(ds["verde"]).values, "float64")
    n = np.asarray(a_2d(ds["nir"]).values, "float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        i = np.where(np.abs(v + n) > 1e-6, (v - n) / (v + n), np.nan)
    return como(i.astype("float32"), a_2d(ds["verde"]), "ndwi")


def agua_ndwi(ds: xr.Dataset, umbral: float = 0.0) -> xr.DataArray:
    i = ndwi(ds)
    return como(np.nan_to_num(i.values, nan=-9.0) > umbral, i, "agua")


def umbral_otsu_db(vv_db: xr.DataArray, tesela: int = 64) -> float:
    """Umbral agua/tierra por Otsu sobre teselas bimodales, acotado a un rango físico.

    El agua suele ocupar poca superficie, y con Otsu global el umbral cae en
    medio de la tierra. Por eso se calcula Otsu en cada tesela y solo se
    aceptan las que tienen dos poblaciones claras (15% a 85% de píxeles oscuros,
    separadas por 4,5 dB o más, con la clase oscura bajo -12 dB). Se devuelve
    la mediana de esos umbrales, o -18 dB si ninguna tesela califica.

    Ajuste tras la validación con datos reales (Biobío, junio de 2023): un río
    crecido tiene agua agitada y más brillante (cerca de -14 dB), que un umbral
    fijo de -18 dB no reconoce.
    """
    from skimage.filters import threshold_otsu

    v = np.clip(np.asarray(vv_db.values, "float64"), -30, 5)
    candidatos = []
    for f in range(0, v.shape[0], tesela):
        for c in range(0, v.shape[1], tesela):
            bloque = v[f:f + tesela, c:c + tesela]
            finito = bloque[np.isfinite(bloque)]
            if finito.size < 0.8 * tesela * tesela * 0.25 or np.ptp(finito) < 1e-3:
                continue
            u = float(threshold_otsu(finito))
            oscuros = finito < u
            fraccion = oscuros.mean()
            if not (0.15 <= fraccion <= 0.85):
                continue
            media_osc, media_cla = finito[oscuros].mean(), finito[~oscuros].mean()
            if media_cla - media_osc >= 4.5 and media_osc < -12.0:
                candidatos.append(u)
    if not candidatos:
        log.info("  sin teselas bimodales; se usa el umbral por defecto de %.1f dB",
                 UMBRAL_DB_POR_DEFECTO)
        return UMBRAL_DB_POR_DEFECTO
    u = float(np.median(candidatos))
    if not (RANGO_UMBRAL_DB[0] <= u <= RANGO_UMBRAL_DB[1]):
        log.info("  Otsu dio %.1f dB (fuera de rango); se usa %.1f dB", u, UMBRAL_DB_POR_DEFECTO)
        return UMBRAL_DB_POR_DEFECTO
    return u


def agua_sar(vv_db: xr.DataArray, umbral_db: Optional[float] = None) -> xr.DataArray:
    """Agua abierta donde VV (dB) queda bajo el umbral (superficie lisa = oscura)."""
    u = umbral_otsu_db(vv_db) if umbral_db is None else float(umbral_db)
    v = np.asarray(vv_db.values, "float64")
    m = como(np.nan_to_num(v, nan=99.0) < u, vv_db, "agua")
    m.attrs["umbral_db"] = u
    return m


def limpiar(mascara: xr.DataArray, area_min_px: int = 8) -> xr.DataArray:
    """Quita manchas menores que ``area_min_px`` (ruido de moteado y sombras)."""
    m = np.asarray(mascara.values).astype(bool)
    m = ndimage.binary_opening(m, structure=np.ones((2, 2), bool))
    etiquetas, n = ndimage.label(m)
    if n:
        tamanos = ndimage.sum(m, etiquetas, index=np.arange(1, n + 1))
        m = np.isin(etiquetas, np.flatnonzero(tamanos >= area_min_px) + 1)
    return como(m, mascara, mascara.name)


def inundacion(
    agua_antes: xr.DataArray,
    agua_despues: xr.DataArray,
    pendiente: Optional[xr.DataArray] = None,
    pendiente_max: float = 8.0,
    area_min_px: int = 8,
    ocurrencia: Optional[xr.DataArray] = None,
    ocurrencia_max: float = 50.0,
) -> xr.DataArray:
    """Agua nueva: presente después y ausente antes.

    Con ``pendiente`` (grados) se descartan laderas, donde el agua no se
    acumula y el SAR suele dar sombras oscuras parecidas al agua.

    Con ``ocurrencia`` (porcentaje histórico de agua, JRC Global Surface
    Water) se descarta el agua habitual: si un píxel tuvo agua en
    ``ocurrencia_max`` % o más de las observaciones históricas, que aparezca
    con agua no es una inundación aunque la escena previa no la haya captado
    (río bajo, viento, sombra).
    """
    nueva = np.asarray(agua_despues.values).astype(bool) & ~np.asarray(agua_antes.values).astype(bool)
    if ocurrencia is not None:
        nueva &= np.nan_to_num(np.asarray(ocurrencia.values), nan=0.0) < ocurrencia_max
    if pendiente is not None:
        nueva &= np.nan_to_num(np.asarray(pendiente.values), nan=0.0) <= pendiente_max
    return limpiar(como(nueva, agua_despues, "inundacion"), area_min_px)
