"""Cambios con Sentinel-1: retrodispersión antes/después y proxy de coherencia.

IMPORTANTE: la coherencia interferométrica real necesita productos SLC y un
procesador InSAR. Aquí se usa un *proxy*: la correlación local entre las
imágenes de retrodispersión (dB) de antes y después. Es menos sensible que la
coherencia real y solo sirve como indicio de dónde mirar.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr
from scipy import ndimage

from ..core import a_decibeles, como


def filtro_moteado(lineal: xr.DataArray, ventana: int = 5) -> xr.DataArray:
    """Media móvil (boxcar) sobre potencia lineal que ignora los NaN."""
    v = np.asarray(lineal.values, dtype="float64")
    valido = np.isfinite(v)
    suma = ndimage.uniform_filter(np.where(valido, v, 0.0), ventana, mode="nearest")
    peso = ndimage.uniform_filter(valido.astype("float64"), ventana, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        media = np.where(peso > 0.2, suma / peso, np.nan)
    return como(media.astype("float32"), lineal, lineal.name)


def log_ratio(antes: xr.DataArray, despues: xr.DataArray, ventana: int = 5) -> xr.DataArray:
    """Cambio en dB (después menos antes) de retrodispersión lineal filtrada."""
    a = a_decibeles(filtro_moteado(antes, ventana))
    d = a_decibeles(filtro_moteado(despues, ventana))
    return como((d.values - a.values).astype("float32"), antes, "log_ratio")


def proxy_coherencia(antes_db: xr.DataArray, despues_db: xr.DataArray, ventana: int = 7) -> xr.DataArray:
    """Correlación de Pearson local (0 a 1) entre dos imágenes en dB.

    Valores bajos indican que la textura local cambió (posible daño, remoción
    en masa o inundación). Valores negativos se recortan a 0.
    """
    a = np.asarray(antes_db.values, dtype="float64")
    d = np.asarray(despues_db.values, dtype="float64")
    valido = np.isfinite(a) & np.isfinite(d)
    a0 = np.where(valido, a, 0.0)
    d0 = np.where(valido, d, 0.0)

    def media(x: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(x, ventana, mode="nearest")

    n = np.maximum(media(valido.astype("float64")), 1e-9)
    ma, md = media(a0) / n, media(d0) / n
    cov = media(a0 * d0) / n - ma * md
    va = np.clip(media(a0 * a0) / n - ma * ma, 0, None)
    vd = np.clip(media(d0 * d0) / n - md * md, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vd)
    r = np.where((va > 1e-6) & (vd > 1e-6) & valido, np.clip(r, 0, 1), np.nan)
    return como(r.astype("float32"), antes_db, "coherencia_proxy")


def clasificar_cambio(lr: xr.DataArray, umbral_db: float = 3.0) -> xr.DataArray:
    """-1 baja de señal (agua nueva, superficies alisadas), 1 alza (escombros,
    suelo removido, vegetación inundada), 0 sin cambio relevante."""
    v = np.asarray(lr.values)
    clases = np.zeros(v.shape, dtype="int8")
    clases[v <= -umbral_db] = -1
    clases[v >= umbral_db] = 1
    return como(clases, lr, "cambio")


def posible_dano_urbano(
    coherencia: xr.DataArray,
    lr: xr.DataArray,
    construido: xr.DataArray,
    umbral_coherencia: float = 0.3,
    umbral_db: float = 2.0,
    area_min_px: int = 6,
) -> xr.DataArray:
    """Zonas construidas con textura alterada y cambio de señal: indicio de daño.

    ``construido`` es una máscara booleana (por ejemplo WorldCover == 50).
    No distingue daño de obras, demoliciones o cambios de humedad.
    """
    candidato = (
        (np.asarray(coherencia.values) < umbral_coherencia)
        & (np.abs(np.asarray(lr.values)) >= umbral_db)
        & np.asarray(construido.values).astype(bool)
    )
    etiquetas, n = ndimage.label(candidato)
    if n:
        tamanos = ndimage.sum(candidato, etiquetas, index=np.arange(1, n + 1))
        candidato = np.isin(etiquetas, np.flatnonzero(tamanos >= area_min_px) + 1)
    return como(candidato, coherencia, "posible_dano")


def cambio_sar(antes: xr.Dataset, despues: xr.Dataset, banda: str = "vv",
               ventana: int = 5) -> xr.Dataset:
    """Dataset con dB antes/después, ``log_ratio``, ``coherencia_proxy`` y ``cambio``.

    ``antes`` y ``despues`` son Datasets Sentinel-1 lineales (con o sin tiempo);
    si hay varias fechas se usa la mediana temporal de cada período.
    """
    import rioxarray  # noqa: F401

    def media_temporal(ds: xr.Dataset) -> xr.DataArray:
        da = ds[banda]
        if "time" in da.dims:
            da = da.median("time", skipna=True)
        return da.rio.write_crs(ds.rio.crs)

    a, d = media_temporal(antes), media_temporal(despues)
    a_db = a_decibeles(filtro_moteado(a, 3))
    d_db = a_decibeles(filtro_moteado(d, 3))
    lr = log_ratio(a, d, ventana)
    salida = xr.Dataset({
        "antes_db": como(a_db.values, a), "despues_db": como(d_db.values, a),
        "log_ratio": lr, "coherencia_proxy": proxy_coherencia(a_db, d_db),
        "cambio": clasificar_cambio(lr),
    })
    return salida.rio.write_crs(a.rio.crs)
