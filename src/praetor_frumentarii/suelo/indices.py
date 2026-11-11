"""Índices de alteración mineral por cocientes de bandas.

Son indicadores espectrales amplios, no identificación de minerales: a 10-30 m
y con pocas bandas no se separan especies minerales, y el polvo, la humedad,
los relaves, los techos y los caminos de tierra dan respuestas parecidas.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import xarray as xr

from ..core import a_2d, como

# nombre -> (numerador, denominador, descripción) para Sentinel-2 y Landsat 8/9
COCIENTES_OPTICOS: Dict[str, tuple] = {
    "oxidos_hierro": ("rojo", "azul", "Óxidos de hierro férrico (rojo / azul)"),
    "hierro_ferroso": ("swir1", "nir", "Hierro ferroso (SWIR1 / NIR)"),
    "arcillas": ("swir1", "swir2", "Arcillas e hidroxilos (SWIR1 / SWIR2)"),
}
# ASTER: profundidad relativa de banda de absorción = (hombro1 + hombro2) / (2 * centro)
COCIENTES_ASTER: Dict[str, tuple] = {
    "oxidos_hierro": (("b2",), ("b1",), "Óxidos de hierro férrico (B2 / B1)"),
    "alunita_caolinita": (("b4", "b6"), ("b5", "b5"), "Alunita, caolinita, pirofilita ((B4+B6) / 2B5)"),
    "sericita_illita": (("b5", "b7"), ("b6", "b6"), "Sericita, illita, esmectita ((B5+B7) / 2B6)"),
    "clorita_epidota": (("b7", "b9"), ("b8", "b8"), "Clorita, epidota, carbonatos ((B7+B9) / 2B8)"),
}


def cociente(a: xr.DataArray, b: xr.DataArray, minimo: float = 0.005) -> np.ndarray:
    """a / b con NaN donde alguna banda es inválida o demasiado oscura (sombras)."""
    x = np.asarray(a_2d(a).values, "float64")
    y = np.asarray(a_2d(b).values, "float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((y > minimo) & (x > minimo), x / y, np.nan)


def indices_alteracion(ds: xr.Dataset, mascara: Optional[xr.DataArray] = None) -> xr.Dataset:
    """Calcula los cocientes que correspondan al sensor del Dataset.

    Detecta ASTER por la presencia de ``b6``; si no, usa las bandas ópticas
    comunes (azul, rojo, nir, swir1, swir2). ``mascara`` es True donde el píxel
    sirve (suelo desnudo); el resto queda en NaN.
    """
    import rioxarray  # noqa: F401

    ref = a_2d(ds[list(ds.data_vars)[0]])
    valido = np.asarray(mascara.values).astype(bool) if mascara is not None else np.ones(ref.shape, bool)
    salida = xr.Dataset()
    if "b6" in ds:
        for nombre, (num, den, descripcion) in COCIENTES_ASTER.items():
            n = sum(np.asarray(a_2d(ds[b]).values, "float64") for b in num)
            d = sum(np.asarray(a_2d(ds[b]).values, "float64") for b in den)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(d > 0.005, n / d, np.nan)
            salida[nombre] = como(np.where(valido, v, np.nan).astype("float32"), ref)
            salida[nombre].attrs["descripcion"] = descripcion
    else:
        for nombre, (num, den, descripcion) in COCIENTES_OPTICOS.items():
            if num not in ds or den not in ds:
                continue
            v = cociente(ds[num], ds[den])
            salida[nombre] = como(np.where(valido, v, np.nan).astype("float32"), ref)
            salida[nombre].attrs["descripcion"] = descripcion
    return salida.rio.write_crs(ds.rio.crs)


def anomalias(indice: xr.DataArray, percentil: float = 97.0) -> xr.DataArray:
    """Máscara de los valores más altos del índice (sobre el percentil dado)."""
    v = np.asarray(indice.values, "float64")
    finitos = v[np.isfinite(v)]
    if finitos.size == 0:
        return como(np.zeros(v.shape, bool), indice, "anomalia")
    umbral = float(np.percentile(finitos, percentil))
    m = como(np.nan_to_num(v, nan=-np.inf) > umbral, indice, "anomalia")
    m.attrs["umbral"] = umbral
    return m
