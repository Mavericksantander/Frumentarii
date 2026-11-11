"""Técnica de Crosta: análisis de componentes principales dirigido.

Se hace PCA sobre cuatro bandas elegidas para un grupo mineral y se busca la
componente cuyos pesos tienen signo opuesto en el par de bandas diagnóstico
(una donde el mineral refleja y otra donde absorbe). Esa componente realza el
mineral y atenúa albedo y topografía.

Variante usada aquí: antes del PCA se toma el logaritmo de la reflectancia y
se resta a cada píxel la media de sus bandas (log-cociente centrado). La
iluminación multiplica todas las bandas por el mismo factor, así que en
logaritmos es una constante por píxel que esa resta elimina exactamente. En el
Crosta clásico se confía en que el brillo quede en la primera componente, pero
en las pruebas sintéticas se filtraba a las demás (correlación de 0,9 con la
iluminación); con esta variante el resultado no depende del relieve.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from ..core import a_2d, como

# objetivo -> (bandas del PCA, banda que refleja, banda que absorbe)
CONFIGURACION: Dict[str, Dict[str, Tuple[Sequence[str], str, str]]] = {
    "optico": {
        "hierro": (("azul", "rojo", "nir", "swir1"), "rojo", "azul"),
        "arcillas": (("azul", "nir", "swir1", "swir2"), "swir1", "swir2"),
    },
    "aster": {
        "hierro": (("b1", "b2", "b3", "b4"), "b2", "b1"),
        "arcillas": (("b1", "b3", "b4", "b6"), "b4", "b6"),
    },
}


def elegir_componente(pesos: np.ndarray, i_refleja: int, i_absorbe: int,
                      desde: int = 1) -> Tuple[int, float]:
    """Índice y signo de la componente con mayor contraste entre el par diagnóstico.

    Se consideran las componentes a partir de ``desde`` (1 salta la primera,
    que en el Crosta clásico es el albedo) y se exige signo opuesto en el par;
    si ninguna lo cumple se toma la de mayor diferencia de pesos.
    """
    mejor, mejor_valor, signo = desde, -np.inf, 1.0
    opuestos = [k for k in range(desde, pesos.shape[0])
                if pesos[k, i_refleja] * pesos[k, i_absorbe] < 0]
    for k in (opuestos or range(desde, pesos.shape[0])):
        valor = abs(pesos[k, i_refleja] - pesos[k, i_absorbe])
        if valor > mejor_valor:
            mejor, mejor_valor = k, valor
            signo = 1.0 if pesos[k, i_refleja] > pesos[k, i_absorbe] else -1.0
    return mejor, signo


def crosta(
    ds: xr.Dataset, objetivo: str = "arcillas", mascara: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """Imagen de Crosta (puntaje z; alto = más probable el grupo mineral).

    ``objetivo`` es 'hierro' u 'arcillas'. El PCA se ajusta solo con los
    píxeles válidos de ``mascara`` (suelo expuesto) para que vegetación y agua
    no dominen la varianza. Los pesos quedan en ``attrs`` para auditarlos.
    """
    from sklearn.decomposition import PCA

    familia = "aster" if "b6" in ds else "optico"
    if objetivo not in CONFIGURACION[familia]:
        raise ValueError("objetivo debe ser 'hierro' o 'arcillas'")
    bandas, refleja, absorbe = CONFIGURACION[familia][objetivo]
    faltan = [b for b in bandas if b not in ds]
    if faltan:
        raise ValueError("Faltan bandas para Crosta %s: %s" % (objetivo, faltan))
    ref = a_2d(ds[bandas[0]])
    cubo = np.stack([np.asarray(a_2d(ds[b]).values, "float64") for b in bandas], axis=-1)
    valido = np.isfinite(cubo).all(axis=-1) & (cubo > 0).all(axis=-1)
    if mascara is not None:
        valido &= np.asarray(mascara.values).astype(bool)
    salida = np.full(ref.shape, np.nan, dtype="float32")
    if valido.sum() < 50:
        da = como(salida, ref, "crosta_%s" % objetivo)
        da.attrs["aviso"] = "muy pocos píxeles válidos para el PCA"
        return da
    x = np.log(cubo[valido])
    x = x - x.mean(axis=1, keepdims=True)      # quita el brillo común (iluminación, albedo)
    x = x - x.mean(axis=0)
    # Tras centrar por píxel el rango es n-1: la última componente es nula.
    pca = PCA(n_components=len(bandas) - 1)
    puntajes = pca.fit_transform(x)
    k, signo = elegir_componente(pca.components_, bandas.index(refleja), bandas.index(absorbe),
                                 desde=0)
    z = puntajes[:, k] * signo
    z = (z - z.mean()) / (z.std() + 1e-12)
    salida[valido] = z.astype("float32")
    da = como(salida, ref, "crosta_%s" % objetivo)
    da.attrs.update(
        componente=int(k + 1), bandas=", ".join(bandas),
        pesos=", ".join("%s=%+.2f" % (b, p * signo) for b, p in zip(bandas, pca.components_[k])),
        varianza_pct=round(float(pca.explained_variance_ratio_[k] * 100), 2),
    )
    return da
