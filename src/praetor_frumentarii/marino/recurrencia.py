"""Recurrencia multifecha de manchas oscuras."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import xarray as xr

from ..core import como


def recurrencia(
    manchas: Sequence[xr.DataArray], validos: Optional[Sequence[xr.DataArray]] = None
) -> xr.Dataset:
    """Cuántas veces cada píxel apareció como mancha.

    Devuelve ``conteo`` (número de fechas con mancha), ``observaciones``
    (fechas con dato válido) y ``frecuencia`` (conteo / observaciones).
    Una mancha que se repite en el mismo lugar sugiere una fuente fija
    (filtración natural, descarga) o un falso positivo persistente (bajo fondo,
    sombra de viento habitual); una que aparece una sola vez, un evento.
    """
    import rioxarray  # noqa: F401

    if not manchas:
        raise ValueError("Se necesita al menos una máscara de manchas")
    ref = manchas[0]
    conteo = np.zeros(ref.shape, dtype="uint16")
    observ = np.zeros(ref.shape, dtype="uint16")
    for i, m in enumerate(manchas):
        if m.shape != ref.shape:
            raise ValueError("Todas las máscaras deben compartir grilla")
        conteo += np.asarray(m.values).astype(bool)
        observ += (np.asarray(validos[i].values).astype(bool) if validos is not None
                   else np.ones(ref.shape, dtype=bool))
    with np.errstate(invalid="ignore", divide="ignore"):
        frecuencia = np.where(observ > 0, conteo / observ.astype("float32"), np.nan)
    return xr.Dataset({
        "conteo": como(conteo, ref), "observaciones": como(observ, ref),
        "frecuencia": como(frecuencia.astype("float32"), ref),
    }).rio.write_crs(ref.rio.crs)
