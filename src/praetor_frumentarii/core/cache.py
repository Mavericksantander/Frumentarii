"""Caché local de escenas ya recortadas y reproyectadas.

Cada entrada es una carpeta con un GeoTIFF por variable (una banda por fecha)
y un ``meta.json``. Se ubica en ``~/.cache/frumentarii`` o en la carpeta que
indique la variable de entorno ``FRUMENTARII_CACHE``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xarray as xr

from .registro import obtener_logger

log = obtener_logger("core.cache")


def carpeta_cache() -> Path:
    ruta = Path(os.environ.get("FRUMENTARII_CACHE", Path.home() / ".cache" / "frumentarii"))
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def clave(*partes: Any) -> str:
    """Huella estable de los parámetros de una carga."""
    texto = json.dumps([str(p) for p in partes], ensure_ascii=False)
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:20]


def guardar(ds: xr.Dataset, clave_: str) -> Path:
    import rioxarray  # noqa: F401

    destino = carpeta_cache() / clave_
    temporal = carpeta_cache() / (clave_ + ".tmp")
    if temporal.exists():
        shutil.rmtree(temporal)
    temporal.mkdir(parents=True)
    tiene_tiempo = "time" in ds.dims
    for nombre, da in ds.data_vars.items():
        da = da.transpose(*([d for d in ("time", "y", "x") if d in da.dims]))
        da.rio.write_crs(ds.rio.crs, inplace=False).rio.to_raster(
            temporal / (nombre + ".tif"), compress="deflate", tiled=True
        )
    meta = {
        "variables": list(ds.data_vars),
        "tiempos": [str(t) for t in ds["time"].values] if tiene_tiempo else None,
        "attrs": {k: v for k, v in ds.attrs.items() if isinstance(v, (str, int, float, bool))},
    }
    with open(temporal / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    if destino.exists():
        shutil.rmtree(destino)
    temporal.rename(destino)
    return destino


def leer(clave_: str) -> Optional[xr.Dataset]:
    import rioxarray  # noqa: F401

    from .raster import leer_geotiff

    carpeta = carpeta_cache() / clave_
    meta_ruta = carpeta / "meta.json"
    if not meta_ruta.exists():
        return None
    try:
        with open(meta_ruta, encoding="utf-8") as f:
            meta = json.load(f)
        variables = {}
        crs = None
        for nombre in meta["variables"]:
            da = leer_geotiff(carpeta / (nombre + ".tif"))
            crs = da.rio.crs
            if meta["tiempos"] is None:
                da = da.isel(band=0, drop=True)
            else:
                da = da.rename({"band": "time"}).assign_coords(
                    time=np.array(meta["tiempos"], dtype="datetime64[ns]")
                )
            da.attrs = {}
            da.encoding = {}
            variables[nombre] = da
        ds = xr.Dataset(variables, attrs=meta.get("attrs", {}))
        if "spatial_ref" in ds.coords:
            ds = ds.drop_vars("spatial_ref")
        return ds.rio.write_crs(crs)
    except Exception as e:  # caché dañado: se ignora y se vuelve a descargar
        log.warning("caché %s ilegible (%s); se descarga de nuevo", clave_, e)
        return None


def limpiar() -> int:
    """Borra todo el caché. Devuelve cuántas entradas se eliminaron."""
    n = 0
    for p in carpeta_cache().iterdir():
        if p.is_dir():
            shutil.rmtree(p)
            n += 1
    return n
