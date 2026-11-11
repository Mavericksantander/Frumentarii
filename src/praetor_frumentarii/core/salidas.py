"""Escritura de resultados en ``salidas/<modulo>/<aoi>/``.

Cada corrida deja GeoTIFF, GeoJSON, un PNG resumen y un ``corrida.json`` con
los indicadores y la lista de capas, que es lo que lee el tablero web.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from .. import NOMBRE
from .aoi import AOI, cargar_aoi
from .registro import obtener_logger

log = obtener_logger("core.salidas")


def carpeta_base() -> Path:
    return Path(os.environ.get("FRUMENTARII_SALIDAS", "salidas")).resolve()


def carpeta_salida(modulo: str, aoi: Union[AOI, str]) -> Path:
    nombre = aoi.nombre if isinstance(aoi, AOI) else str(aoi)
    ruta = carpeta_base() / modulo / nombre
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def guardar_geotiff(da: xr.DataArray, modulo: str, aoi: Union[AOI, str], nombre: str) -> Path:
    """Escribe un GeoTIFF comprimido. Los booleanos se guardan como uint8."""
    import rioxarray  # noqa: F401

    ruta = carpeta_salida(modulo, aoi) / (nombre + ".tif")
    if da.dtype == bool:
        da = da.astype("uint8")
    elif da.dtype == np.float64:
        da = da.astype("float32")
    da = da.rio.write_crs(da.rio.crs)
    if np.issubdtype(da.dtype, np.floating):
        da = da.rio.write_nodata(np.nan)
    da.rio.to_raster(ruta, compress="deflate", tiled=True)
    log.info("  GeoTIFF  -> %s", _relativa(ruta))
    return ruta


def guardar_geojson(gdf, modulo: str, aoi: Union[AOI, str], nombre: str) -> Path:
    """Escribe un GeoJSON en EPSG:4326 (también cuando no hay entidades)."""
    ruta = carpeta_salida(modulo, aoi) / (nombre + ".geojson")
    if len(gdf) == 0:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
    else:
        gdf.to_crs(4326).to_file(ruta, driver="GeoJSON")
    log.info("  GeoJSON  -> %s (%d entidades)", _relativa(ruta), len(gdf))
    return ruta


@dataclass
class Panel:
    """Un cuadro del PNG resumen.

    ``datos`` es un DataArray 2D, o un arreglo (y, x, 3) ya realzado con
    ``referencia`` para la georreferencia. ``clases`` mapea valor -> (color, etiqueta).
    ``vectores`` es una lista de (GeoDataFrame, estilo matplotlib).
    """

    datos: Any
    titulo: str
    cmap: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    etiqueta: str = ""
    clases: Optional[Dict[int, Tuple[str, str]]] = None
    vectores: List[Tuple[Any, Dict[str, Any]]] = field(default_factory=list)
    referencia: Optional[xr.DataArray] = None
    fondo: Optional[np.ndarray] = None


def dibujar_panel(ax, panel: Panel) -> None:
    """Dibuja un Panel en un eje matplotlib (lo usan el PNG y el visor en vivo)."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    ref = panel.referencia if panel.referencia is not None else panel.datos
    xs, ys = np.asarray(ref["x"].values), np.asarray(ref["y"].values)
    dx = abs(xs[1] - xs[0]) if xs.size > 1 else 1.0
    dy = abs(ys[1] - ys[0]) if ys.size > 1 else 1.0
    extension = (xs.min() - dx / 2, xs.max() + dx / 2, ys.min() - dy / 2, ys.max() + dy / 2)
    valores = np.asarray(getattr(panel.datos, "values", panel.datos))
    ax.set_facecolor("#1b1f24")
    if panel.fondo is not None:
        ax.imshow(panel.fondo, extent=extension, origin="upper", interpolation="nearest")
    if valores.ndim == 3:
        ax.imshow(valores, extent=extension, origin="upper", interpolation="nearest")
    elif panel.clases:
        claves = sorted(panel.clases)
        cmap = ListedColormap([panel.clases[k][0] for k in claves])
        norma = BoundaryNorm([k - 0.5 for k in claves] + [claves[-1] + 0.5], cmap.N)
        indice = np.full(valores.shape, np.nan, dtype="float32")
        for k in claves:
            indice[valores == k] = k
        ax.imshow(indice, extent=extension, origin="upper", cmap=cmap, norm=norma,
                  interpolation="nearest")
        presentes = [k for k in claves if np.any(valores == k)]
        ax.legend(
            handles=[Patch(color=panel.clases[k][0], label=panel.clases[k][1]) for k in presentes],
            loc="lower left", fontsize=6, framealpha=0.85,
        )
    else:
        v = valores.astype("float32")
        v = np.where(np.isfinite(v), v, np.nan)
        img = ax.imshow(v, extent=extension, origin="upper", cmap=panel.cmap, vmin=panel.vmin,
                        vmax=panel.vmax, interpolation="nearest")
        barra = ax.figure.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
        barra.ax.tick_params(labelsize=7)
        if panel.etiqueta:
            barra.set_label(panel.etiqueta, fontsize=7)
    for gdf, estilo in panel.vectores:
        if gdf is not None and len(gdf):
            crs = getattr(ref, "rio", None)
            g = gdf.to_crs(ref.rio.crs) if crs is not None and ref.rio.crs is not None else gdf
            g.plot(ax=ax, **estilo)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.set_xlim(extension[0], extension[1])
    ax.set_ylim(extension[2], extension[3])
    ax.set_title(panel.titulo, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    # Barra de escala simple (un quinto del ancho, redondeada).
    ancho_m = extension[1] - extension[0]
    if ancho_m > 200:
        largo = _redondo(ancho_m / 5.0)
        x0 = extension[1] - ancho_m * 0.05 - largo
        y0 = extension[2] + (extension[3] - extension[2]) * 0.04
        ax.plot([x0, x0 + largo], [y0, y0], color="white", lw=3, solid_capstyle="butt")
        ax.plot([x0, x0 + largo], [y0, y0], color="black", lw=1, solid_capstyle="butt")
        texto = "%g km" % (largo / 1000.0) if largo >= 1000 else "%g m" % largo
        ax.text(x0 + largo / 2, y0 + (extension[3] - extension[2]) * 0.015, texto, ha="center",
                va="bottom", fontsize=7, color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=1, lw=0))


def _redondo(valor: float) -> float:
    base = 10 ** np.floor(np.log10(valor))
    for m in (5, 2, 1):
        if valor >= m * base:
            return float(m * base)
    return float(base)


def guardar_resumen_png(
    paneles: Sequence[Panel],
    modulo: str,
    aoi: Union[AOI, str],
    titulo: str,
    nombre: str = "resumen",
    pie: str = "",
) -> Path:
    """Compone los paneles en una figura y la guarda como PNG."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = len(paneles)
    columnas = min(n, 3) if n != 4 else 2
    filas = int(np.ceil(n / columnas))
    import textwrap

    # Ancho mínimo para que título y pie no se corten con un solo panel angosto.
    ancho = max(5.2 * columnas, 10.5)
    fig = Figure(figsize=(ancho, 4.6 * filas + (0.9 if n > 1 else 3.4)), dpi=130)
    FigureCanvasAgg(fig)
    for i, panel in enumerate(paneles):
        dibujar_panel(fig.add_subplot(filas, columnas, i + 1), panel)
    fig.suptitle(textwrap.fill("%s | %s" % (NOMBRE, titulo), int(ancho * 9)), fontsize=12,
                 fontweight="bold")
    nota = pie or "Resultado preliminar de 10 a 30 m: validar en terreno antes de decidir."
    fig.text(0.5, 0.012, textwrap.fill(nota, int(ancho * 17)), ha="center", va="bottom", fontsize=7,
             color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    ruta = carpeta_salida(modulo, aoi) / (nombre + ".png")
    fig.savefig(ruta)
    log.info("  PNG      -> %s", _relativa(ruta))
    return ruta


def registrar_corrida(
    modulo: str,
    aoi: Union[AOI, str],
    titulo: str,
    indicadores: Optional[Dict[str, Any]] = None,
    capas: Optional[Sequence[Union[str, Path]]] = None,
    avisos: Optional[Sequence[str]] = None,
    parametros: Optional[Dict[str, Any]] = None,
    fuentes: Optional[Sequence[str]] = None,
    capas_web: Optional[Sequence[Dict[str, Any]]] = None,
    analisis: Optional[str] = None,
    png: Optional[Union[str, Path]] = None,
) -> Path:
    """Escribe ``corrida.json``: el resumen legible por máquina de la corrida."""
    aoi = cargar_aoi(aoi)
    carpeta = carpeta_salida(modulo, aoi)
    datos = {
        "modulo": modulo,
        "aoi": aoi.nombre,
        "aoi_descripcion": aoi.descripcion,
        "aoi_aproximada": aoi.aproximada,
        "bbox": [round(v, 5) for v in aoi.bbox],
        "titulo": titulo,
        "analisis": analisis or "",
        "resumen": Path(png).name if png else "resumen.png",
        "generado": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "indicadores": _limpio(indicadores or {}),
        "capas": [Path(c).name for c in (capas or [])],
        "avisos": list(avisos or []),
        "parametros": _limpio(parametros or {}),
        "fuentes": list(fuentes or []),
        # Cómo dibujar cada capa en el tablero web: archivo, titulo y estilo.
        "capas_web": [dict(c) for c in (capas_web or [])],
    }
    # Un módulo puede dejar varios análisis en la misma AOI: corrida_<analisis>.json
    ruta = carpeta / ("corrida_%s.json" % analisis if analisis else "corrida.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=1)
    log.info("  resumen  -> %s", _relativa(ruta))
    return ruta


def _limpio(d: Dict[str, Any]) -> Dict[str, Any]:
    salida: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            salida[k] = None if not np.isfinite(v) else round(float(v), 4)
        elif isinstance(v, (np.integer,)):
            salida[k] = int(v)
        elif isinstance(v, (str, int, bool)) or v is None:
            salida[k] = v
        else:
            salida[k] = str(v)
    return salida


def _relativa(ruta: Path) -> str:
    try:
        return str(ruta.relative_to(Path.cwd()))
    except ValueError:
        return str(ruta)
