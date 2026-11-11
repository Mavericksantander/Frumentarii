"""Áreas de interés (AOI): lectura de GeoJSON, bbox y zona UTM."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

from pyproj import CRS, Transformer
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as _transformar
from shapely.ops import unary_union

CARPETA_AOIS = Path(__file__).resolve().parent.parent / "aois"


@dataclass(frozen=True)
class AOI:
    """Área de interés en WGS84 (EPSG:4326)."""

    nombre: str
    geometria: BaseGeometry
    descripcion: str = ""
    aproximada: bool = False

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """(oeste, sur, este, norte) en grados."""
        return tuple(self.geometria.bounds)  # type: ignore[return-value]

    @property
    def centro(self) -> Tuple[float, float]:
        c = self.geometria.centroid
        return (c.x, c.y)

    @property
    def crs_utm(self) -> CRS:
        """Zona UTM que contiene el centroide (326xx norte, 327xx sur)."""
        lon, lat = self.centro
        zona = int((lon + 180.0) // 6.0) % 60 + 1
        return CRS.from_epsg((32600 if lat >= 0 else 32700) + zona)

    @property
    def geometria_utm(self) -> BaseGeometry:
        t = Transformer.from_crs(4326, self.crs_utm, always_xy=True)
        return _transformar(t.transform, self.geometria)

    @property
    def area_km2(self) -> float:
        return self.geometria_utm.area / 1e6

    def a_geojson(self) -> dict:
        return {
            "type": "Feature",
            "properties": {
                "nombre": self.nombre,
                "descripcion": self.descripcion,
                "aproximada": self.aproximada,
            },
            "geometry": mapping(self.geometria),
        }

    def a_geodataframe(self):
        import geopandas as gpd

        return gpd.GeoDataFrame(
            {"nombre": [self.nombre], "aproximada": [self.aproximada]},
            geometry=[self.geometria],
            crs=4326,
        )


def aoi_desde_bbox(nombre: str, bbox: Sequence[float], descripcion: str = "") -> AOI:
    """Crea una AOI rectangular desde (oeste, sur, este, norte)."""
    oeste, sur, este, norte = [float(v) for v in bbox]
    if not (oeste < este and sur < norte):
        raise ValueError("bbox inválido: se espera (oeste, sur, este, norte)")
    return AOI(nombre, box(oeste, sur, este, norte), descripcion, False)


def listar_aois() -> List[str]:
    """Nombres de las AOIs de ejemplo incluidas en el paquete."""
    return sorted(p.stem for p in CARPETA_AOIS.glob("*.geojson"))


def _leer_geojson(ruta: Path) -> AOI:
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    if datos.get("type") == "FeatureCollection":
        entidades = datos["features"]
    elif datos.get("type") == "Feature":
        entidades = [datos]
    else:
        entidades = [{"type": "Feature", "properties": {}, "geometry": datos}]
    if not entidades:
        raise ValueError("El GeoJSON %s no tiene geometrías" % ruta)
    geometria = unary_union([shape(e["geometry"]) for e in entidades])
    if geometria.is_empty or not geometria.is_valid:
        raise ValueError("Geometría vacía o inválida en %s" % ruta)
    oeste, sur, este, norte = geometria.bounds
    if not (-180 <= oeste <= 180 and -90 <= sur <= 90 and este <= 180 and norte <= 90):
        raise ValueError("El GeoJSON %s debe estar en EPSG:4326 (lon/lat)" % ruta)
    props = entidades[0].get("properties") or {}
    return AOI(
        nombre=str(props.get("nombre") or ruta.stem),
        geometria=geometria,
        descripcion=str(props.get("descripcion") or ""),
        aproximada=bool(props.get("aproximada", False)),
    )


def cargar_aoi(ref: Union[AOI, str, Path, Sequence[float]]) -> AOI:
    """Acepta una AOI, un nombre de ejemplo, una ruta GeoJSON o un bbox."""
    if isinstance(ref, AOI):
        return ref
    if isinstance(ref, (list, tuple)):
        return aoi_desde_bbox("bbox", ref)
    texto = str(ref)
    ruta = Path(texto)
    if ruta.suffix.lower() in (".geojson", ".json") and ruta.exists():
        return _leer_geojson(ruta)
    candidato = CARPETA_AOIS / (texto + ".geojson")
    if candidato.exists():
        return _leer_geojson(candidato)
    partes = texto.split(",")
    if len(partes) == 4:
        try:
            return aoi_desde_bbox("bbox", [float(p) for p in partes])
        except ValueError:
            pass
    raise FileNotFoundError(
        "No encuentro la AOI '%s'. Usa una ruta .geojson, un bbox 'O,S,E,N' o una de: %s"
        % (texto, ", ".join(listar_aois()))
    )
