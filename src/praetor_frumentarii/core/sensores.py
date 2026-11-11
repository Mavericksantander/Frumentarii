"""Cargadores de alto nivel por sensor, con nombres de banda comunes.

Ajustes verificados contra los catálogos reales (noviembre de 2026):

- Sentinel-2 L2A en Earth Search usa assets ``blue, green, red, nir, nir08,
  swir16, swir22, scl``. Con ``earthsearch:boa_offset_applied = true`` el
  desplazamiento de la línea base 04.00 o posterior ya viene aplicado, así que
  la reflectancia es ``DN * 0.0001`` (el ``offset: -0.1`` que declara
  ``raster:bands`` NO debe aplicarse otra vez).
- Sentinel-2 L2A en Planetary Computer usa ``B02, B03, B04, B08, B8A, B11,
  B12, SCL`` y NO trae el desplazamiento aplicado: desde la línea base 04.00
  la reflectancia es ``(DN - 1000) * 0.0001``.
- Sentinel-1 se lee de ``sentinel-1-rtc`` en Planetary Computer (gamma0 lineal
  con corrección de terreno, assets ``vv`` y ``vh``). El GRD de ambos catálogos
  no está georreferenciado en grilla y el de Earth Search es de pago por
  solicitante.
- Landsat Colección 2 Nivel 2 se lee de Planetary Computer: reflectancia
  ``DN * 0.0000275 - 0.2``.
- ASTER L1T (Planetary Computer, ``aster-l1t``) trae assets multibanda ``VNIR``
  (bandas 1, 2, 3N a 15 m) y ``SWIR`` (bandas 4 a 9 a 30 m) en DN de 8 bits. En
  odc-stac se piden como ``VNIR.1`` ... ``SWIR.6`` (índice desde 1). El SWIR de
  ASTER falló en abril de 2008: solo hay escenas útiles entre 2000 y 2008.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import xarray as xr

from .aoi import AOI, cargar_aoi
from .carga import cargar
from .nubes import enmascarar_nubes
from .registro import obtener_logger
from .stac import Fechas, buscar

log = obtener_logger("core.sensores")

BANDAS_S2: Dict[str, Dict[str, str]] = {
    "earth-search": {
        "azul": "blue", "verde": "green", "rojo": "red", "nir": "nir", "nir08": "nir08",
        "swir1": "swir16", "swir2": "swir22", "scl": "scl",
    },
    "planetary": {
        "azul": "B02", "verde": "B03", "rojo": "B04", "nir": "B08", "nir08": "B8A",
        "swir1": "B11", "swir2": "B12", "scl": "SCL",
    },
}
BANDAS_LANDSAT = {
    "azul": "blue", "verde": "green", "rojo": "red", "nir": "nir08",
    "swir1": "swir16", "swir2": "swir22", "qa_pixel": "qa_pixel",
}
OPTICAS = ("azul", "verde", "rojo", "nir", "swir1", "swir2")


def desplazamiento_s2(item: Any, catalogo: str) -> float:
    """DN que hay que restar antes de escalar (0 o 1000) para un ítem Sentinel-2."""
    props = item.properties
    if props.get("earthsearch:boa_offset_applied"):
        return 0.0
    try:
        linea_base = float(str(props.get("s2:processing_baseline", "0")))
    except ValueError:
        linea_base = 0.0
    return 1000.0 if linea_base >= 4.0 else 0.0


def dn_a_reflectancia_s2(ds: xr.Dataset, items: Sequence[Any], catalogo: str) -> xr.Dataset:
    """Convierte DN a reflectancia, con el desplazamiento correcto por fecha."""
    por_fecha: Dict[str, float] = {}
    for it in items:
        por_fecha[it.datetime.strftime("%Y-%m-%d")] = desplazamiento_s2(it, catalogo)
    comun = Counter(por_fecha.values()).most_common(1)[0][0] if por_fecha else 0.0
    fechas = [str(t)[:10] for t in ds["time"].values]
    resta = xr.DataArray(
        np.array([por_fecha.get(f, comun) for f in fechas], dtype="float32"),
        dims="time", coords={"time": ds["time"]},
    )
    salida = xr.Dataset(attrs=dict(ds.attrs))
    for nombre, da in ds.data_vars.items():
        if nombre == "scl":
            salida[nombre] = da
            continue
        dn = da.astype("float32")
        salida[nombre] = ((dn - resta) * 1e-4).where(dn > 0).clip(min=0.0)
    return salida.rio.write_crs(ds.rio.crs)


def cargar_s2(
    aoi: Union[AOI, str],
    fechas: Fechas,
    bandas: Sequence[str] = OPTICAS,
    catalogo: str = "earth-search",
    nubes_max: Optional[float] = 60.0,
    resolucion: float = 10.0,
    limite: Optional[int] = None,
    enmascarar: bool = True,
    items: Optional[Sequence[Any]] = None,
) -> xr.Dataset:
    """Sentinel-2 L2A como reflectancia (0 a 1), con nubes en NaN."""
    import rioxarray  # noqa: F401

    aoi = cargar_aoi(aoi)
    tabla = BANDAS_S2[catalogo]
    if items is None:
        items = buscar(aoi, "sentinel-2-l2a", fechas, catalogo=catalogo, nubes_max=nubes_max,
                       limite=limite)
    if not items:
        raise LookupError(
            "Sin escenas Sentinel-2 para %s en %s con nubes < %s%%. Amplía fechas o sube nubes_max."
            % (aoi.nombre, fechas, nubes_max)
        )
    pedir: List[str] = list(bandas) + (["scl"] if enmascarar and "scl" not in bandas else [])
    crudo = cargar(items, aoi, [tabla[b] for b in pedir], resolucion=resolucion,
                   remuestreo="nearest" if resolucion <= 10 else "bilinear")
    crudo = crudo.rename({tabla[b]: b for b in pedir})
    if "scl" in crudo:
        # La capa de clases nunca se interpola.
        crudo["scl"] = crudo["scl"].round().astype("uint8")
    ds = dn_a_reflectancia_s2(crudo, items, catalogo)
    if enmascarar:
        ds = enmascarar_nubes(ds, "scl").rio.write_crs(crudo.rio.crs)
    ds.attrs.update(sensor="Sentinel-2 L2A", catalogo=catalogo)
    return ds


def cargar_s1(
    aoi: Union[AOI, str],
    fechas: Fechas,
    polarizaciones: Sequence[str] = ("vv", "vh"),
    resolucion: float = 20.0,
    orbita: Optional[str] = None,
    limite: Optional[int] = None,
    items: Optional[Sequence[Any]] = None,
) -> xr.Dataset:
    """Sentinel-1 RTC (gamma0 lineal) desde Planetary Computer; sin dato = NaN.

    ``orbita`` puede ser 'ascending' o 'descending'. Para comparar antes y
    después conviene usar la misma órbita relativa (ver ``misma_orbita``).
    """
    import rioxarray  # noqa: F401

    aoi = cargar_aoi(aoi)
    if items is None:
        consulta = {"sat:orbit_state": {"eq": orbita}} if orbita else None
        items = buscar(aoi, "sentinel-1-rtc", fechas, catalogo="planetary", limite=limite,
                       consulta=consulta)
    if not items:
        raise LookupError("Sin escenas Sentinel-1 RTC para %s en %s" % (aoi.nombre, fechas))
    crudo = cargar(items, aoi, list(polarizaciones), resolucion=resolucion, remuestreo="average",
                   dtype="float32", nodata=-32768)
    ds = crudo.astype("float32").where(crudo > 0)
    ds = ds.rio.write_crs(crudo.rio.crs)
    ds.attrs.update(sensor="Sentinel-1 RTC", catalogo="planetary")
    return ds


def misma_orbita(items: Sequence[Any], referencia: Optional[Any] = None) -> List[Any]:
    """Filtra los ítems Sentinel-1 a una sola órbita relativa (la más frecuente)."""
    items = list(items)
    if not items:
        return items
    if referencia is not None:
        objetivo = referencia.properties.get("sat:relative_orbit")
    else:
        objetivo = Counter(
            it.properties.get("sat:relative_orbit") for it in items
        ).most_common(1)[0][0]
    return [it for it in items if it.properties.get("sat:relative_orbit") == objetivo]


def cargar_landsat(
    aoi: Union[AOI, str],
    fechas: Fechas,
    bandas: Sequence[str] = OPTICAS,
    nubes_max: Optional[float] = 60.0,
    resolucion: float = 30.0,
    limite: Optional[int] = None,
    enmascarar: bool = True,
) -> xr.Dataset:
    """Landsat 8/9 Colección 2 Nivel 2 como reflectancia, desde Planetary Computer."""
    import rioxarray  # noqa: F401

    aoi = cargar_aoi(aoi)
    items = buscar(aoi, "landsat-c2-l2", fechas, catalogo="planetary", nubes_max=nubes_max,
                   limite=limite, consulta={"platform": {"in": ["landsat-8", "landsat-9"]}})
    if not items:
        raise LookupError("Sin escenas Landsat 8/9 para %s en %s" % (aoi.nombre, fechas))
    pedir = list(bandas) + ["qa_pixel"]
    crudo = cargar(items, aoi, [BANDAS_LANDSAT[b] for b in pedir], resolucion=resolucion,
                   remuestreo="nearest")
    crudo = crudo.rename({BANDAS_LANDSAT[b]: b for b in pedir if BANDAS_LANDSAT[b] != b})
    ds = xr.Dataset(attrs=dict(crudo.attrs))
    for b in bandas:
        dn = crudo[b].astype("float32")
        ds[b] = (dn * 0.0000275 - 0.2).where(dn > 0).clip(min=0.0)
    ds["qa_pixel"] = crudo["qa_pixel"]
    ds = ds.rio.write_crs(crudo.rio.crs)
    if enmascarar:
        ds = enmascarar_nubes(ds, "qa_pixel").rio.write_crs(crudo.rio.crs)
    else:
        ds = ds.drop_vars("qa_pixel")
    ds.attrs.update(sensor="Landsat 8/9 C2 L2", catalogo="planetary")
    return ds


def cargar_dem(
    aoi: Union[AOI, str], resolucion: float = 30.0, catalogo: str = "earth-search"
) -> xr.DataArray:
    """Copernicus DEM GLO-30 en metros, en la grilla UTM de la AOI."""
    import rioxarray  # noqa: F401

    aoi = cargar_aoi(aoi)
    items = buscar(aoi, "cop-dem-glo-30", catalogo=catalogo)
    if not items:
        raise LookupError("Sin teselas Copernicus DEM para %s" % aoi.nombre)
    ds = cargar(items, aoi, ["data"], resolucion=resolucion, remuestreo="bilinear", agrupar=None)
    # Varias teselas: se fusionan tomando el primer valor válido.
    dem = ds["data"].astype("float32")
    dem = dem.where(dem > -1000).max("time") if dem.sizes["time"] > 1 else dem.isel(time=0, drop=True)
    dem.name = "elevacion"
    return dem.rio.write_crs(ds.rio.crs)


def cargar_worldcover(aoi: Union[AOI, str], como: xr.DataArray, anio: int = 2021) -> xr.DataArray:
    """ESA WorldCover (clases 10 a 100) remuestreado a la grilla de ``como``."""
    import rioxarray  # noqa: F401

    aoi = cargar_aoi(aoi)
    items = buscar(aoi, "esa-worldcover", catalogo="planetary")
    del_anio = [it for it in items if str(anio) in it.id] or items
    if not del_anio:
        raise LookupError("Sin teselas ESA WorldCover para %s" % aoi.nombre)
    ds = cargar(del_anio, aoi, ["map"], como=como, remuestreo="mode", agrupar=None)
    capa = ds["map"].max("time").astype("uint8")
    capa.name = "cobertura"
    return capa.rio.write_crs(como.rio.crs)


def mostrar_escenas(visor: Any, ds: xr.Dataset, tipo: str = "rgb", prefijo: str = "") -> None:
    """Envía cada fecha del Dataset al visor en vivo (si hay visor)."""
    if visor is None or "time" not in ds.dims:
        return
    from .raster import componer_rgb
    from .salidas import Panel

    for i in range(ds.sizes["time"]):
        escena = ds.isel(time=i)
        fecha = str(ds["time"].values[i])[:10]
        if tipo == "rgb":
            visor.mostrar(Panel(componer_rgb(escena), "%s Sentinel-2 %s" % (prefijo, fecha),
                                referencia=escena["rojo"]))
        else:
            from .carga import a_decibeles

            banda = "vv" if "vv" in escena else list(escena.data_vars)[0]
            visor.mostrar(Panel(a_decibeles(escena[banda]), "%s Sentinel-1 %s %s"
                                % (prefijo, banda.upper(), fecha), "gray", -25, 0, "dB"))


def compuesto_s2(
    aoi: Union[AOI, str],
    fechas: Fechas,
    bandas: Sequence[str] = OPTICAS,
    catalogo: str = "earth-search",
    nubes_max: Optional[float] = 60.0,
    resolucion: float = 20.0,
    max_escenas: int = 4,
    visor: Any = None,
    prefijo: str = "",
) -> xr.Dataset:
    """Mediana sin nubes de las ``max_escenas`` fechas menos nubosas del período."""
    import rioxarray  # noqa: F401

    from .nubes import compuesto_mediana

    aoi = cargar_aoi(aoi)
    items = buscar(aoi, "sentinel-2-l2a", fechas, catalogo=catalogo, nubes_max=nubes_max)
    if not items:
        raise LookupError(
            "Sin escenas Sentinel-2 para %s en %s con nubes < %s%%" % (aoi.nombre, fechas, nubes_max)
        )
    # Se eligen los días menos nubosos (todas las teselas de esos días).
    por_dia: Dict[str, List[Any]] = {}
    for it in items:
        por_dia.setdefault(it.datetime.strftime("%Y-%m-%d"), []).append(it)
    dias = sorted(por_dia, key=lambda d: np.mean(
        [it.properties.get("eo:cloud_cover", 100) for it in por_dia[d]]))[:max_escenas]
    elegidos = [it for d in sorted(dias) for it in por_dia[d]]
    log.info("  se usan %d día(s): %s", len(dias), ", ".join(sorted(dias)))
    ds = cargar_s2(aoi, fechas, bandas, catalogo, nubes_max, resolucion, items=elegidos)
    mostrar_escenas(visor, ds, "rgb", prefijo)
    compuesto = compuesto_mediana(ds).rio.write_crs(ds.rio.crs)
    compuesto.attrs.update(ds.attrs, fechas_usadas=", ".join(sorted(dias)))
    return compuesto


def par_s1(
    aoi: Union[AOI, str],
    antes: Fechas,
    despues: Fechas,
    polarizaciones: Sequence[str] = ("vv", "vh"),
    resolucion: float = 20.0,
    max_escenas: int = 3,
    visor: Any = None,
):
    """Sentinel-1 antes y después con la MISMA órbita relativa (misma geometría).

    Devuelve (ds_antes, ds_despues) con dimensión de tiempo. Comparar órbitas
    distintas produce falsos cambios por el ángulo de observación.
    """
    aoi = cargar_aoi(aoi)
    items_a = buscar(aoi, "sentinel-1-rtc", antes, catalogo="planetary")
    items_d = buscar(aoi, "sentinel-1-rtc", despues, catalogo="planetary")
    if not items_a or not items_d:
        raise LookupError("Faltan escenas Sentinel-1 antes (%d) o después (%d) para %s"
                          % (len(items_a), len(items_d), aoi.nombre))
    orbitas_a = set(it.properties.get("sat:relative_orbit") for it in items_a)
    comunes = [it for it in items_d if it.properties.get("sat:relative_orbit") in orbitas_a]
    if not comunes:
        raise LookupError("No hay una órbita relativa común entre antes y después; amplía fechas")
    # La primera pasada posterior al evento manda: se usa su órbita.
    orbita = comunes[0].properties.get("sat:relative_orbit")
    sel_a = [it for it in items_a if it.properties.get("sat:relative_orbit") == orbita][-max_escenas:]
    sel_d = [it for it in items_d if it.properties.get("sat:relative_orbit") == orbita][:max_escenas]
    log.info("  órbita relativa %s: %d escena(s) antes, %d después", orbita, len(sel_a), len(sel_d))
    ds_a = cargar_s1(aoi, antes, polarizaciones, resolucion, items=sel_a)
    mostrar_escenas(visor, ds_a, "sar", "Antes:")
    ds_d = cargar_s1(aoi, despues, polarizaciones, resolucion, items=sel_d)
    mostrar_escenas(visor, ds_d, "sar", "Después:")
    return ds_a, ds_d


# ASTER: banda -> (asset.indice, coeficientes de conversión a radiancia por ganancia
# [W m-2 sr-1 um-1 por DN] según el manual de usuario de ASTER, irradiancia solar ESUN
# [W m-2 um-1] de Thome et al.)
ASTER = {
    "b1": ("VNIR.1", {"HGH": 0.676, "NOR": 1.688, "LOW": 2.25}, 1848.0),
    "b2": ("VNIR.2", {"HGH": 0.708, "NOR": 1.415, "LOW": 1.89}, 1549.0),
    "b3": ("VNIR.3", {"HGH": 0.423, "NOR": 0.862, "LOW": 1.15}, 1114.0),
    "b4": ("SWIR.1", {"HGH": 0.1087, "NOR": 0.2174, "LO1": 0.290, "LO2": 0.290}, 225.4),
    "b5": ("SWIR.2", {"HGH": 0.0348, "NOR": 0.0696, "LO1": 0.0925, "LO2": 0.409}, 86.63),
    "b6": ("SWIR.3", {"HGH": 0.0313, "NOR": 0.0625, "LO1": 0.0830, "LO2": 0.390}, 81.85),
    "b7": ("SWIR.4", {"HGH": 0.0299, "NOR": 0.0597, "LO1": 0.0795, "LO2": 0.332}, 74.85),
    "b8": ("SWIR.5", {"HGH": 0.0209, "NOR": 0.0417, "LO1": 0.0556, "LO2": 0.245}, 66.49),
    "b9": ("SWIR.6", {"HGH": 0.0159, "NOR": 0.0318, "LO1": 0.0424, "LO2": 0.265}, 59.85),
}
_CLAVE_GANANCIA = {"b1": "01", "b2": "02", "b3": "3N", "b4": "04", "b5": "05", "b6": "06",
                   "b7": "07", "b8": "08", "b9": "09"}


def ganancias_aster(texto_xml: str) -> Dict[str, str]:
    """Lee ``ASTERGains`` del XML de la escena: {'b1': 'NOR', ...}. Vacío si no está."""
    import re

    m = re.search(r"<PSAName>ASTERGains</PSAName>\s*<PSAValue>(.*?)</PSAValue>", texto_xml, re.S)
    if not m:
        return {}
    pares = dict(p.strip().split() for p in m.group(1).split(",") if len(p.split()) == 2)
    return {b: pares[c] for b, c in _CLAVE_GANANCIA.items() if c in pares}


def _ganancias_de_item(item: Any) -> Dict[str, str]:
    import urllib.request

    try:
        with urllib.request.urlopen(item.assets["xml"].href, timeout=30) as r:
            return ganancias_aster(r.read().decode("utf-8", "replace"))
    except Exception as e:
        log.warning("  no pude leer las ganancias ASTER de %s (%s); se supone ganancia normal",
                    item.id, type(e).__name__)
        return {}


def dn_a_reflectancia_aster(dn: xr.DataArray, banda: str, elevacion_solar: float,
                            ganancia: str = "NOR") -> xr.DataArray:
    """Reflectancia aparente (al tope de la atmósfera) de ASTER L1T.

    radiancia = (DN - 1) * coeficiente(ganancia);
    reflectancia = pi * L / (ESUN * sen(elevación solar)).
    No hay corrección atmosférica: el visible queda sobrestimado por la
    dispersión, lo que afecta los cocientes con la banda 1.
    """
    _, coefs, esun = ASTER[banda]
    coef = coefs.get(ganancia, coefs["NOR"])
    d = dn.astype("float32")
    radiancia = (d - 1.0) * coef
    refl = np.pi * radiancia / (esun * np.sin(np.radians(elevacion_solar)))
    return refl.where(d > 1).clip(min=0.0).astype("float32")


def cargar_aster(
    aoi: Union[AOI, str],
    fechas: Fechas = "2000-03-01/2008-03-31",
    nubes_max: Optional[float] = 10.0,
    resolucion: float = 30.0,
    max_escenas: int = 2,
) -> xr.Dataset:
    """Compuesto ASTER L1T (bandas b1 a b9, reflectancia aparente) desde Planetary Computer."""
    import rioxarray  # noqa: F401

    aoi = cargar_aoi(aoi)
    items = buscar(aoi, "aster-l1t", fechas, catalogo="planetary", nubes_max=nubes_max)
    completos = [it for it in items if "VNIR" in it.assets and "SWIR" in it.assets]
    if not completos:
        raise LookupError(
            "Sin escenas ASTER con VNIR y SWIR para %s en %s (el SWIR solo existe hasta abril "
            "de 2008)" % (aoi.nombre, fechas)
        )
    # Se prefieren las escenas que cubren más de la AOI y con menos nubes.
    from shapely.geometry import shape

    def cobertura(it: Any) -> float:
        try:
            return shape(it.geometry).intersection(aoi.geometria).area / aoi.geometria.area
        except Exception:
            return 0.0

    completos.sort(key=lambda it: (-round(cobertura(it), 1), it.properties.get("eo:cloud_cover", 100)))
    elegidos = completos[:max_escenas]
    log.info("  ASTER: %d escena(s): %s", len(elegidos),
             ", ".join(it.datetime.strftime("%Y-%m-%d") for it in elegidos))
    capas: Dict[str, List[xr.DataArray]] = {b: [] for b in ASTER}
    crs = None
    referencia = None
    for it in elegidos:
        crudo = cargar([it], aoi, [ASTER[b][0] for b in ASTER], resolucion=resolucion,
                       remuestreo="bilinear", como=referencia)
        crs = crudo.rio.crs
        elev = float(it.properties.get("view:sun_elevation", 60.0))
        ganancias = _ganancias_de_item(it)
        log.info("  %s: elevación solar %.1f°, ganancias %s", it.datetime.strftime("%Y-%m-%d"), elev,
                 " ".join("%s=%s" % kv for kv in ganancias.items()) or "desconocidas (NOR)")
        for b in ASTER:
            da = crudo[ASTER[b][0]].isel(time=0, drop=True)
            if referencia is None:
                referencia = da.rio.write_crs(crs)
            capas[b].append(dn_a_reflectancia_aster(da, b, elev, ganancias.get(b, "NOR")))
    ds = xr.Dataset()
    for b in ASTER:
        pila = xr.concat(capas[b], dim="escena")
        ds[b] = pila.median("escena", skipna=True) if len(capas[b]) > 1 else pila.isel(escena=0, drop=True)
    ds = ds.rio.write_crs(crs)
    ds.attrs.update(sensor="ASTER L1T", catalogo="planetary",
                    fechas_usadas=", ".join(it.datetime.strftime("%Y-%m-%d") for it in elegidos))
    return ds


def cargar_agua_historica(aoi: Union[AOI, str], como: xr.DataArray) -> xr.DataArray:
    """Ocurrencia histórica de agua (0 a 100 %) de JRC Global Surface Water.

    Porcentaje de las observaciones Landsat de 1984 a 2020 en que cada píxel
    de 30 m tuvo agua, en la grilla de ``como``. Verificado en Planetary
    Computer: colección ``jrc-gsw``, asset ``occurrence``.
    """
    import rioxarray  # noqa: F401

    aoi = cargar_aoi(aoi)
    items = buscar(aoi, "jrc-gsw", catalogo="planetary")
    if not items:
        raise LookupError("Sin teselas JRC Global Surface Water para %s" % aoi.nombre)
    ds = cargar(items, aoi, ["occurrence"], como=como, remuestreo="nearest", agrupar=None)
    capa = ds["occurrence"].astype("float32")
    capa = capa.where(capa <= 100).max("time").fillna(0.0)
    capa.name = "ocurrencia_agua"
    return capa.rio.write_crs(como.rio.crs)
