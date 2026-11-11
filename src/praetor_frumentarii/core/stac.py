"""Búsqueda STAC en Element84 Earth Search y Microsoft Planetary Computer."""
from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional, Sequence, Union

from .aoi import AOI, cargar_aoi
from .registro import obtener_logger

log = obtener_logger("core.stac")

CATALOGOS: Dict[str, str] = {
    "earth-search": "https://earth-search.aws.element84.com/v1",
    "planetary": "https://planetarycomputer.microsoft.com/api/stac/v1",
}

Fechas = Union[str, Sequence[str]]


def hay_red(timeout: float = 4.0) -> bool:
    """True si se puede abrir una conexión HTTPS a alguno de los catálogos."""
    for anfitrion in ("planetarycomputer.microsoft.com", "earth-search.aws.element84.com"):
        try:
            with socket.create_connection((anfitrion, 443), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def normalizar_fechas(fechas: Fechas) -> str:
    """Acepta 'AAAA-MM-DD/AAAA-MM-DD', una fecha sola o un par (inicio, fin)."""
    if isinstance(fechas, str):
        return fechas
    fechas = list(fechas)
    if len(fechas) != 2:
        raise ValueError("fechas debe ser un texto o un par (inicio, fin)")
    return "%s/%s" % (fechas[0], fechas[1])


def abrir_catalogo(catalogo: str = "earth-search"):
    """Abre el cliente STAC. Planetary Computer firma las URLs automáticamente."""
    import pystac_client

    if catalogo not in CATALOGOS:
        raise ValueError("Catálogo desconocido '%s'. Opciones: %s" % (catalogo, list(CATALOGOS)))
    if catalogo == "planetary":
        import planetary_computer

        return pystac_client.Client.open(
            CATALOGOS[catalogo], modifier=planetary_computer.sign_inplace
        )
    return pystac_client.Client.open(CATALOGOS[catalogo])


def buscar(
    aoi: Union[AOI, str],
    coleccion: str,
    fechas: Optional[Fechas] = None,
    catalogo: str = "earth-search",
    nubes_max: Optional[float] = None,
    limite: Optional[int] = None,
    consulta: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Busca ítems STAC que intersectan la AOI, ordenados por fecha.

    ``nubes_max`` filtra por ``eo:cloud_cover`` (porcentaje). ``consulta`` permite
    filtros extra con la sintaxis ``query`` de STAC.
    """
    aoi = cargar_aoi(aoi)
    cliente = abrir_catalogo(catalogo)
    filtro: Dict[str, Any] = dict(consulta or {})
    if nubes_max is not None:
        filtro["eo:cloud_cover"] = {"lt": float(nubes_max)}
    log.info(
        "buscando %s en %s | AOI %s | fechas %s%s",
        coleccion,
        catalogo,
        aoi.nombre,
        normalizar_fechas(fechas) if fechas else "todas",
        " | nubes < %s%%" % nubes_max if nubes_max is not None else "",
    )
    busqueda = cliente.search(
        collections=[coleccion],
        intersects=aoi.geometria.__geo_interface__,
        datetime=normalizar_fechas(fechas) if fechas else None,
        query=filtro or None,
        max_items=limite,
    )
    items = list(busqueda.items())
    items.sort(key=lambda it: it.datetime or it.common_metadata.start_datetime)
    log.info("  %d escena(s) encontrada(s)", len(items))
    for it in items[:8]:
        nubes = it.properties.get("eo:cloud_cover")
        log.info(
            "    %s  %s%s",
            (it.datetime or it.common_metadata.start_datetime).strftime("%Y-%m-%d %H:%M"),
            it.id,
            "" if nubes is None else "  nubes %.0f%%" % nubes,
        )
    if len(items) > 8:
        log.info("    ... y %d más", len(items) - 8)
    return items
