"""Flujos completos de cambios: quemas, inundación y cambio SAR."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from ..core import (
    AOI, Panel, Progreso, a_decibeles, cargar_aoi, cargar_dem, cargar_worldcover, como,
    componer_rgb, compuesto_s2, guardar_geojson, guardar_geotiff, guardar_resumen_png,
    obtener_logger, par_s1, registrar_corrida, resolucion, vectorizar,
)
from . import inundacion as _inu
from . import quemas as _quemas
from . import sar as _sar

log = obtener_logger("cambios")
MODULO = "cambios"

FUENTE_S2 = "Contiene datos Copernicus Sentinel-2 modificados (ESA), procesados por PRÆTOR Frumentarii"
FUENTE_S1 = ("Contiene datos Copernicus Sentinel-1 modificados (ESA); corrección radiométrica de "
             "terreno (RTC) de Microsoft Planetary Computer")
FUENTE_DEM = "Copernicus DEM GLO-30 (© DLR e.V. 2010-2014 y © Airbus Defence and Space GmbH 2014-2018)"
FUENTE_WC = "ESA WorldCover 10 m 2021 v200 (CC BY 4.0)"
PIE = ("AOI aproximada. Detección automática con falsos positivos posibles: validar en terreno. "
       "Contiene datos Copernicus Sentinel modificados.")


def _ha(mascara: xr.DataArray) -> float:
    return float(np.asarray(mascara.values).astype(bool).sum()) * resolucion(mascara) ** 2 / 1e4


def _mostrar(visor: Any, panel: Panel) -> None:
    if visor is not None:
        visor.mostrar(panel)


def ejecutar_quemas(
    aoi: Union[AOI, str], antes: str, despues: str, umbral: float = 0.27,
    nubes_max: float = 40.0, resolucion_m: float = 20.0, area_min_ha: float = 1.0,
    catalogo: str = "earth-search", visor: Any = None,
) -> Dict[str, Any]:
    """Área quemada entre dos períodos con dNBR de Sentinel-2."""
    aoi = cargar_aoi(aoi)
    p = Progreso(4, log)
    bandas = ("azul", "verde", "rojo", "nir", "swir2")
    p.paso("compuesto Sentinel-2 ANTES (%s)" % antes)
    ds_a = compuesto_s2(aoi, antes, bandas, catalogo, nubes_max, resolucion_m, visor=visor,
                        prefijo="Antes:")
    p.paso("compuesto Sentinel-2 DESPUÉS (%s)" % despues)
    ds_d = compuesto_s2(aoi, despues, bandas, catalogo, nubes_max, resolucion_m, visor=visor,
                        prefijo="Después:")
    p.paso("dNBR, severidad y polígonos")
    r = _quemas.area_quemada(ds_a, ds_d, umbral)
    quemado = _inu.limpiar(r["quemado"], 6)
    poligonos = vectorizar(quemado, area_min_ha, {"tipo": "area_quemada", "umbral_dnbr": umbral})
    panel_dnbr = Panel(r["dnbr"], "dNBR (antes menos después)", "RdYlGn_r", -0.5, 1.0, "dNBR")
    _mostrar(visor, panel_dnbr)

    p.paso("guardando resultados")
    archivos = [
        guardar_geotiff(r["dnbr"], MODULO, aoi, "dnbr"),
        guardar_geotiff(r["severidad"], MODULO, aoi, "severidad"),
        guardar_geojson(poligonos, MODULO, aoi, "area_quemada"),
    ]
    contorno = {"facecolor": "none", "edgecolor": "#00e5ff", "linewidth": 0.8}
    sev = r["severidad"].where(r["severidad"] != 255)
    # Falso color SWIR2-NIR-rojo: las quemas recientes se ven rojizas.
    falso = np.nan_to_num(np.dstack([
        np.clip(ds_d[b].values / q, 0, 1) for b, q in (("swir2", 0.35), ("nir", 0.45), ("rojo", 0.25))
    ]))
    ha = float(poligonos["area_ha"].sum()) if len(poligonos) else 0.0
    titulo = "Área quemada (dNBR) | %s | %.0f ha" % (aoi.nombre, ha)
    png = guardar_resumen_png(
        [Panel(componer_rgb(ds_a), "Antes: color real (mediana de %d fechas)"
               % len(str(ds_a.attrs.get("fechas_usadas", "x")).split(",")), referencia=ds_a["rojo"]),
         Panel(falso, "Después: falso color SWIR2-NIR-rojo (mediana de %d fechas)"
               % len(str(ds_d.attrs.get("fechas_usadas", "x")).split(",")), referencia=ds_d["rojo"],
               vectores=[(poligonos, contorno)]),
         panel_dnbr,
         Panel(sev, "Severidad (clases USGS)", clases=_quemas.CLASES_SEVERIDAD)],
        MODULO, aoi, titulo, nombre="resumen_quemas", pie=PIE,
    )
    validos = np.isfinite(r["dnbr"].values)
    indicadores = {
        "area_quemada_ha": ha, "poligonos": len(poligonos),
        "severidad_alta_ha": float((r["severidad"].values >= 3).sum() - (
            r["severidad"].values == 255).sum()) * resolucion_m ** 2 / 1e4,
        "dnbr_maximo": float(np.nanmax(r["dnbr"].values)) if validos.any() else None,
        "pixeles_validos_pct": round(100 * float(validos.mean()), 1),
    }
    avisos: List[str] = []
    if indicadores["pixeles_validos_pct"] < 70:
        avisos.append("Solo %.0f%% de píxeles válidos (nubes, humo o agua): el área puede estar "
                      "subestimada" % indicadores["pixeles_validos_pct"])
    registrar_corrida(
        MODULO, aoi, titulo, indicadores, archivos + [png], avisos,
        {"antes": antes, "despues": despues, "umbral_dnbr": umbral,
         "fechas_antes": ds_a.attrs.get("fechas_usadas"),
         "fechas_despues": ds_d.attrs.get("fechas_usadas")},
        [FUENTE_S2], analisis="quemas", png=png,
        capas_web=[
            {"archivo": "dnbr.tif", "titulo": "dNBR", "cmap": "RdYlGn_r", "vmin": -0.5,
             "vmax": 1.0, "visible": False},
            {"archivo": "severidad.tif", "titulo": "Severidad de quema", "visible": True,
             "clases": {"1": "#ffee58", "2": "#ffa726", "3": "#ef5350", "4": "#6a1b9a"}},
            {"archivo": "area_quemada.geojson", "titulo": "Polígonos de área quemada",
             "color": "#00e5ff", "relleno": 0.0, "visible": True},
        ],
    )
    p.fin("quemas listo")
    return {"antes": ds_a, "despues": ds_d, "resultado": r, "poligonos": poligonos, "png": png,
            "archivos": archivos, "indicadores": indicadores, "avisos": avisos}


def ejecutar_inundacion(
    aoi: Union[AOI, str], antes: str, despues: str, metodo: str = "sar",
    umbral_db: Optional[float] = None, nubes_max: float = 60.0, resolucion_m: float = 20.0,
    area_min_ha: float = 0.5, visor: Any = None,
) -> Dict[str, Any]:
    """Agua nueva entre dos períodos. ``metodo``: 'sar', 'ndwi' o 'ambos'."""
    if metodo not in ("sar", "ndwi", "ambos"):
        raise ValueError("metodo debe ser 'sar', 'ndwi' o 'ambos'")
    aoi = cargar_aoi(aoi)
    avisos: List[str] = []
    fuentes: List[str] = [FUENTE_DEM]
    paneles: List[Panel] = []
    p = Progreso(4, log)
    agua_a: Optional[xr.DataArray] = None
    agua_d: Optional[xr.DataArray] = None
    parametros: Dict[str, Any] = {"antes": antes, "despues": despues, "metodo": metodo}

    p.paso("agua por SAR (Sentinel-1)" if metodo != "ndwi" else "SAR: omitido")
    if metodo in ("sar", "ambos"):
        ds_a, ds_d = par_s1(aoi, antes, despues, ("vv",), resolucion_m, visor=visor)
        a_db = a_decibeles(_sar.filtro_moteado(ds_a["vv"].median("time").rio.write_crs(ds_a.rio.crs)))
        d_db = a_decibeles(_sar.filtro_moteado(ds_d["vv"].median("time").rio.write_crs(ds_a.rio.crs)))
        a_db, d_db = como(a_db.values, a_db, "vv_db"), como(d_db.values, a_db, "vv_db")
        # Un umbral por fecha: el agua agitada de una crecida es más brillante.
        u_a = _inu.umbral_otsu_db(a_db) if umbral_db is None else float(umbral_db)
        u_d = _inu.umbral_otsu_db(d_db) if umbral_db is None else float(umbral_db)
        log.info("  umbral agua/tierra: %.1f dB antes, %.1f dB después", u_a, u_d)
        agua_a, agua_d = _inu.agua_sar(a_db, u_a), _inu.agua_sar(d_db, u_d)
        parametros.update(umbral_db_antes=round(u_a, 2), umbral_db_despues=round(u_d, 2))
        if u_d > -14.0:
            avisos.append("Agua agitada después del evento (umbral %.1f dB): mayor riesgo de "
                          "falsos positivos en suelos lisos o húmedos" % u_d)
        fuentes.append(FUENTE_S1)
        paneles += [Panel(a_db, "Antes: Sentinel-1 VV", "gray", -25, 0, "dB"),
                    Panel(d_db, "Después: Sentinel-1 VV", "gray", -25, 0, "dB")]

    p.paso("agua por NDWI (Sentinel-2)" if metodo != "sar" else "NDWI: omitido")
    if metodo in ("ndwi", "ambos"):
        try:
            bandas = ("azul", "verde", "rojo", "nir")
            o_a = compuesto_s2(aoi, antes, bandas, nubes_max=nubes_max, resolucion=resolucion_m,
                               visor=visor, prefijo="Antes:")
            o_d = compuesto_s2(aoi, despues, bandas, nubes_max=nubes_max, resolucion=resolucion_m,
                               visor=visor, prefijo="Después:")
            n_a, n_d = _inu.agua_ndwi(o_a), _inu.agua_ndwi(o_d)
            fuentes.append(FUENTE_S2)
            if agua_d is None:
                agua_a, agua_d = n_a, n_d
                paneles += [Panel(componer_rgb(o_a), "Antes: Sentinel-2", referencia=o_a["rojo"]),
                            Panel(componer_rgb(o_d), "Después: Sentinel-2", referencia=o_d["rojo"])]
            else:
                # 'ambos': el agua previa es la unión (más conservador) y la posterior
                # se acepta si la ve el SAR o el óptico.
                n_a = n_a.interp_like(agua_a, method="nearest").fillna(0).astype(bool)
                n_d = n_d.interp_like(agua_d, method="nearest").fillna(0).astype(bool)
                agua_a = como(agua_a.values | n_a.values, agua_a, "agua")
                agua_d = como(agua_d.values | n_d.values, agua_d, "agua")
        except LookupError as e:
            if agua_d is None:
                raise
            avisos.append("NDWI no disponible (%s); se usa solo SAR" % e)
            log.warning(avisos[-1])
    assert agua_a is not None and agua_d is not None

    p.paso("agua nueva, filtro de pendiente y polígonos")
    from ..terreno.relieve import pendiente as _pendiente

    pend = None
    try:
        dem = cargar_dem(aoi, 30.0)
        pend = _pendiente(dem).interp_like(agua_d, method="nearest")
    except Exception as e:
        avisos.append("Sin DEM para filtrar laderas (%s)" % type(e).__name__)
    nueva = _inu.inundacion(agua_a, agua_d, pend)
    poligonos = vectorizar(nueva, area_min_ha, {"tipo": "inundacion", "metodo": metodo})
    ha = float(poligonos["area_ha"].sum()) if len(poligonos) else 0.0

    estado = np.zeros(nueva.shape, dtype="uint8")
    estado[agua_a.values.astype(bool) & agua_d.values.astype(bool)] = 1
    estado[nueva.values.astype(bool)] = 2
    estado[agua_a.values.astype(bool) & ~agua_d.values.astype(bool)] = 3
    estado_da = como(estado, nueva, "estado_agua")
    clases = {1: ("#1565c0", "Agua permanente"), 2: ("#ff1744", "Agua nueva (inundación)"),
              3: ("#ffb300", "Agua que retrocede")}
    panel_estado = Panel(estado_da.where(estado_da > 0), "Cambio de agua (%s)" % metodo,
                         clases=clases, fondo=paneles[-1].datos if paneles and getattr(
                             paneles[-1].datos, "ndim", 2) == 3 else None,
                         referencia=estado_da)
    _mostrar(visor, panel_estado)
    paneles.append(panel_estado)

    p.paso("guardando resultados")
    archivos = [guardar_geotiff(estado_da, MODULO, aoi, "estado_agua"),
                guardar_geojson(poligonos, MODULO, aoi, "inundacion")]
    titulo = "Inundación (%s) | %s | %.0f ha de agua nueva" % (metodo, aoi.nombre, ha)
    png = guardar_resumen_png(paneles, MODULO, aoi, titulo, nombre="resumen_inundacion", pie=PIE)
    indicadores = {"agua_nueva_ha": ha, "poligonos": len(poligonos),
                   "agua_permanente_ha": float((estado == 1).sum()) * resolucion_m ** 2 / 1e4,
                   "agua_que_retrocede_ha": float((estado == 3).sum()) * resolucion_m ** 2 / 1e4}
    registrar_corrida(
        MODULO, aoi, titulo, indicadores, archivos + [png], avisos, parametros, fuentes,
        analisis="inundacion", png=png,
        capas_web=[
            {"archivo": "estado_agua.tif", "titulo": "Agua permanente, nueva y en retroceso",
             "visible": True, "clases": {"1": "#1565c0", "2": "#ff1744", "3": "#ffb300"}},
            {"archivo": "inundacion.geojson", "titulo": "Polígonos de agua nueva",
             "color": "#ff1744", "relleno": 0.15, "visible": False},
        ],
    )
    p.fin("inundación listo")
    return {"estado": estado_da, "inundacion": nueva, "poligonos": poligonos, "png": png,
            "archivos": archivos, "indicadores": indicadores, "avisos": avisos}


def ejecutar_sar(
    aoi: Union[AOI, str], antes: str, despues: str, umbral_db: float = 3.0,
    resolucion_m: float = 20.0, area_min_ha: float = 0.5, visor: Any = None,
) -> Dict[str, Any]:
    """Cambio de retrodispersión y proxy de coherencia (deslizamientos, daño urbano)."""
    aoi = cargar_aoi(aoi)
    avisos: List[str] = []
    p = Progreso(4, log)
    p.paso("Sentinel-1 antes y después (misma órbita)")
    ds_a, ds_d = par_s1(aoi, antes, despues, ("vv",), resolucion_m, visor=visor)
    p.paso("log-ratio y proxy de coherencia")
    r = _sar.cambio_sar(ds_a, ds_d)
    r["cambio"] = _sar.clasificar_cambio(r["log_ratio"], umbral_db)
    _mostrar(visor, Panel(r["log_ratio"], "Cambio de retrodispersión", "RdBu", -6, 6, "dB"))
    _mostrar(visor, Panel(r["coherencia_proxy"], "Proxy de coherencia", "viridis", 0, 1))

    p.paso("zonas construidas con indicio de daño (ESA WorldCover)")
    dano = None
    try:
        cobertura = cargar_worldcover(aoi, r["log_ratio"])
        dano = _sar.posible_dano_urbano(r["coherencia_proxy"], r["log_ratio"], cobertura == 50)
    except Exception as e:
        avisos.append("Sin cobertura WorldCover: no se evalúa daño urbano (%s)" % type(e).__name__)
        log.warning(avisos[-1])
    fuerte = _inu.limpiar(como(r["cambio"].values != 0, r["cambio"], "cambio_fuerte"), 8)
    poligonos = vectorizar(fuerte, area_min_ha, {"tipo": "cambio_sar", "umbral_db": umbral_db})
    pol_dano = vectorizar(dano, 0.2, {"tipo": "posible_dano_urbano"}) if dano is not None else None

    p.paso("guardando resultados")
    archivos = [guardar_geotiff(r["log_ratio"], MODULO, aoi, "sar_log_ratio"),
                guardar_geotiff(r["coherencia_proxy"], MODULO, aoi, "sar_coherencia_proxy"),
                guardar_geojson(poligonos, MODULO, aoi, "sar_cambio")]
    if pol_dano is not None:
        archivos.append(guardar_geojson(pol_dano, MODULO, aoi, "sar_posible_dano_urbano"))
    titulo = "Cambio SAR | %s | %.0f ha con cambio fuerte" % (
        aoi.nombre, float(poligonos["area_ha"].sum()) if len(poligonos) else 0.0)
    png = guardar_resumen_png(
        [Panel(r["antes_db"], "Antes: Sentinel-1 VV", "gray", -25, 0, "dB"),
         Panel(r["despues_db"], "Después: Sentinel-1 VV", "gray", -25, 0, "dB"),
         Panel(r["log_ratio"], "Cambio (después menos antes)", "RdBu", -6, 6, "dB",
               vectores=[(pol_dano, {"facecolor": "none", "edgecolor": "#ffea00",
                                     "linewidth": 1.0})] if pol_dano is not None else []),
         Panel(r["coherencia_proxy"], "Proxy de coherencia (correlación local)", "viridis", 0, 1)],
        MODULO, aoi, titulo, nombre="resumen_sar",
        pie=PIE + " La coherencia es un proxy por correlación local, no interferometría.",
    )
    indicadores = {
        "cambio_fuerte_ha": float(poligonos["area_ha"].sum()) if len(poligonos) else 0.0,
        "baja_de_senal_ha": float((r["cambio"].values == -1).sum()) * resolucion_m ** 2 / 1e4,
        "alza_de_senal_ha": float((r["cambio"].values == 1).sum()) * resolucion_m ** 2 / 1e4,
        "coherencia_proxy_media": float(np.nanmean(r["coherencia_proxy"].values)),
        "posible_dano_urbano_ha": float(pol_dano["area_ha"].sum()) if pol_dano is not None
        and len(pol_dano) else 0.0,
    }
    avisos.append("La coherencia es un proxy por correlación de amplitud, no coherencia "
                  "interferométrica: úsese solo como indicio")
    capas_web = [
        {"archivo": "sar_log_ratio.tif", "titulo": "Cambio de retrodispersión (dB)",
         "cmap": "RdBu", "vmin": -6.0, "vmax": 6.0, "visible": True},
        {"archivo": "sar_coherencia_proxy.tif", "titulo": "Proxy de coherencia",
         "cmap": "viridis", "vmin": 0.0, "vmax": 1.0, "visible": False},
        {"archivo": "sar_cambio.geojson", "titulo": "Polígonos de cambio fuerte",
         "color": "#ffffff", "relleno": 0.0, "visible": False},
        {"archivo": "sar_posible_dano_urbano.geojson", "titulo": "Posible daño en zona construida",
         "color": "#ffea00", "relleno": 0.3, "visible": True},
    ]
    registrar_corrida(MODULO, aoi, titulo, indicadores, archivos + [png], avisos,
                      {"antes": antes, "despues": despues, "umbral_db": umbral_db},
                      [FUENTE_S1, FUENTE_WC], analisis="sar", png=png, capas_web=capas_web)
    p.fin("cambio SAR listo")
    return {"resultado": r, "poligonos": poligonos, "posible_dano": pol_dano, "png": png,
            "archivos": archivos, "indicadores": indicadores, "avisos": avisos}
