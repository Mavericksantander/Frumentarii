"""Tablero web local de operaciones civiles, generado con folium.

Lee todos los ``salidas/<modulo>/<aoi>/corrida.json`` y arma una sola página
``salidas/tablero.html`` con mapa oscuro, lista de corridas, indicadores,
avisos y capas activables. Solo muestra lo que los módulos produjeron de
verdad: no inventa datos.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import NOMBRE, __version__
from ..core import obtener_logger
from ..core.salidas import carpeta_base

log = obtener_logger("viewer.web")

ETIQUETA_MODULO = {
    "terreno": "TERRENO", "cambios": "CAMBIOS", "marino": "MARINO", "suelo": "SUELO",
    "inteligencia": "INTELIGENCIA",
}
COLOR_MODULO = {
    "terreno": "#38d39f", "cambios": "#ff7a45", "marino": "#3ea6ff", "suelo": "#e7b84b",
    "inteligencia": "#c58bff",
}
COLOR_NIVEL = {"alta": "#ff1744", "media": "#ff9100", "baja": "#ffd54f"}


def leer_corridas(base: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Todas las corridas registradas, la más reciente primero."""
    base = Path(base) if base else carpeta_base()
    corridas = []
    for ruta in sorted(base.glob("*/*/corrida*.json")):
        try:
            with open(ruta, encoding="utf-8") as f:
                datos = json.load(f)
        except (OSError, ValueError) as e:
            log.warning("no pude leer %s (%s)", ruta, e)
            continue
        datos["carpeta"] = ruta.parent
        corridas.append(datos)
    corridas.sort(key=lambda c: c.get("generado", ""), reverse=True)
    return corridas


def raster_a_png(
    tif: Path, destino: Path, cmap: str = "viridis", vmin: Optional[float] = None,
    vmax: Optional[float] = None, clases: Optional[Dict[str, str]] = None, lado_max: int = 1400,
) -> List[List[float]]:
    """Colorea un GeoTIFF y lo guarda como PNG transparente en Web Mercator.

    Devuelve los límites [[sur, oeste], [norte, este]] para Leaflet.
    """
    import matplotlib
    from matplotlib.colors import to_rgba
    from PIL import Image
    from pyproj import Transformer

    from ..core.raster import leer_geotiff, reproyectar

    da = leer_geotiff(tif, enmascarar=True).isel(band=0, drop=True)
    discreto = clases is not None or not np.issubdtype(da.dtype, np.floating)
    en_3857 = reproyectar(da, "EPSG:3857", "nearest" if discreto else "bilinear")
    paso = max(1, int(np.ceil(max(en_3857.shape) / float(lado_max))))
    en_3857 = en_3857[::paso, ::paso]
    v = np.asarray(en_3857.values, dtype="float64")
    v[~np.isfinite(v)] = np.nan
    rgba = np.zeros(v.shape + (4,), dtype="float32")
    if clases is not None:
        for valor, color in clases.items():
            rgba[v == float(valor)] = to_rgba(color)
    else:
        finitos = v[np.isfinite(v)]
        if vmin is None:
            vmin = float(np.percentile(finitos, 2)) if finitos.size else 0.0
        if vmax is None:
            vmax = float(np.percentile(finitos, 98)) if finitos.size else 1.0
        norm = np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1)
        rgba = matplotlib.colormaps[cmap](np.nan_to_num(norm)).astype("float32")
        rgba[..., 3] = np.where(np.isfinite(v), 1.0, 0.0)
        if not np.issubdtype(da.dtype, np.floating):
            rgba[..., 3] = np.where(v > 0, 1.0, 0.0)  # máscaras: el 0 es transparente
    destino.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgba * 255).astype("uint8"), "RGBA").save(destino, optimize=True)
    oeste, sur, este, norte = en_3857.rio.bounds()
    t = Transformer.from_crs(3857, 4326, always_xy=True)
    lon0, lat0 = t.transform(oeste, sur)
    lon1, lat1 = t.transform(este, norte)
    return [[lat0, lon0], [lat1, lon1]]


def _capas_por_defecto(corrida: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Si un módulo no declaró estilo, se muestran sus GeoJSON con color del módulo."""
    return [
        {"archivo": c, "titulo": Path(c).stem.replace("_", " "), "visible": True,
         "color": COLOR_MODULO.get(corrida.get("modulo", ""), "#ffffff")}
        for c in corrida.get("capas", []) if c.endswith(".geojson")
    ]


def _alertas(base: Path, mapa: Any, maximo: int = 60) -> Tuple[List[Dict[str, Any]], str]:
    """Lee la evaluación de inteligencia y agrega su capa al mapa.

    Devuelve la lista de alertas para la página y el nombre JavaScript de la capa
    ('' si todavía no hay evaluación).
    """
    import folium

    ruta = base / "inteligencia" / "general" / "hallazgos.geojson"
    if not ruta.exists():
        return [], ""
    try:
        import geopandas as gpd

        from ..inteligencia.exposicion import describir
        from ..inteligencia.hallazgos import ETIQUETAS

        h = gpd.read_file(ruta)
    except Exception as e:
        log.warning("no pude leer la evaluación de inteligencia (%s)", e)
        return [], ""
    if len(h) == 0:
        return [], ""
    h = h.sort_values("prioridad", ascending=False).head(maximo).reset_index(drop=True)
    lista = []
    for _, f in h.iterrows():
        oeste, sur, este, norte = f.geometry.bounds
        evaluada = bool(f.get("exposicion_evaluada", False))
        lista.append({
            "id": f["id"], "tipo": f["tipo"], "etiqueta": ETIQUETAS.get(f["tipo"], f["tipo"]),
            "categoria": f["categoria"], "aoi": f["aoi"],
            "fecha": str(f["fecha"]).replace("T", " ").replace(" 00:00:00", "")[:16],
            "magnitud": float(f["magnitud"]), "unidad": f["unidad"], "nivel": f["nivel"],
            "confianza": float(f["confianza"]), "severidad": float(f["severidad"]),
            "prioridad": float(f["prioridad"]), "fuentes": f["fuentes"],
            "evidencia": [t for t in str(f["evidencia"] or "").split(" | ") if t],
            "expuesto": (describir(f) + (" (%s)" % f["nombres"] if f.get("nombres") else ""))
            if evaluada else "No evaluado (OpenStreetMap no respondió o quedó fuera del límite "
                             "de consultas)",
            "reservas": str(f.get("reservas") or ""), "verificar": str(f.get("verificar") or ""),
            "motivo": str(f.get("motivo") or ""),
            "bbox": [[sur - 0.01, oeste - 0.01], [norte + 0.01, este + 0.01]],
        })
    dibujo = h[["id", "tipo", "categoria", "nivel", "prioridad", "geometry"]].copy()
    dibujo["geometry"] = dibujo.geometry.simplify(0.0003)
    grupo = folium.FeatureGroup(name="Alertas de inteligencia", show=True)
    folium.GeoJson(
        json.loads(dibujo.to_json()),
        style_function=lambda f: {
            "color": COLOR_NIVEL.get(f["properties"]["nivel"], "#8a98a6")
            if f["properties"]["categoria"] == "alerta" else "#8a98a6",
            "weight": 2, "fillOpacity": 0.25,
        },
        tooltip=folium.GeoJsonTooltip(fields=["id", "tipo", "nivel", "prioridad"]),
    ).add_to(grupo)
    grupo.add_to(mapa)
    return lista, grupo.get_name()


def generar_tablero(base: Optional[Path] = None, destino: Optional[Path] = None) -> Path:
    """Genera ``tablero.html`` (autocontenido) y devuelve su ruta."""
    import folium

    base = Path(base) if base else carpeta_base()
    base.mkdir(parents=True, exist_ok=True)
    destino = Path(destino) if destino else base / "tablero.html"
    corridas = leer_corridas(base)
    log.info("tablero: %d corrida(s) en %s", len(corridas), base)

    cajas = [c["bbox"] for c in corridas if c.get("bbox")]
    if cajas:
        limites = [[min(b[1] for b in cajas), min(b[0] for b in cajas)],
                   [max(b[3] for b in cajas), max(b[2] for b in cajas)]]
    else:
        limites = [[-56.0, -76.0], [-17.5, -66.0]]  # Chile continental

    mapa = folium.Map(tiles=None, control_scale=True, zoom_control=True, prefer_canvas=True)
    # Mapas base de Esri: funcionan también al abrir el HTML como archivo local.
    # No se usa OpenStreetMap directo porque su servidor bloquea las teselas cuando
    # la página no envía origen (file://), ni CARTO porque ahora exige clave de API.
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/%s/MapServer/tile/{z}/{y}/{x}"
    base_oscura = folium.TileLayer(
        esri % "Canvas/World_Dark_Gray_Base", name="Base oscura",
        attr="Teselas &copy; Esri, HERE, Garmin, &copy; colaboradores de OpenStreetMap | "
             "Contiene datos Copernicus modificados",
        max_zoom=19, max_native_zoom=11,
    )
    base_satelite = folium.TileLayer(
        esri % "World_Imagery", name="Satélite",
        attr="Teselas &copy; Esri, Maxar, Earthstar Geographics | Contiene datos Copernicus "
             "modificados",
        max_zoom=19, max_native_zoom=18, show=False,
    )
    base_oscura.add_to(mapa)
    base_satelite.add_to(mapa)
    mapa.fit_bounds(limites)

    datos_js: List[Dict[str, Any]] = []
    temporales = base / ".tablero_tmp"
    for i, c in enumerate(corridas):
        modulo, aoi = c.get("modulo", "?"), c.get("aoi", "?")
        color = COLOR_MODULO.get(modulo, "#cccccc")
        carpeta: Path = c["carpeta"]
        capas_js = []
        oeste, sur, este, norte = c["bbox"]
        marco = folium.FeatureGroup(name="AOI %s/%s" % (modulo, aoi))
        folium.Rectangle(
            [[sur, oeste], [norte, este]], color=color, weight=1.5, fill=False, dash_array="6 6",
            tooltip="%s | %s%s" % (ETIQUETA_MODULO.get(modulo, modulo.upper()), aoi,
                                   " (AOI aproximada)" if c.get("aoi_aproximada") else ""),
        ).add_to(marco)
        marco.add_to(mapa)
        for j, capa in enumerate(c.get("capas_web") or _capas_por_defecto(c)):
            archivo = carpeta / capa["archivo"]
            if not archivo.exists():
                continue
            grupo = folium.FeatureGroup(name="%s/%s %s" % (modulo, aoi, capa["titulo"]),
                                        show=bool(capa.get("visible", True)))
            try:
                if archivo.suffix == ".tif":
                    png = temporales / ("%d_%d.png" % (i, j))
                    lim = raster_a_png(archivo, png, capa.get("cmap", "viridis"),
                                       capa.get("vmin"), capa.get("vmax"), capa.get("clases"))
                    folium.raster_layers.ImageOverlay(
                        str(png), bounds=lim, opacity=float(capa.get("opacidad", 0.8)),
                    ).add_to(grupo)
                else:
                    if archivo.stat().st_size > 4_000_000:
                        log.info("  %s es muy pesado para el tablero; se omite", archivo.name)
                        continue
                    with open(archivo, encoding="utf-8") as f:
                        gj = json.load(f)
                    if not gj.get("features"):
                        continue
                    col = capa.get("color", color)
                    folium.GeoJson(
                        gj,
                        marker=folium.CircleMarker(radius=4, weight=1, fill=True,
                                                   fill_opacity=0.85) if capa.get("punto") else None,
                        style_function=lambda _f, col=col, capa=capa: {
                            "color": col, "weight": capa.get("grosor", 1.5),
                            "fillColor": col, "fillOpacity": capa.get("relleno", 0.25),
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=[k for k in list(gj["features"][0]["properties"])[:4]]
                        ) if gj["features"][0].get("properties") else None,
                    ).add_to(grupo)
            except Exception as e:
                log.warning("  no pude dibujar %s (%s)", archivo.name, e)
                continue
            grupo.add_to(mapa)
            capas_js.append({"titulo": capa["titulo"], "js": grupo.get_name(),
                             "visible": bool(capa.get("visible", True)),
                             "tipo": "raster" if archivo.suffix == ".tif" else "vector"})
        nombre_png = c.get("resumen") or "resumen.png"
        png_resumen = carpeta / nombre_png
        datos_js.append({
            "modulo": modulo,
            "etiqueta": ETIQUETA_MODULO.get(modulo, modulo.upper())
            + (" / " + c["analisis"].upper() if c.get("analisis") else ""),
            "color": color, "aoi": aoi, "descripcion": c.get("aoi_descripcion", ""),
            "aproximada": bool(c.get("aoi_aproximada")), "titulo": c.get("titulo", ""),
            "generado": c.get("generado", ""), "indicadores": c.get("indicadores", {}),
            "avisos": c.get("avisos", []), "fuentes": c.get("fuentes", []),
            "parametros": c.get("parametros", {}),
            "bbox": [[sur, oeste], [norte, este]], "capas": capas_js,
            "resumen": "%s/%s/%s" % (modulo, aoi, nombre_png) if png_resumen.exists() else None,
            "carpeta": "%s/%s/" % (modulo, aoi),
        })

    alertas_js, capa_alertas = _alertas(base, mapa)
    from ..core import listar_aois

    raiz = mapa.get_root()
    def crudo(texto: str) -> "folium.Element":
        # folium interpreta el texto como plantilla Jinja: se marca como literal.
        return folium.Element("{% raw %}" + texto.replace("{% endraw %}", "") + "{% endraw %}")

    raiz.header.add_child(crudo("<title>%s | Tablero</title>" % NOMBRE))
    raiz.header.add_child(crudo("<style>%s</style>" % _CSS))
    raiz.html.add_child(crudo(_HTML.replace("__NOMBRE__", NOMBRE.upper())
                              .replace("__VERSION__", __version__)))
    raiz.script.add_child(crudo(
        _JS.replace("__MAPA__", mapa.get_name())
        .replace("__BASE_OSCURA__", base_oscura.get_name())
        .replace("__BASE_SATELITE__", base_satelite.get_name())
        .replace("__DATOS__", json.dumps(datos_js, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__LIMITES__", json.dumps(limites))
        .replace("__ALERTAS__", json.dumps(alertas_js, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__CAPA_ALERTAS__", capa_alertas)
        .replace("__AOIS__", json.dumps(listar_aois()))
    ))
    mapa.save(str(destino))
    if temporales.exists():
        import shutil

        shutil.rmtree(temporales, ignore_errors=True)
    log.info("tablero -> %s", destino)
    return destino


def servir(base: Optional[Path] = None, puerto: int = 8765, abrir: bool = True) -> None:
    """Sirve el tablero con consola en vivo (ver ``viewer.servidor``)."""
    from .servidor import servir as _servir

    _servir(base, puerto, abrir)


def mapa_web(modulo: Optional[str] = None, aoi: Optional[str] = None) -> Path:
    """Compatibilidad con la API propuesta: genera el tablero y devuelve su ruta."""
    return generar_tablero()


_CSS = r"""
:root{--fondo:#070b10;--panel:#0c131b;--panel2:#101a24;--linea:#1c2a38;--texto:#c9d6e2;
--tenue:#6b7f92;--acento:#39d0ff;--ambar:#ffb020;--rojo:#ff5263;--verde:#38d39f;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
html,body{height:100%;margin:0;background:var(--fondo);color:var(--texto);
font:13px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif;overflow:hidden}
.folium-map{position:absolute!important;top:52px!important;left:310px!important;
right:370px!important;bottom:30px!important;width:auto!important;height:auto!important;
background:#05080c}
#pf-top{position:absolute;top:0;left:0;right:0;height:52px;display:flex;align-items:center;
gap:18px;padding:0 16px;background:linear-gradient(#0d151e,#0a1118);
border-bottom:1px solid var(--linea);box-sizing:border-box;z-index:1000}
#pf-top .marca{font:700 15px var(--mono);letter-spacing:.22em;color:#eaf4fb;white-space:nowrap}
#pf-top .marca b{color:var(--acento);font-weight:700}
#pf-top .sub{font:11px var(--mono);letter-spacing:.14em;color:var(--tenue);white-space:nowrap}
#pf-top .hueco{flex:1}
.pf-dato{font:11px var(--mono);color:var(--tenue);text-align:right;white-space:nowrap}
.pf-dato strong{display:block;font-size:14px;color:var(--texto);font-weight:600}
.pf-sello{font:700 10px var(--mono);letter-spacing:.16em;padding:4px 9px;border-radius:2px;
border:1px solid var(--verde);color:var(--verde);white-space:nowrap}
.pf-lado{position:absolute;top:52px;bottom:30px;background:var(--panel);overflow-y:auto;
box-sizing:border-box;z-index:900}
#pf-izq{left:0;width:310px;border-right:1px solid var(--linea)}
#pf-der{right:0;width:370px;border-left:1px solid var(--linea)}
.pf-tit{font:700 10px var(--mono);letter-spacing:.12em;color:var(--tenue);padding:12px 14px 6px;gap:8px;
text-transform:uppercase;display:flex;justify-content:space-between}
.pf-corrida{margin:6px 10px;padding:10px 11px;background:var(--panel2);
border:1px solid var(--linea);border-left:3px solid var(--c);border-radius:3px;cursor:pointer}
.pf-corrida:hover{border-color:#2c4257;border-left-color:var(--c)}
.pf-corrida.activa{outline:1px solid var(--acento);background:#12202d}
.pf-corrida .mod{font:700 10px var(--mono);letter-spacing:.16em;color:var(--c)}
.pf-corrida .aoi{font-weight:600;color:#eaf4fb;margin:2px 0}
.pf-corrida .meta{font:11px var(--mono);color:var(--tenue)}
.pf-punto{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;
background:var(--verde);box-shadow:0 0 6px var(--verde)}
.pf-punto.av{background:var(--ambar);box-shadow:0 0 6px var(--ambar)}
#pf-tabs{display:flex;border-bottom:1px solid var(--linea)}
#pf-tabs span{flex:1;text-align:center;padding:11px 0 9px;font:700 10px var(--mono);
letter-spacing:.14em;color:var(--tenue);cursor:pointer;border-bottom:2px solid transparent}
#pf-tabs span.on{color:var(--acento);border-bottom-color:var(--acento)}
.pf-alerta{display:flex;gap:10px;margin:6px 10px;padding:9px 10px;background:var(--panel2);
border:1px solid var(--linea);border-left:3px solid var(--c);border-radius:3px;cursor:pointer}
.pf-alerta:hover{border-color:#2c4257;border-left-color:var(--c)}
.pf-alerta.activa{outline:1px solid var(--acento);background:#12202d}
.pf-alerta .pri{font:700 17px var(--mono);color:var(--c);min-width:42px;text-align:right}
.pf-alerta .cuerpo{min-width:0}
.pf-alerta .mod{font:700 10px var(--mono);letter-spacing:.08em;color:var(--c);text-transform:uppercase}
.pf-alerta .aoi{font-weight:600;color:#eaf4fb}
.pf-alerta .meta{font:11px var(--mono);color:var(--tenue)}
.pf-campo{padding:5px 14px}
.pf-campo label{display:block;font:10px var(--mono);letter-spacing:.08em;color:var(--tenue);
text-transform:uppercase;margin:0 0 3px;font-weight:400}
.pf-campo select,.pf-campo input{width:100%;box-sizing:border-box;background:#0a1118;
color:var(--texto);border:1px solid var(--linea);border-radius:3px;padding:6px 7px;
font:12px var(--mono)}
.pf-boton{margin:8px 14px 4px;padding:9px;text-align:center;font:700 11px var(--mono);
letter-spacing:.14em;border:1px solid var(--acento);color:var(--acento);border-radius:3px;
cursor:pointer;background:transparent;display:block;width:calc(100% - 28px)}
.pf-boton:hover{background:#0f2836}
.pf-boton.sec{border-color:var(--linea);color:var(--tenue)}
#pf-log{margin:6px 14px 14px;padding:8px;background:#05080c;border:1px solid var(--linea);
border-radius:3px;font:10.5px/1.5 var(--mono);color:#9fb3c8;height:240px;overflow:auto;
white-space:pre-wrap;word-break:break-word}
#pf-msg{margin:4px 14px;color:#ffd58a;font-size:12px;min-height:16px}
#pf-estado{margin:8px 14px 0;font:11px var(--mono);color:var(--texto)}
.pf-texto{margin:2px 14px 6px;color:var(--texto);font-size:12px}
#pf-det ul.pf-ev{color:var(--texto);font-size:12px}
.pf-vacio{margin:14px;padding:14px;border:1px dashed var(--linea);color:var(--tenue);
border-radius:3px;font-size:12px}
.pf-vacio code{display:block;margin-top:8px;color:var(--acento);font:11px var(--mono);
white-space:pre-wrap;word-break:break-all}
#pf-det h2{font-size:15px;margin:0;padding:2px 14px 0;color:#eaf4fb;font-weight:600}
#pf-det .desc{padding:2px 14px 0;color:var(--tenue);font-size:12px}
.pf-ind{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:4px 14px}
.pf-ind div{background:var(--panel2);border:1px solid var(--linea);border-radius:3px;
padding:8px 9px;min-width:0}
.pf-ind span{display:block;font:10px var(--mono);letter-spacing:.08em;color:var(--tenue);
text-transform:uppercase;overflow:hidden;text-overflow:ellipsis}
.pf-ind strong{font:600 17px var(--mono);color:#eaf4fb;word-break:break-word}
.pf-aviso{margin:4px 14px;padding:7px 9px;border-left:3px solid var(--ambar);
background:#1d1808;color:#ffd58a;font-size:12px;border-radius:2px}
.pf-capa{display:flex;align-items:center;gap:8px;padding:5px 14px;font-size:12px;color:var(--texto);margin:0;font-weight:400}
.pf-capa input{accent-color:var(--acento)}
.pf-capa .tipo{margin-left:auto;font:10px var(--mono);color:var(--tenue)}
#pf-det img{display:block;width:calc(100% - 28px);margin:6px 14px;border:1px solid var(--linea);
border-radius:3px;background:#fff}
#pf-det ul{margin:2px 14px 6px 30px;padding:0;color:var(--tenue);font-size:11px}
#pf-det a{color:var(--acento)}
.pf-nota{margin:10px 14px 16px;padding:9px;border:1px solid var(--linea);border-radius:3px;
color:var(--tenue);font-size:11px}
#pf-pie{position:absolute;left:0;right:0;bottom:0;height:30px;display:flex;align-items:center;
gap:14px;padding:0 14px;background:#0a1118;border-top:1px solid var(--linea);
font:11px var(--mono);color:var(--tenue);box-sizing:border-box;z-index:1000;white-space:nowrap;
overflow:hidden}
#pf-pie .cinta{flex:1;overflow:hidden;text-overflow:ellipsis;color:#ffd58a}
#pf-pie label{cursor:pointer;margin:0;font-weight:400}
.leaflet-container{font:12px var(--mono)}
.leaflet-control-attribution{background:rgba(7,11,16,.8)!important;color:var(--tenue)!important}
.leaflet-control-attribution a{color:var(--acento)!important}
.leaflet-bar a{background:#0c131b!important;color:var(--texto)!important;
border-color:var(--linea)!important}
@media (max-width:1100px){#pf-der{width:300px}.folium-map{right:300px!important;left:250px!important}
#pf-izq{width:250px}#pf-top .sub{display:none}}
@media (max-width:760px){html,body{overflow:auto}.pf-lado{position:static;width:auto!important;
border:0}.folium-map{position:relative!important;top:auto!important;left:auto!important;
right:auto!important;bottom:auto!important;height:55vh!important;margin-top:52px}#pf-pie{position:static}}
"""

_HTML = r"""
<div id="pf-top">
  <div class="marca">__NOMBRE__ <b>//</b></div>
  <div class="sub">TABLERO DE OPERACIONES CIVILES &middot; OBSERVACI&Oacute;N DE LA TIERRA &middot; CHILE</div>
  <div class="hueco"></div>
  <div class="pf-dato">CORRIDAS<strong id="pf-n">0</strong></div>
  <div class="pf-dato">AVISOS<strong id="pf-na">0</strong></div>
  <div class="pf-dato">UTC<strong id="pf-utc">--:--:--</strong></div>
  <div class="pf-dato">CHILE<strong id="pf-cl">--:--:--</strong></div>
  <div class="pf-sello" id="pf-sello-vivo" style="border-color:#6b7f92;color:#6b7f92">ARCHIVO</div>
  <div class="pf-sello">USO CIVIL</div>
</div>
<div id="pf-izq" class="pf-lado">
  <div id="pf-tabs"><span id="pf-t-alertas">ALERTAS</span><span id="pf-t-corridas">CORRIDAS</span><span id="pf-t-observar">OBSERVAR</span></div>
  <div id="pf-p-alertas">
    <div class="pf-tit"><span>Alertas priorizadas</span><span id="pf-todo2" style="cursor:pointer;color:#39d0ff">VER TODO</span></div>
    <div id="pf-alertas"></div>
  </div>
  <div id="pf-p-corridas" style="display:none">
    <div class="pf-tit"><span>Corridas registradas</span><span id="pf-todo" style="cursor:pointer;color:#39d0ff">VER TODO</span></div>
    <div id="pf-lista"></div>
  </div>
  <div id="pf-p-observar" style="display:none">
    <div class="pf-tit"><span>Nueva observaci&oacute;n</span><span><span class="pf-punto" id="pf-vivo"></span></span></div>
    <div id="pf-sinserv" class="pf-vacio">Este archivo solo muestra resultados ya calculados. Para lanzar observaciones y verlas en vivo, abre el tablero con el servidor local:<code>frumentarii viewer servir</code></div>
    <div id="pf-form" style="display:none">
      <div class="pf-campo"><label>An&aacute;lisis</label><select id="pf-an">
        <option value="focos">Focos de calor activos (FIRMS)</option>
        <option value="quemas">&Aacute;rea quemada (dNBR)</option>
        <option value="inundacion">Inundaci&oacute;n</option>
        <option value="sar">Cambio radar (SAR)</option>
        <option value="marino">Manchas en el mar</option>
        <option value="suelo">Alteraci&oacute;n mineral</option>
        <option value="terreno">Costo de desplazamiento</option>
        <option value="inteligencia">Reevaluar inteligencia</option></select></div>
      <div class="pf-campo" id="pf-f-aoi"><label>Zona</label><select id="pf-i-aoi"></select>
        <button class="pf-boton sec" id="pf-dibujar" style="margin:6px 0 0;width:100%">DIBUJAR ZONA EN EL MAPA</button>
        <div id="pf-zona-txt" style="font:11px var(--mono);color:var(--tenue);margin-top:4px"></div></div>
      <div class="pf-campo" id="pf-f-nombre" style="display:none"><label>Nombre de la zona</label><input id="pf-i-nombre" placeholder="mi_zona" maxlength="40"></div>
      <div class="pf-campo" id="pf-f-periodo"><label>Ventana</label><select id="pf-i-periodo"><option value="24h">&Uacute;ltimas 24 h</option><option value="48h">&Uacute;ltimas 48 h</option><option value="7d">&Uacute;ltimos 7 d&iacute;as</option></select></div>
      <div class="pf-campo" id="pf-f-antes"><label>Antes (AAAA-MM-DD/AAAA-MM-DD)</label><input id="pf-i-antes"></div>
      <div class="pf-campo" id="pf-f-despues"><label>Despu&eacute;s (AAAA-MM-DD/AAAA-MM-DD)</label><input id="pf-i-despues"></div>
      <div class="pf-campo" id="pf-f-fechas"><label>Fechas (AAAA-MM-DD/AAAA-MM-DD)</label><input id="pf-i-fechas"></div>
      <div class="pf-campo" id="pf-f-metodo"><label>M&eacute;todo</label><select id="pf-i-metodo"><option value="sar">Radar (funciona con nubes)</option><option value="ndwi">&Oacute;ptico NDWI</option><option value="ambos">Ambos</option></select></div>
      <div class="pf-campo" id="pf-f-sensor"><label>Sensor</label><select id="pf-i-sensor"><option value="s2">Sentinel-2</option><option value="landsat">Landsat 8/9</option><option value="aster">ASTER (2000 a 2008)</option></select></div>
      <div class="pf-campo" id="pf-f-modo"><label>Modo</label><select id="pf-i-modo"><option value="pie">A pie</option><option value="4x4">4x4</option><option value="camion">Cami&oacute;n</option></select></div>
      <button class="pf-boton" id="pf-lanzar">LANZAR OBSERVACI&Oacute;N</button>
      <button class="pf-boton sec" id="pf-detener">DETENER</button>
      <div id="pf-msg"></div>
    </div>
    <div id="pf-estado"></div>
    <div id="pf-log"></div>
  </div>
</div>
<div id="pf-der" class="pf-lado"><div id="pf-det"></div></div>
<div id="pf-pie">
  <span>v__VERSION__</span>
  <span class="cinta" id="pf-cinta"></span>
  <label><input type="checkbox" id="pf-sat"> base satelital</label>
  <label><input type="checkbox" id="pf-auto"> actualizar cada 30 s</label>
  <span>10 a 30 m &middot; validar en terreno</span>
</div>
"""

_JS = r"""
window.addEventListener("load", function(){
  var mapa = window["__MAPA__"], D = __DATOS__, LIM = __LIMITES__, activa = -1;
  function el(id){return document.getElementById(id);}
  function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,function(c){
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c];});}
  function fecha(s){return s ? s.replace("T"," ").slice(0,16) : "";}
  function valor(v){
    if(v===null||v===undefined) return "s/d";
    if(typeof v==="number") return (Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=10?v.toFixed(1):
      String(+v.toFixed(2))).replace(".",",");
    return esc(v);
  }
  function reloj(){
    var a=new Date();
    el("pf-utc").textContent=a.toISOString().slice(11,19);
    try{el("pf-cl").textContent=a.toLocaleTimeString("es-CL",{timeZone:"America/Santiago",hour12:false});}
    catch(e){el("pf-cl").textContent="";}
  }
  setInterval(reloj,1000); reloj();
  var avisos=[]; D.forEach(function(c){c.avisos.forEach(function(a){
    avisos.push(c.etiqueta+"/"+c.aoi+": "+a);});});
  el("pf-n").textContent=D.length; el("pf-na").textContent=avisos.length;
  el("pf-cinta").textContent=avisos.length?("AVISOS  >  "+avisos.join("   |   ")):"";
  var lista=el("pf-lista");
  if(!D.length){
    lista.innerHTML='<div class="pf-vacio">Todav&iacute;a no hay corridas. Ejecuta un m&oacute;dulo y vuelve a cargar:'+
      '<code>frumentarii terreno costo --aoi chiloe</code></div>';
  }
  D.forEach(function(c,i){
    var d=document.createElement("div"); d.className="pf-corrida"; d.style.setProperty("--c",c.color);
    d.innerHTML='<div class="mod"><span class="pf-punto'+(c.avisos.length?" av":"")+'"></span>'+esc(c.etiqueta)+
      '</div><div class="aoi">'+esc(c.aoi)+(c.aproximada?' <span style="color:#6b7f92;font-weight:400">(aprox.)</span>':'')+
      '</div><div class="meta">'+esc(fecha(c.generado))+' &middot; '+c.capas.length+' capas'+
      (c.avisos.length?' &middot; '+c.avisos.length+' avisos':'')+'</div>';
    d.onclick=function(){elegir(i,true);}; lista.appendChild(d); c._el=d;
  });
  function capa(c,j){return window[c.capas[j].js];}
  function elegir(i,volar){
    activa=i; var c=D[i];
    D.forEach(function(x){x._el.classList.toggle("activa",x===c);});
    // Solo se ven las capas de la corrida elegida: varias corridas comparten zona.
    D.forEach(function(x){x.capas.forEach(function(k,j){
      var L=capa(x,j), ver=(x===c)&&k.on;
      if(ver&&!mapa.hasLayer(L)) mapa.addLayer(L);
      if(!ver&&mapa.hasLayer(L)) mapa.removeLayer(L);});});
    if(volar) mapa.flyToBounds(c.bbox,{padding:[30,30],duration:0.8});
    var h='<div class="pf-tit"><span style="color:'+c.color+'">'+esc(c.etiqueta)+'</span><span>'+esc(fecha(c.generado))+'</span></div>'+
      '<h2>'+esc(c.titulo)+'</h2><div class="desc">'+esc(c.descripcion)+
      (c.aproximada?' <b style="color:#ffb020">AOI aproximada.</b>':'')+'</div>';
    var ks=Object.keys(c.indicadores);
    if(ks.length){h+='<div class="pf-tit"><span>Indicadores</span></div><div class="pf-ind">';
      ks.forEach(function(k){h+='<div><span>'+esc(k.replace(/_/g," "))+'</span><strong>'+valor(c.indicadores[k])+'</strong></div>';});
      h+='</div>';}
    if(c.avisos.length){h+='<div class="pf-tit"><span>Avisos</span></div>';
      c.avisos.forEach(function(a){h+='<div class="pf-aviso">'+esc(a)+'</div>';});}
    if(c.capas.length){h+='<div class="pf-tit"><span>Capas en el mapa</span></div>';
      c.capas.forEach(function(k,j){h+='<label class="pf-capa"><input type="checkbox" data-j="'+j+'"'+
        (mapa.hasLayer(capa(c,j))?' checked':'')+'>'+esc(k.titulo)+'<span class="tipo">'+k.tipo+'</span></label>';});
      h+='<div class="pf-capa"><span>Opacidad r&aacute;ster</span><input type="range" id="pf-op" min="0" max="100" value="80" style="flex:1"></div>';}
    if(c.resumen){h+='<div class="pf-tit"><span>Resumen de la corrida</span><a href="'+esc(c.resumen)+'" target="_blank">ABRIR PNG</a></div>'+
      '<a href="'+esc(c.resumen)+'" target="_blank"><img src="'+esc(c.resumen)+'?t='+encodeURIComponent(c.generado)+'" alt="Resumen"></a>';}
    if(c.fuentes.length){h+='<div class="pf-tit"><span>Fuentes y licencias</span></div><ul>';
      c.fuentes.forEach(function(f){h+='<li>'+esc(f)+'</li>';}); h+='</ul>';}
    h+='<div class="pf-nota">Resultado preliminar a partir de im&aacute;genes p&uacute;blicas de 10 a 30 m. Puede contener falsos positivos: '+
       'validar en terreno antes de tomar decisiones. Archivos en <code>salidas/'+esc(c.carpeta)+'</code></div>';
    el("pf-det").innerHTML=h;
    Array.prototype.forEach.call(el("pf-det").querySelectorAll("input[data-j]"),function(x){
      x.onchange=function(){var j=+x.getAttribute("data-j"), L=capa(c,j); c.capas[j].on=x.checked;
        if(x.checked) mapa.addLayer(L); else mapa.removeLayer(L);};});
    var op=el("pf-op"); if(op) op.oninput=function(){c.capas.forEach(function(k,j){
      if(k.tipo==="raster") capa(c,j).eachLayer(function(l){if(l.setOpacity) l.setOpacity(op.value/100);});});};
  }
  D.forEach(function(x){x.capas.forEach(function(k){k.on=k.visible;});});
  el("pf-todo").onclick=function(){
    // Vista general: vuelven las capas por defecto de todas las corridas.
    D.forEach(function(x){x._el.classList.remove("activa"); x.capas.forEach(function(k,j){
      var L=capa(x,j); if(k.on&&!mapa.hasLayer(L)) mapa.addLayer(L);});});
    mapa.flyToBounds(LIM,{padding:[30,30],duration:0.8});};
  var sat=el("pf-sat"), oscura=window["__BASE_OSCURA__"], satelite=window["__BASE_SATELITE__"];
  function base(){
    // La base gris de Esri no tiene teselas de detalle fuera de las ciudades: al
    // acercar (zoom 12 o más) se pasa sola a la imagen satelital.
    var usarSat = sat.checked || mapa.getZoom() >= 12;
    if(usarSat){ if(mapa.hasLayer(oscura)) mapa.removeLayer(oscura);
      if(!mapa.hasLayer(satelite)){mapa.addLayer(satelite); satelite.bringToBack();} }
    else{ if(mapa.hasLayer(satelite)) mapa.removeLayer(satelite);
      if(!mapa.hasLayer(oscura)){mapa.addLayer(oscura); oscura.bringToBack();} }
  }
  mapa.on("zoomend", base);
  try{sat.checked=localStorage.getItem("pf-sat")==="1";}catch(e){}
  sat.onchange=function(){try{localStorage.setItem("pf-sat",sat.checked?"1":"0");}catch(e){} base();};
  base();
  var auto=el("pf-auto"), temporizador=null;
  function programar(){ if(temporizador) clearTimeout(temporizador);
    if(auto.checked) temporizador=setTimeout(function(){location.reload();},30000); }
  try{auto.checked=localStorage.getItem("pf-auto")==="1";}catch(e){}
  auto.onchange=function(){try{localStorage.setItem("pf-auto",auto.checked?"1":"0");}catch(e){} programar();};
  programar();
  // ---------- Alertas de inteligencia ----------
  var A = __ALERTAS__, capaAl = "__CAPA_ALERTAS__" ? window["__CAPA_ALERTAS__"] : null;
  var NIV = {alta:"#ff1744", media:"#ff9100", baja:"#ffd54f"};
  function colorAl(a){return a.categoria==="alerta" ? (NIV[a.nivel]||"#8a98a6") : "#8a98a6";}
  function ocultarCorridas(){
    activa=-1;
    D.forEach(function(x){x._el.classList.remove("activa"); x.capas.forEach(function(k,j){
      var L=capa(x,j); if(mapa.hasLayer(L)) mapa.removeLayer(L);});});
  }
  var la=el("pf-alertas");
  if(!A.length){
    la.innerHTML='<div class="pf-vacio">Todav&iacute;a no hay evaluaci&oacute;n. Ejecuta primero alg&uacute;n an&aacute;lisis y luego:<code>frumentarii inteligencia evaluar</code></div>';
  }
  A.forEach(function(a,i){
    var d=document.createElement("div"); d.className="pf-alerta"; d.style.setProperty("--c",colorAl(a));
    d.innerHTML='<div class="pri">'+valor(a.prioridad)+'</div><div class="cuerpo"><div class="mod">'+esc(a.etiqueta)+
      '</div><div class="aoi">'+esc(a.aoi)+'</div><div class="meta">'+esc(a.fecha)+' &middot; '+valor(a.magnitud)+' '+esc(a.unidad)+
      ' &middot; conf. '+esc(a.nivel)+'</div></div>';
    d.onclick=function(){verAlerta(i);}; la.appendChild(d); a._el=d;
  });
  function seccion(t){return '<div class="pf-tit"><span>'+t+'</span></div>';}
  function verAlerta(i){
    var a=A[i]; A.forEach(function(x){x._el.classList.toggle("activa",x===a);});
    mapa.flyToBounds(a.bbox,{padding:[60,60],maxZoom:13,duration:0.8});
    var h='<div class="pf-tit"><span style="color:'+colorAl(a)+'">'+(a.categoria==="alerta"?"ALERTA ":"CONTEXTO ")+esc(a.id)+
      '</span><span>'+esc(a.fecha)+'</span></div><h2>'+esc(a.etiqueta)+' en '+esc(a.aoi)+'</h2>'+
      '<div class="desc">Fuente: '+esc(a.fuentes)+'</div><div class="pf-ind">'+
      '<div><span>prioridad</span><strong>'+valor(a.prioridad)+'</strong></div>'+
      '<div><span>confianza</span><strong style="color:'+colorAl(a)+'">'+esc(a.nivel)+' '+valor(a.confianza)+'</strong></div>'+
      '<div><span>magnitud</span><strong>'+valor(a.magnitud)+' '+esc(a.unidad)+'</strong></div>'+
      '<div><span>severidad</span><strong>'+valor(a.severidad)+'</strong></div></div>';
    h+=seccion("Por qu&eacute; esa confianza");
    if(a.evidencia.length){h+='<ul class="pf-ev">'; a.evidencia.forEach(function(t){h+='<li>'+esc(t)+'</li>';}); h+='</ul>';}
    else h+='<div class="pf-texto">Una sola fuente, sin corroboraci&oacute;n disponible.</div>';
    h+=seccion("Infraestructura civil expuesta")+'<div class="pf-texto">'+esc(a.expuesto)+'</div>';
    if(a.reservas) h+=seccion("Reservas de la corrida")+'<div class="pf-aviso">'+esc(a.reservas)+'</div>';
    h+=seccion("Qu&eacute; verificar")+'<div class="pf-texto">'+esc(a.verificar)+'</div>';
    h+=seccion("C&aacute;lculo de la prioridad")+'<div class="pf-texto" style="color:#6b7f92">'+esc(a.motivo)+'</div>';
    h+='<div class="pf-nota">La prioridad ordena qu&eacute; revisar primero; no confirma da&ntilde;os. Informe completo: '+
       '<a href="inteligencia/general/parte.md" target="_blank">parte.md</a></div>';
    el("pf-det").innerHTML=h;
  }
  function resumenAlertas(){
    var n=A.filter(function(a){return a.categoria==="alerta";});
    var c={alta:0,media:0,baja:0}; n.forEach(function(a){c[a.nivel]=(c[a.nivel]||0)+1;});
    el("pf-det").innerHTML='<div class="pf-tit"><span style="color:#c58bff">EVALUACI&Oacute;N DE SITUACI&Oacute;N</span></div>'+
      '<h2>'+n.length+' alertas en pantalla</h2><div class="desc">Hallazgos de todos los m&oacute;dulos, corroborados entre fuentes y ordenados por prioridad.</div>'+
      '<div class="pf-ind"><div><span>confianza alta</span><strong style="color:#ff1744">'+c.alta+'</strong></div>'+
      '<div><span>confianza media</span><strong style="color:#ff9100">'+c.media+'</strong></div>'+
      '<div><span>confianza baja</span><strong style="color:#ffd54f">'+c.baja+'</strong></div>'+
      '<div><span>contexto</span><strong>'+(A.length-n.length)+'</strong></div></div>'+
      '<div class="pf-nota">Selecciona una alerta para ver por qu&eacute; tiene esa confianza, qu&eacute; infraestructura civil hay cerca y qu&eacute; conviene verificar. '+
      (A.length?'Informe completo: <a href="inteligencia/general/parte.md" target="_blank">parte.md</a>':'')+'</div>';
  }
  el("pf-todo2").onclick=function(){A.forEach(function(x){x._el.classList.remove("activa");}); resumenAlertas();
    mapa.flyToBounds(LIM,{padding:[30,30],duration:0.8});};

  // ---------- Pestañas ----------
  var TABS=["alertas","corridas","observar"];
  function pestana(t){
    if(TABS.indexOf(t)<0) t="alertas";
    TABS.forEach(function(n){el("pf-p-"+n).style.display=(n===t?"block":"none"); el("pf-t-"+n).classList.toggle("on",n===t);});
    try{localStorage.setItem("pf-tab",t);}catch(e){}
    if(t==="corridas"){
      if(capaAl&&mapa.hasLayer(capaAl)) mapa.removeLayer(capaAl);
      if(D.length&&activa<0) elegir(0,false);
      if(!D.length) el("pf-det").innerHTML='<div class="pf-vacio">Todav&iacute;a no hay corridas.</div>';
    }else{
      ocultarCorridas(); if(capaAl&&!mapa.hasLayer(capaAl)) mapa.addLayer(capaAl);
      if(t==="alertas") resumenAlertas();
    }
  }
  TABS.forEach(function(n){el("pf-t-"+n).onclick=function(){pestana(n);};});

  // ---------- Consola en vivo ----------
  var AOIS=__AOIS__, CAMPOS={focos:["periodo"],quemas:["antes","despues"],inundacion:["antes","despues","metodo"],
    sar:["antes","despues"],marino:["fechas"],suelo:["sensor"],terreno:["modo"],inteligencia:[]};
  var PRE={focos:{aoi:"chile_continental"},
    quemas:{aoi:"valparaiso_vina",antes:"2024-01-05/2024-02-01",despues:"2024-02-04/2024-03-05"},
    inundacion:{aoi:"biobio",antes:"2023-05-20/2023-06-18",despues:"2023-06-22/2023-07-03"},
    sar:{aoi:"biobio",antes:"2023-05-20/2023-06-18",despues:"2023-06-22/2023-07-03"},
    marino:{aoi:"magallanes",fechas:"2024-01-01/2024-02-29"},suelo:{aoi:"atacama"},terreno:{aoi:"chiloe"}};
  var selAoi=el("pf-i-aoi");
  AOIS.forEach(function(n){var o=document.createElement("option"); o.value=n; o.textContent=n; selAoi.appendChild(o);});
  var oz=document.createElement("option"); oz.value="__zona__"; oz.textContent="(zona dibujada en el mapa)"; selAoi.appendChild(oz);
  function msg(t){el("pf-msg").textContent=t||"";}
  function formulario(pre){
    var an=el("pf-an").value;
    ["periodo","antes","despues","metodo","fechas","sensor","modo"].forEach(function(c){
      el("pf-f-"+c).style.display=(CAMPOS[an].indexOf(c)>=0?"block":"none");});
    el("pf-f-aoi").style.display=(an==="inteligencia"?"none":"block");
    if(pre){var p=PRE[an]||{}; Object.keys(p).forEach(function(k){var x=el("pf-i-"+k); if(x) x.value=p[k];});}
    el("pf-f-nombre").style.display=(an!=="inteligencia"&&selAoi.value==="__zona__"?"block":"none");
  }
  el("pf-an").onchange=function(){formulario(true);}; selAoi.onchange=function(){formulario(false);};
  formulario(true);
  var zona=null, rect=null, dibujando=false, esquina=null;
  el("pf-dibujar").onclick=function(){dibujando=true; esquina=null; mapa.getContainer().style.cursor="crosshair";
    el("pf-zona-txt").textContent="Haz clic en dos esquinas opuestas de la zona, sobre el mapa";};
  mapa.on("click",function(e){
    if(!dibujando) return;
    if(!esquina){esquina=e.latlng; el("pf-zona-txt").textContent="Ahora la esquina opuesta"; return;}
    var b=L.latLngBounds(esquina,e.latlng); if(rect) mapa.removeLayer(rect);
    rect=L.rectangle(b,{color:"#39d0ff",weight:2,dashArray:"5 5",fill:false}).addTo(mapa);
    zona=[b.getWest(),b.getSouth(),b.getEast(),b.getNorth()]; dibujando=false; mapa.getContainer().style.cursor="";
    el("pf-zona-txt").textContent="Zona: "+zona.map(function(v){return v.toFixed(3);}).join(", ");
    selAoi.value="__zona__"; formulario(false);
  });
  function api(ruta,cuerpo){
    return fetch(ruta,{method:cuerpo?"POST":"GET",cache:"no-store",
      headers:cuerpo?{"Content-Type":"application/json","X-Frumentarii":"1"}:{},
      body:cuerpo?JSON.stringify(cuerpo):undefined}).then(function(r){
        return r.json().then(function(j){j._ok=r.ok; return j;});});
  }
  el("pf-lanzar").onclick=function(){
    var an=el("pf-an").value, p={analisis:an};
    CAMPOS[an].forEach(function(c){p[c]=el("pf-i-"+c).value;});
    if(an!=="inteligencia"){
      if(selAoi.value==="__zona__"){
        if(!zona){msg("Dibuja primero la zona en el mapa"); return;}
        p.zona={nombre:el("pf-i-nombre").value,bbox:zona};
      }else p.aoi=selAoi.value;
    }
    msg("Enviando...");
    api("api/lanzar",p).then(function(j){
      if(!j._ok){msg(j.error||"El servidor rechazó el pedido"); return;}
      msg("Iniciado: "+j.iniciado); sondear();
    }).catch(function(){msg("El servidor local no responde");});
  };
  el("pf-detener").onclick=function(){api("api/detener",{}).then(function(j){msg(j.detenido?"Deteniendo...":"No hay nada en curso");});};
  var versionVista=null, sondeo=null;
  function enVivo(si){
    el("pf-sinserv").style.display=(si?"none":"block"); el("pf-form").style.display=(si?"block":"none");
    var s=el("pf-sello-vivo"); s.textContent=(si?"EN VIVO":"ARCHIVO");
    s.style.borderColor=s.style.color=(si?"#39d0ff":"#6b7f92");
  }
  function sondear(){
    clearTimeout(sondeo);
    api("api/estado").then(function(j){
      enVivo(true);
      if(j.registro.length){el("pf-log").textContent=j.registro.join("\n"); el("pf-log").scrollTop=1e9;}
      el("pf-estado").textContent=j.activo?("EN CURSO: "+j.descripcion+" (desde "+j.inicio+")"):
        (j.codigo===null?"Sin trabajos en curso":("Último: "+j.descripcion+(j.codigo===0?" (correcto)":" (con error)")));
      el("pf-vivo").className="pf-punto"+(j.activo?" av":"");
      if(versionVista===null) versionVista=j.version;
      else if(j.version!==versionVista&&!j.activo){
        // Hay resultados nuevos: se recarga para traer el mapa y las alertas actualizados.
        try{sessionStorage.setItem("pf-log",el("pf-log").textContent);}catch(e){}
        location.reload(); return;
      }
      sondeo=setTimeout(sondear,j.activo?1500:6000);
    }).catch(function(){enVivo(false);});
  }
  try{var previo=sessionStorage.getItem("pf-log"); if(previo) el("pf-log").textContent=previo;}catch(e){}
  sondear();

  setTimeout(function(){mapa.invalidateSize(); mapa.fitBounds(LIM,{padding:[30,30]});
    var t0="alertas"; try{t0=localStorage.getItem("pf-tab")||"alertas";}catch(e){}
    pestana(t0);
  },150);
});
"""
