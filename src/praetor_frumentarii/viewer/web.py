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
}
COLOR_MODULO = {
    "terreno": "#38d39f", "cambios": "#ff7a45", "marino": "#3ea6ff", "suelo": "#e7b84b",
}


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
        max_zoom=19, max_native_zoom=16,
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
    ))
    mapa.save(str(destino))
    if temporales.exists():
        import shutil

        shutil.rmtree(temporales, ignore_errors=True)
    log.info("tablero -> %s", destino)
    return destino


def servir(base: Optional[Path] = None, puerto: int = 8765, abrir: bool = True) -> None:
    """Sirve ``salidas/`` en http://127.0.0.1:<puerto> y regenera el tablero en cada visita."""
    import functools
    import http.server
    import webbrowser

    base = Path(base) if base else carpeta_base()
    generar_tablero(base)

    class Manejador(http.server.SimpleHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?")[0] in ("/", "/tablero.html"):
                try:
                    generar_tablero(base)
                except Exception as e:
                    log.warning("no pude regenerar el tablero (%s)", e)
                self.path = "/tablero.html"
            super().do_GET()

        def log_message(self, *args: Any) -> None:
            pass

    servidor = http.server.ThreadingHTTPServer(
        ("127.0.0.1", puerto), functools.partial(Manejador, directory=str(base))
    )
    url = "http://127.0.0.1:%d/" % puerto
    log.info("tablero en %s  (Ctrl+C para detener)", url)
    if abrir:
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        log.info("servidor detenido")
    finally:
        servidor.server_close()


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
  <div class="pf-sello">USO CIVIL</div>
</div>
<div id="pf-izq" class="pf-lado">
  <div class="pf-tit"><span>Corridas registradas</span><span id="pf-todo" style="cursor:pointer;color:#39d0ff">VER TODO</span></div>
  <div id="pf-lista"></div>
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
    if(sat.checked){mapa.removeLayer(oscura); mapa.addLayer(satelite); satelite.bringToBack();}
    else{mapa.removeLayer(satelite); mapa.addLayer(oscura); oscura.bringToBack();}
  }
  try{sat.checked=localStorage.getItem("pf-sat")==="1";}catch(e){}
  sat.onchange=function(){try{localStorage.setItem("pf-sat",sat.checked?"1":"0");}catch(e){} base();};
  base();
  var auto=el("pf-auto"), temporizador=null;
  function programar(){ if(temporizador) clearTimeout(temporizador);
    if(auto.checked) temporizador=setTimeout(function(){location.reload();},30000); }
  try{auto.checked=localStorage.getItem("pf-auto")==="1";}catch(e){}
  auto.onchange=function(){try{localStorage.setItem("pf-auto",auto.checked?"1":"0");}catch(e){} programar();};
  programar();
  setTimeout(function(){mapa.invalidateSize(); mapa.fitBounds(LIM,{padding:[30,30]});
    if(D.length) elegir(0,D.length===1);
    else el("pf-det").innerHTML='<div class="pf-vacio">Selecciona una corrida para ver sus indicadores, avisos y capas.</div>';
  },150);
});
"""
