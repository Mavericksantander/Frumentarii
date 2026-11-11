"""Servidor local del tablero con consola en vivo.

Además de servir ``salidas/``, expone una API mínima para lanzar observaciones
desde el navegador y seguir su registro mientras corren:

- ``GET  /api/estado``   estado del trabajo en curso y últimas líneas del registro
- ``POST /api/lanzar``   inicia un análisis (uno a la vez)
- ``POST /api/detener``  detiene el trabajo en curso

Seguridad: escucha solo en 127.0.0.1, exige una cabecera propia y un ``Host``
local (para que ninguna otra página web pueda lanzar trabajos), valida cada
parámetro contra listas cerradas y ejecuta el comando sin intérprete de shell.
"""
from __future__ import annotations

import collections
import datetime as dt
import functools
import http.server
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..core import listar_aois, obtener_logger
from ..core.salidas import carpeta_base

log = obtener_logger("viewer.servidor")

RANGO = re.compile(r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$")
NOMBRE_ZONA = re.compile(r"^[a-z0-9_]{1,40}$")
# análisis -> (argumentos fijos, campos requeridos, área máxima en km2 para una zona dibujada)
ANALISIS: Dict[str, Tuple[List[str], Tuple[str, ...], float]] = {
    "focos": (["cambios", "focos"], ("periodo",), 5e6),
    "quemas": (["cambios", "quemas"], ("antes", "despues"), 3000.0),
    "inundacion": (["cambios", "inundacion"], ("antes", "despues", "metodo"), 3000.0),
    "sar": (["cambios", "sar"], ("antes", "despues"), 3000.0),
    "marino": (["marino", "manchas"], ("fechas",), 6000.0),
    "suelo": (["suelo", "alteracion"], ("sensor",), 3000.0),
    "terreno": (["terreno", "costo"], ("modo",), 3000.0),
    "inteligencia": (["inteligencia", "evaluar"], (), 5e6),
}
OPCIONES = {
    "periodo": ("24h", "48h", "7d"), "metodo": ("sar", "ndwi", "ambos"),
    "sensor": ("s2", "landsat", "aster"), "modo": ("pie", "4x4", "camion"),
}
BANDERA = {"periodo": "--periodo", "metodo": "--metodo", "sensor": "--sensor", "modo": "--modo",
           "antes": "--antes", "despues": "--despues", "fechas": "--fechas"}


def _area_km2(bbox: List[float]) -> float:
    import math

    oeste, sur, este, norte = bbox
    return (este - oeste) * 111.32 * math.cos(math.radians((sur + norte) / 2)) * (norte - sur) * 110.57


def construir_comando(pedido: Dict[str, Any], base: Path) -> Tuple[List[str], str]:
    """Valida un pedido del navegador y devuelve (argumentos, descripción).

    Lanza ``ValueError`` con un mensaje para la persona si algo no es válido.
    Nunca copia texto libre a la línea de comandos: cada valor se compara con
    una lista cerrada o con un patrón estricto.
    """
    analisis = pedido.get("analisis")
    if analisis not in ANALISIS:
        raise ValueError("Análisis desconocido")
    fijos, requeridos, area_max = ANALISIS[analisis]
    args = list(fijos)
    descripcion = analisis

    if analisis != "inteligencia":
        zona = pedido.get("zona")
        if zona:
            nombre = str(zona.get("nombre", "")).strip().lower()
            if not NOMBRE_ZONA.match(nombre):
                raise ValueError("El nombre de la zona solo admite minúsculas, números y guion "
                                 "bajo (máximo 40)")
            if nombre in listar_aois():
                raise ValueError("Ese nombre ya lo usa una AOI de ejemplo; elige otro")
            try:
                bbox = [float(v) for v in zona.get("bbox", [])]
            except (TypeError, ValueError):
                raise ValueError("La zona dibujada no es válida")
            if len(bbox) != 4 or not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
                raise ValueError("La zona dibujada no es válida")
            area = _area_km2(bbox)
            if area > area_max:
                raise ValueError("La zona mide %.0f km², más que el máximo de %.0f km² para este "
                                 "análisis: dibuja una más pequeña" % (area, area_max))
            if area < 1.0:
                raise ValueError("La zona es demasiado pequeña (menos de 1 km²)")
            carpeta = Path(base) / "zonas"
            carpeta.mkdir(parents=True, exist_ok=True)
            oeste, sur, este, norte = bbox
            ruta = carpeta / (nombre + ".geojson")
            ruta.write_text(json.dumps({"type": "FeatureCollection", "features": [{
                "type": "Feature",
                "properties": {"nombre": nombre, "descripcion": "Zona dibujada en el tablero",
                               "aproximada": False},
                "geometry": {"type": "Polygon", "coordinates": [[
                    [oeste, sur], [este, sur], [este, norte], [oeste, norte], [oeste, sur]]]},
            }]}, ensure_ascii=False), encoding="utf-8")
            args += ["--aoi", str(ruta)]
            descripcion += " en " + nombre
        else:
            aoi = pedido.get("aoi")
            if aoi not in listar_aois():
                raise ValueError("Elige una AOI de la lista o dibuja una zona en el mapa")
            args += ["--aoi", aoi]
            descripcion += " en " + aoi

    for campo in requeridos:
        valor = str(pedido.get(campo, "")).strip()
        if campo in OPCIONES:
            if valor not in OPCIONES[campo]:
                raise ValueError("Valor no válido para %s" % campo)
        elif not RANGO.match(valor):
            raise ValueError("'%s' debe tener la forma AAAA-MM-DD/AAAA-MM-DD" % campo)
        else:
            try:
                inicio, fin = [dt.date.fromisoformat(p) for p in valor.split("/")]
            except ValueError:
                raise ValueError("'%s' contiene una fecha que no existe" % campo)
            if fin < inicio:
                raise ValueError("En '%s' la fecha final es anterior a la inicial" % campo)
        args += [BANDERA[campo], valor]
    return args, descripcion


class Trabajo:
    """Un análisis en ejecución (o el último terminado) y su registro."""

    def __init__(self, base: Path) -> None:
        self.base = Path(base)
        self.candado = threading.Lock()
        self.lineas: Deque[str] = collections.deque(maxlen=400)
        self.proceso: Optional[subprocess.Popen] = None
        self.descripcion = ""
        self.inicio = ""
        self.codigo: Optional[int] = None
        self.activo = False
        self.version = 0          # sube cuando hay resultados nuevos que mostrar

    def estado(self) -> Dict[str, Any]:
        with self.candado:
            return {"activo": self.activo, "descripcion": self.descripcion, "inicio": self.inicio,
                    "codigo": self.codigo, "version": self.version, "registro": list(self.lineas)}

    def _ejecutar(self, args: List[str]) -> int:
        entorno = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
                       FRUMENTARII_SALIDAS=str(self.base))
        comando = [sys.executable, "-W", "ignore", "-m", "praetor_frumentarii.cli"] + args
        with self.candado:
            self.lineas.append("$ frumentarii " + " ".join(args))
        proceso = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=entorno,
                                   cwd=str(self.base.parent))
        with self.candado:
            self.proceso = proceso
        assert proceso.stdout is not None
        for linea in proceso.stdout:
            linea = linea.rstrip()
            if linea and "Warning" not in linea and not linea.startswith("  warn"):
                with self.candado:
                    self.lineas.append(linea)
        return proceso.wait()

    def _correr(self, args: List[str], encadenar: bool) -> None:
        from .web import generar_tablero

        codigo = -1
        try:
            codigo = self._ejecutar(args)
            if codigo == 0 and encadenar and args[0] != "inteligencia":
                with self.candado:
                    self.lineas.append("--- actualizando la evaluación de inteligencia ---")
                self._ejecutar(["inteligencia", "evaluar"])
            generar_tablero(self.base)
        except Exception as e:  # el hilo nunca debe morir en silencio
            with self.candado:
                self.lineas.append("ERROR del servidor: %s: %s" % (type(e).__name__, e))
        finally:
            with self.candado:
                self.codigo = codigo
                self.activo = False
                self.proceso = None
                self.version += 1
                self.lineas.append("--- terminado %s ---" % (
                    "correctamente" if codigo == 0 else "con error (código %s)" % codigo))

    def lanzar(self, args: List[str], descripcion: str, encadenar: bool = True) -> bool:
        with self.candado:
            if self.activo:
                return False
            self.activo, self.codigo = True, None
            self.descripcion = descripcion
            self.inicio = dt.datetime.now().strftime("%H:%M:%S")
            self.lineas.clear()
        threading.Thread(target=self._correr, args=(args, encadenar), daemon=True).start()
        return True

    def detener(self) -> bool:
        with self.candado:
            proceso = self.proceso
        if proceso is None:
            return False
        proceso.terminate()
        return True


def crear_servidor(base: Optional[Path] = None, puerto: int = 8765):
    """Crea (sin iniciar) el servidor HTTP del tablero. Devuelve (servidor, trabajo)."""
    from .web import generar_tablero

    base = Path(base) if base else carpeta_base()
    base.mkdir(parents=True, exist_ok=True)
    trabajo = Trabajo(base)
    anfitriones = {"127.0.0.1", "localhost"}

    class Manejador(http.server.SimpleHTTPRequestHandler):
        def _json(self, codigo: int, cuerpo: Dict[str, Any]) -> None:
            datos = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(datos)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(datos)

        def _local(self) -> bool:
            return self.headers.get("Host", "").split(":")[0] in anfitriones

        def do_GET(self) -> None:  # noqa: N802
            ruta = self.path.split("?")[0]
            if ruta == "/api/estado":
                if not self._local():
                    return self._json(403, {"error": "solo local"})
                cuerpo = trabajo.estado()
                cuerpo["aois"] = listar_aois()
                return self._json(200, cuerpo)
            if ruta in ("/", "/tablero.html"):
                if not trabajo.activo:
                    try:
                        generar_tablero(base)
                    except Exception as e:
                        log.warning("no pude regenerar el tablero (%s)", e)
                self.path = "/tablero.html"
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            ruta = self.path.split("?")[0]
            # La cabecera propia obliga al navegador a una verificación previa que este
            # servidor no responde: otra página web no puede enviar este pedido.
            if not self._local() or self.headers.get("X-Frumentarii") != "1":
                return self._json(403, {"error": "pedido no autorizado"})
            if ruta == "/api/detener":
                return self._json(200, {"detenido": trabajo.detener()})
            if ruta != "/api/lanzar":
                return self._json(404, {"error": "no existe"})
            try:
                largo = int(self.headers.get("Content-Length", "0"))
                if largo > 10000:
                    raise ValueError("pedido demasiado grande")
                pedido = json.loads(self.rfile.read(largo).decode("utf-8"))
                args, descripcion = construir_comando(pedido, base)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            if not trabajo.lanzar(args, descripcion):
                return self._json(409, {"error": "Ya hay un análisis en curso; espera a que termine"})
            return self._json(200, {"iniciado": descripcion})

        def end_headers(self) -> None:
            if self.path.endswith(".html"):
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    servidor = http.server.ThreadingHTTPServer(
        ("127.0.0.1", puerto), functools.partial(Manejador, directory=str(base))
    )
    return servidor, trabajo


def servir(base: Optional[Path] = None, puerto: int = 8765, abrir: bool = True) -> None:
    """Sirve el tablero con consola en vivo en http://127.0.0.1:<puerto>."""
    import webbrowser

    from .web import generar_tablero

    base = Path(base) if base else carpeta_base()
    generar_tablero(base)
    servidor, _ = crear_servidor(base, puerto)
    url = "http://127.0.0.1:%d/" % puerto
    log.info("tablero con consola en vivo en %s  (Ctrl+C para detener)", url)
    if abrir:
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        log.info("servidor detenido")
    finally:
        servidor.server_close()
