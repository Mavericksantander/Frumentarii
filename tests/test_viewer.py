from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from praetor_frumentarii import core, viewer
from praetor_frumentarii.core.salidas import Panel


def _corrida_sintetica(rng, modulo="terreno", analisis=None):
    aoi = core.cargar_aoi("chiloe")
    xs = np.linspace(0, 1, 60)
    da = core.crear_raster((np.outer(np.ones(50), xs) + rng.random((50, 60)) * 0.1).astype("float32"),
                           "EPSG:32718", 600000, 5300000, 30)
    da.values[:5, :5] = np.nan
    clases = core.crear_raster(rng.integers(0, 3, (50, 60)).astype("uint8"), "EPSG:32718",
                               600000, 5300000, 30)
    linea = gpd.GeoDataFrame({"modo": ["pie"]}, geometry=[LineString(
        [(600100, 5299900), (601500, 5298700)])], crs="EPSG:32718")
    t = core.guardar_geotiff(da, modulo, aoi, "costo")
    c = core.guardar_geotiff(clases, modulo, aoi, "clases")
    g = core.guardar_geojson(linea, modulo, aoi, "ruta")
    png = core.guardar_resumen_png([Panel(da, "costo")], modulo, aoi, "prueba",
                                   nombre="resumen_x" if analisis else "resumen")
    core.registrar_corrida(
        modulo, aoi, "Corrida de prueba <b>&", {"ruta_km": 1.84, "modo": "pie"}, [t, c, g, png],
        ["aviso de prueba"], fuentes=["Fuente de prueba"], analisis=analisis, png=png,
        capas_web=[{"archivo": "costo.tif", "titulo": "Costo", "cmap": "viridis", "visible": True},
                   {"archivo": "clases.tif", "titulo": "Clases", "visible": False,
                    "clases": {"1": "#ff0000", "2": "#00ff00"}},
                   {"archivo": "ruta.geojson", "titulo": "Ruta", "color": "#ff2d78"},
                   {"archivo": "no_existe.tif", "titulo": "Fantasma"}],
    )
    return t


def test_visor_vivo_cuenta_escenas_sin_pantalla(rng):
    visor = viewer.VisorVivo(pausa_s=0)
    da = core.crear_raster(rng.random((20, 30)).astype("float32"))
    visor.mostrar(Panel(da, "escena 1"))
    visor.mostrar(Panel(np.dstack([da.values] * 3), "escena 2 rgb", referencia=da))
    assert visor.cuadros == 2
    visor.cerrar()
    visor.cerrar()  # cerrar dos veces no falla
    assert viewer.VisorVivo().pausa_s == 10.0 and viewer.VisorVivo(pausa_s=-3).pausa_s == 0.0


def test_raster_a_png_transparente_y_con_limites(rng, tmp_path):
    from PIL import Image

    tif = _corrida_sintetica(rng)
    destino = tmp_path / "capa.png"
    (sur, oeste), (norte, este) = viewer.raster_a_png(tif, destino, "viridis", 0.0, 1.0)
    assert -43 < sur < norte < -42 and -74 < oeste < este < -73
    img = np.asarray(Image.open(destino))
    assert img.shape[2] == 4 and img[..., 3].max() == 255 and img[..., 3].min() == 0
    clases = tif.parent / "clases.tif"
    viewer.raster_a_png(clases, destino, clases={"1": "#ff0000", "2": "#00ff00"})
    img = np.asarray(Image.open(destino))
    colores = {tuple(c) for c in img.reshape(-1, 4)}
    assert (255, 0, 0, 255) in colores and (0, 255, 0, 255) in colores and (0, 0, 0, 0) in colores


def test_tablero_vacio_y_con_corridas(rng):
    vacio = viewer.generar_tablero()
    html = vacio.read_text(encoding="utf-8")
    assert "PRÆTOR FRUMENTARII" in html and "USO CIVIL" in html and "var mapa" in html
    assert viewer.leer_corridas() == []

    _corrida_sintetica(rng, "terreno")
    _corrida_sintetica(rng, "cambios", analisis="quemas")
    corridas = viewer.leer_corridas()
    assert {c["modulo"] for c in corridas} == {"terreno", "cambios"}
    ruta = viewer.generar_tablero()
    assert ruta.name == "tablero.html" and ruta.parent.name == "salidas"
    html = ruta.read_text(encoding="utf-8")
    datos = json.loads(html.split('"], D = ', 1)[1].split(", LIM = ", 1)[0])
    assert len(datos) == 2
    por_modulo = {d["modulo"]: d for d in datos}
    assert por_modulo["cambios"]["etiqueta"] == "CAMBIOS / QUEMAS"
    assert por_modulo["cambios"]["resumen"] == "cambios/chiloe/resumen_x.png"
    assert por_modulo["terreno"]["resumen"] == "terreno/chiloe/resumen.png"
    assert [c["titulo"] for c in por_modulo["terreno"]["capas"]] == ["Costo", "Clases", "Ruta"]
    assert por_modulo["terreno"]["capas"][1]["visible"] is False
    assert por_modulo["terreno"]["avisos"] == ["aviso de prueba"] and por_modulo["terreno"]["aproximada"]
    # El título con HTML se entrega como dato y la página lo escapa al mostrarlo.
    assert "Corrida de prueba <b>&" in por_modulo["terreno"]["titulo"]
    assert "data:image/png;base64" in html            # capas incrustadas: archivo autocontenido
    assert not (ruta.parent / ".tablero_tmp").exists()


def test_corrida_ilegible_no_rompe_el_tablero(rng):
    _corrida_sintetica(rng)
    malo = core.carpeta_salida("suelo", "atacama") / "corrida.json"
    malo.write_text("{esto no es json", encoding="utf-8")
    assert len(viewer.leer_corridas()) == 1
    assert viewer.generar_tablero().exists()


def test_cli_viewer():
    from praetor_frumentarii.viewer.cli import crear_parser, main

    a = crear_parser().parse_args(["servir", "--puerto", "9000", "--sin-abrir"])
    assert a.puerto == 9000 and a.sin_abrir
    assert main(["tablero"]) == 0 and main([]) == 2
