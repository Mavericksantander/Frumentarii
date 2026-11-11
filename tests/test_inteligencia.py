from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from praetor_frumentarii import core, inteligencia as intel
from praetor_frumentarii.inteligencia.hallazgos import Regla


def _poligonos(cajas, **columnas):
    """Cajas (x0, y0, x1, y1) en metros UTM 19S, devueltas en EPSG:4326."""
    g = gpd.GeoDataFrame(columnas, geometry=[box(*c) for c in cajas], crs="EPSG:32719")
    return g.to_crs(4326)


def _escribir_corrida(modulo, aoi, analisis, archivo, gdf, parametros, avisos=()):
    carpeta = core.carpeta_salida(modulo, aoi)
    gdf.to_file(carpeta / archivo, driver="GeoJSON")
    nombre = "corrida_%s.json" % analisis if analisis else "corrida.json"
    (carpeta / nombre).write_text(json.dumps({
        "modulo": modulo, "analisis": analisis, "aoi": aoi, "parametros": parametros,
        "avisos": list(avisos), "generado": "2024-02-20T10:00:00-03:00"}), encoding="utf-8")


X, Y = 260000, 6338000   # cerca de Viña del Mar en UTM 19S


def _escenario(con_focos=True):
    quemas = _poligonos([(X, Y, X + 1000, Y + 1000), (X + 1100, Y, X + 2100, Y + 1000),
                         (X + 30000, Y, X + 30200, Y + 200)], area_ha=[100.0, 100.0, 4.0])
    _escribir_corrida("cambios", "valparaiso_vina", "quemas", "area_quemada.geojson", quemas,
                      {"despues": "2024-02-04/2024-03-05"})
    if con_focos:
        focos = _poligonos([(X + 500, Y + 500, X + 900, Y + 900),
                            (X + 60000, Y, X + 60400, Y + 400),
                            (X + 90000, Y, X + 90400, Y + 400)],
                           frp_total=[300.0, 500.0, 5.0], focos=[12, 30, 1],
                           ultima_utc=["2024-02-03 05:42", "2024-02-06 18:00", "2024-02-05 18:00"],
                           dias_distintos=[2, 5, 1], persistente=[False, True, False])
        _escribir_corrida("cambios", "chile_continental", "focos", "focos_frentes.geojson", focos,
                          {"periodo": "7d"}, ["aviso general de la corrida"])


def test_severidad_logaritmica():
    assert intel.severidad_desde(200, 200) == pytest.approx(0.5)
    assert intel.severidad_desde(2000, 200) == pytest.approx(1.0)
    assert intel.severidad_desde(20, 200) == pytest.approx(0.05)      # piso
    assert intel.severidad_desde(0, 200) == 0 and intel.severidad_desde(float("nan"), 1) == 0
    assert intel.severidad_desde(50, 200) < intel.severidad_desde(100, 200)


def test_recolectar_agrupa_poligonos_en_eventos():
    _escenario()
    h = intel.recolectar()
    incendios = h[h["tipo"] == "incendio"].sort_values("magnitud", ascending=False)
    assert len(incendios) == 2                       # dos parches vecinos = un evento; el lejano, otro
    grande = incendios.iloc[0]
    assert grande["magnitud"] == 200.0 and grande["partes"] == 2 and grande["fecha"] == "2024-02-04"
    assert grande["severidad"] == pytest.approx(0.5) and grande["confianza"] == 0.60
    focos = h[h["tipo"] == "foco_calor"]
    assert len(focos) == 3 and set(focos["fecha"]) == {"2024-02-03 05:42", "2024-02-06 18:00",
                                                       "2024-02-05 18:00"}
    assert (focos["reservas"] == "").all()           # el aviso general no se copia a cada foco
    assert h["id"].is_unique and h.crs.to_epsg() == 4326
    assert len(intel.recolectar(core.carpeta_salida("vacio", "x"))) == 0


def test_corroboracion_sube_baja_y_reclasifica():
    _escenario()
    h = intel.corroborar(intel.recolectar())
    inc = h[h["tipo"] == "incendio"].sort_values("magnitud", ascending=False)
    confirmado, aislado = inc.iloc[0], inc.iloc[1]
    assert confirmado["confianza"] == pytest.approx(0.90) and confirmado["nivel"] == "alta"
    assert "focos FIRMS a menos de" in confirmado["evidencia"]
    # Había focos en la fecha, pero ninguno cerca del parche lejano: baja.
    assert aislado["confianza"] == pytest.approx(0.50) and "ninguno cerca" in aislado["evidencia"]
    fijo = h[h["tipo"] == "fuente_termica_fija"]
    assert len(fijo) == 1 and fijo.iloc[0]["categoria"] == "contexto"
    assert fijo.iloc[0]["confianza"] == pytest.approx(0.10) and "5 días" in fijo.iloc[0]["evidencia"]
    focos = h[h["tipo"] == "foco_calor"].sort_values("magnitud", ascending=False)
    assert focos.iloc[0]["confianza"] == pytest.approx(0.90)       # 0,45 + 0,15 + 0,30
    assert focos.iloc[1]["confianza"] == pytest.approx(0.35)       # una sola detección


def test_sin_datos_no_es_evidencia_en_contra():
    _escenario(con_focos=False)
    h = intel.corroborar(intel.recolectar())
    assert (h["confianza"] == 0.60).all()
    assert h["evidencia"].str.contains("no evaluable").all()


def test_manchas_recurrentes_e_inundacion():
    manchas = _poligonos([(X, Y, X + 400, Y + 400), (X + 100, Y + 100, X + 500, Y + 500),
                          (X + 9000, Y, X + 9400, Y + 400)], area_ha=[16.0, 16.0, 16.0],
                         fecha=["2024-01-07", "2024-01-19", "2024-01-19"])
    _escribir_corrida("marino", "magallanes", "", "manchas.geojson", manchas,
                      {"fechas": "2024-01-01/2024-02-29"})
    agua = _poligonos([(X + 20000, Y, X + 20500, Y + 500)], area_ha=[25.0])
    _escribir_corrida("cambios", "biobio", "inundacion", "inundacion.geojson", agua,
                      {"despues": "2023-06-22/2023-07-03"}, ["Agua agitada después del evento"])
    cambio = _poligonos([(X + 20100, Y + 100, X + 20300, Y + 300)], area_ha=[4.0])
    _escribir_corrida("cambios", "biobio", "sar", "sar_cambio.geojson", cambio,
                      {"despues": "2023-06-22/2023-07-03"})
    h = intel.corroborar(intel.recolectar())
    assert (h["tipo"] == "mancha_recurrente").sum() == 2 and (h["tipo"] == "mancha_marina").sum() == 1
    assert h[h["tipo"] == "mancha_recurrente"]["confianza"].iloc[0] == pytest.approx(0.40)
    inundacion = h[h["tipo"] == "inundacion"].iloc[0]
    assert inundacion["confianza"] == pytest.approx(0.50)          # +0,10 apoyo, -0,10 agua agitada
    assert "mismo sensor" in inundacion["evidencia"] and "agitada" in inundacion["evidencia"]


def _infraestructura():
    return gpd.GeoDataFrame(
        {"clase": ["salud", "educacion", "educacion", "rutas", "lineas", "localidades",
                   "subestaciones", "salud"],
         "nombre": ["Hospital A", "Escuela B", "", "", "", "Villa C", "PFV Campos del Sol",
                    "Posta lejana"]},
        geometry=[Point(X + 500, Y + 500), Point(X + 1200, Y + 200), Point(X + 1400, Y + 900),
                  LineString([(X - 5000, Y + 500), (X + 8000, Y + 500)]),
                  LineString([(X + 500, Y - 9000), (X + 500, Y + 9000)]),
                  Point(X + 4000, Y + 500), Point(X + 60200, Y + 200), Point(X + 50000, Y)],
        crs="EPSG:32719").to_crs(4326)


def test_exposicion_cuenta_lo_que_esta_cerca():
    zona = box(X, Y, X + 1000, Y + 1000)
    r = intel.exposicion_de(zona, _infraestructura().to_crs("EPSG:32719"))
    assert r["salud"] == 1 and r["educacion"] == 2 and r["bomberos"] == 0
    assert r["rutas_km"] == pytest.approx(2.0, abs=0.05)           # 1000 m + 500 m a cada lado
    assert r["lineas_km"] == pytest.approx(2.0, abs=0.05)
    assert r["localidades"] == 1 and "Villa C" in r["nombres"] and "Hospital A" in r["nombres"]
    assert 0.8 < r["exposicion"] <= 1.0
    assert "1 centro de salud" in intel.describir(r) and "2,0 km de ruta" in intel.describir(r)
    vacio = intel.exposicion_de(box(X + 200000, Y, X + 201000, Y + 1000),
                                _infraestructura().to_crs("EPSG:32719"))
    assert vacio["exposicion"] == 0 and "sin infraestructura" in intel.describir(vacio)
    assert intel.indice_exposicion({"salud": 1}) < intel.indice_exposicion({"salud": 3})
    assert intel.clasificar({"amenity": "hospital"}) == "salud"
    assert intel.clasificar({"power": "line"}) == "lineas" and intel.clasificar({"shop": "x"}) is None
    assert intel.clasificar({"highway": "residential", "amenity": float("nan")}) is None


def test_priorizacion_y_regla_de_instalaciones():
    _escenario()
    h = intel.calcular_exposicion(intel.corroborar(intel.recolectar()), _infraestructura())
    assert h["exposicion_evaluada"].all()
    # El foco de 500 MW ya era persistente; se prueba la regla con uno que no lo es.
    h.loc[h["tipo"] == "fuente_termica_fija", ["tipo", "categoria"]] = ["foco_calor", "alerta"]
    h = intel.ajustar_por_instalaciones(h)
    sobre_planta = h[h["nombres"].str.contains("PFV")].iloc[0]
    assert sobre_planta["tipo"] == "fuente_termica_fija" and "reflejo solar" in sobre_planta["evidencia"]
    p = intel.priorizar(h)
    assert list(p["prioridad"]) == sorted(p["prioridad"], reverse=True)
    primero = p.iloc[0]
    assert primero["tipo"] in ("incendio", "foco_calor") and primero["prioridad"] > 40
    assert "severidad" in primero["motivo"] and primero["verificar"]
    contexto = p[p["categoria"] != "alerta"]
    assert (contexto["prioridad"] < 15).all() and contexto["motivo"].str.contains("contexto").all()
    r = intel.resumen(p)
    assert r["alertas"] == int((p["categoria"] == "alerta").sum()) and r["confianza_alta"] >= 1
    sin = intel.priorizar(h.assign(exposicion_evaluada=False))
    assert sin["motivo"].str.contains("no evaluada").all()
    assert intel.resumen(h.iloc[0:0])["alertas"] == 0


def test_evaluar_escribe_parte_y_archivos():
    _escenario()
    r = intel.evaluar(infraestructura=_infraestructura())
    texto = r["parte"].read_text(encoding="utf-8")
    assert r["parte"].parent.parts[-2:] == ("inteligencia", "general")
    for esperado in ("# Parte de situación: general", "## Alertas priorizadas", "Área quemada",
                     "Por qué esa confianza", "## Lo que no se sabe", "## Método", "Hospital A"):
        assert esperado in texto, esperado
    assert r["indicadores"]["alertas"] >= 3 and r["png"].stat().st_size > 5000
    guardado = gpd.read_file(r["parte"].parent / "hallazgos.geojson")
    assert {"tipo", "prioridad", "confianza", "evidencia", "verificar"} <= set(guardado.columns)
    solo = intel.evaluar("atacama", con_exposicion=False)       # sin hallazgos en esa zona
    assert solo["indicadores"]["hallazgos"] == 0
    assert "No hay alertas" in solo["parte"].read_text(encoding="utf-8")


def test_cli_inteligencia():
    from praetor_frumentarii.inteligencia.cli import crear_parser, main

    a = crear_parser().parse_args(["evaluar", "--sin-exposicion"])
    assert a.aoi is None and a.sin_exposicion
    assert main(["evaluar", "--sin-exposicion"]) == 0
