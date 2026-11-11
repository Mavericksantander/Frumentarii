#!/bin/bash
# PRÆTOR Frumentarii: observación en vivo.
# Procesa una AOI real y muestra cada escena en una ventana a medida que se
# procesa; al terminar evalúa la situación y abre el tablero con consola en vivo.
#
#   ./observar_en_vivo.command              -> demostración de incendios (Valparaíso/Viña)
#   ./observar_en_vivo.command terreno      -> terreno | quemas | inundacion | marino | suelo
#   PAUSA=3 ./observar_en_vivo.command      -> segundos por escena (por defecto 10)
set -u
echo "=============================================================="
echo " PRÆTOR Frumentarii | observación en vivo"
echo "=============================================================="
cd "$(dirname "$0")" || { echo "ERROR: no pude entrar a la carpeta del proyecto"; exit 1; }
DEMO="${1:-quemas}"
PAUSA="${PAUSA:-10}"
echo "[1/4] Demostración: $DEMO | pausa por escena: ${PAUSA} s"

pausa_final() {
  if [ -t 0 ] && [ -z "${FRUMENTARII_SIN_PAUSA:-}" ]; then
    echo ""
    read -r -p "Presiona Enter para cerrar..." _
  fi
}

echo "[2/4] Revisando la instalación..."
if [ ! -x ".venv/bin/frumentarii" ]; then
  echo "      No está instalado todavía: ejecuto instalar_mac.command primero."
  FRUMENTARII_SIN_PAUSA=1 ./instalar_mac.command || { pausa_final; exit 1; }
fi
echo "      Instalación encontrada."

echo "[3/4] Procesando con ventana en vivo (necesita internet; cierra la ventana o Ctrl+C para cortar)..."
F=".venv/bin/frumentarii"
case "$DEMO" in
  quemas)
    "$F" cambios quemas --aoi valparaiso_vina --antes 2024-01-05/2024-02-01 \
      --despues 2024-02-04/2024-03-05 --en-vivo --pausa "$PAUSA" ;;
  inundacion)
    "$F" cambios inundacion --aoi biobio --antes 2023-05-20/2023-06-18 \
      --despues 2023-06-22/2023-07-03 --en-vivo --pausa "$PAUSA" ;;
  terreno)
    "$F" terreno ruta --aoi chiloe --modo 4x4 --origen=-73.765,-42.480 \
      --destino=-73.652,-42.379 --en-vivo --pausa "$PAUSA" ;;
  marino)
    "$F" marino manchas --aoi magallanes --fechas 2024-01-01/2024-02-29 \
      --en-vivo --pausa "$PAUSA" ;;
  suelo)
    "$F" suelo alteracion --aoi atacama --sensor s2 --en-vivo --pausa "$PAUSA" ;;
  *)
    echo "ERROR: demostración desconocida '$DEMO'. Opciones: quemas, inundacion, terreno, marino, suelo"
    pausa_final
    exit 2 ;;
esac
ESTADO=$?
if [ $ESTADO -ne 0 ]; then
  echo "ATENCIÓN: el procesamiento terminó con error ($ESTADO). Revisa los mensajes de arriba."
  pausa_final
  exit $ESTADO
fi

echo "[4/4] Evaluando la situación y abriendo el tablero con consola en vivo..."
"$F" inteligencia evaluar || echo "ATENCIÓN: la evaluación de inteligencia falló; el tablero se abre igual."
echo ""
echo "Los archivos quedaron en la carpeta salidas/."
echo "El tablero queda abierto en http://127.0.0.1:8765  (Ctrl+C aquí para cerrarlo)."
"$F" viewer servir
