#!/bin/bash
# PRÆTOR Frumentarii: instalador para macOS (también funciona en Linux con bash).
# Crea .venv, instala las dependencias y corre los tests.
# Doble clic en Finder o:  ./instalar_mac.command
set -u
echo "=============================================================="
echo " PRÆTOR Frumentarii | instalación"
echo " Análisis geoespacial con imágenes satelitales abiertas, uso civil"
echo "=============================================================="
cd "$(dirname "$0")" || { echo "ERROR: no pude entrar a la carpeta del proyecto"; exit 1; }
echo "[1/5] Carpeta del proyecto: $(pwd)"

pausa_final() {
  # Al abrir con doble clic la ventana se cerraría sin dejar leer el resultado.
  if [ -t 0 ] && [ -z "${FRUMENTARII_SIN_PAUSA:-}" ]; then
    echo ""
    read -r -p "Presiona Enter para cerrar..." _
  fi
}

echo "[2/5] Buscando Python 3.9 a 3.12..."
PY=""
for candidato in python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$candidato" >/dev/null 2>&1; then
    if "$candidato" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
      PY="$(command -v "$candidato")"
      break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: no encontré Python 3.9 a 3.12."
  echo "       Instálalo con Homebrew:  brew install python@3.12"
  pausa_final
  exit 1
fi
echo "      Usando $PY ($("$PY" --version 2>&1), $(uname -m))"

echo "[3/5] Creando el entorno virtual .venv..."
if [ ! -x ".venv/bin/python" ]; then
  "$PY" -m venv .venv || { echo "ERROR: no pude crear .venv"; pausa_final; exit 1; }
  echo "      Entorno creado."
else
  echo "      Ya existía; se reutiliza."
fi

echo "[4/5] Instalando dependencias (la primera vez tarda entre 2 y 5 minutos)..."
.venv/bin/python -m pip install --upgrade pip --quiet --disable-pip-version-check \
  || { echo "ERROR: no pude actualizar pip (¿hay conexión a internet?)"; pausa_final; exit 1; }
.venv/bin/python -m pip install -e ".[dev]" --disable-pip-version-check --progress-bar off 2>&1 \
  | grep -E "^(Collecting|Building|Successfully|ERROR)" \
  | sed -e 's/^/      /'
if ! .venv/bin/python -c "import praetor_frumentarii" 2>/dev/null; then
  echo "ERROR: la instalación falló. Repite el comando para ver el detalle:"
  echo "       .venv/bin/python -m pip install -e '.[dev]'"
  pausa_final
  exit 1
fi
echo "      Dependencias listas."

echo "[5/5] Corriendo los tests (las pruebas con red se saltan solas si no hay conexión)..."
if .venv/bin/python -m pytest -q; then
  echo ""
  echo "=============================================================="
  echo " Instalación correcta. Para empezar:"
  echo "   source .venv/bin/activate"
  echo "   frumentarii --help"
  echo "   frumentarii terreno costo --aoi chiloe"
  echo " O abre  observar_en_vivo.command  para una demostración."
  echo "=============================================================="
  ESTADO=0
else
  echo ""
  echo "ATENCIÓN: algunos tests fallaron. Revisa los mensajes de arriba."
  ESTADO=1
fi
pausa_final
exit $ESTADO
