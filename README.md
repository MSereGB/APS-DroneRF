# Trabajo Final de Análisis y Procesamiento de Señales

En esta carpeta están la memoria final, los notebooks y los archivos necesarios para seguir
y ejecutar el trabajo “Detección de actividad de drones a partir de señales RF públicas”.

El análisis se hizo con DroneRF y con señales sintéticas de apoyo. Las amplitudes son
normalizadas y las comparaciones de potencia son relativas. La aplicación incluida simula
una adquisición desde archivos; no recibe señales desde hardware de RF.

## Por dónde empezar

1. Abrir `Trabajo_Final_APS_Maria_Serena_Gil.pdf`.
2. Recorrer los notebooks de `notebooks/` en orden, del 00 al 06.
3. Usar `src/` y `scripts/` si se quiere repetir alguna parte del procesamiento.

## Notebooks

| Archivo | Contenido | Necesita DroneRF local |
| --- | --- | --- |
| `00_verificaciones_sinteticas.ipynb` | FFT, PSD, Welch, ventanas, ruido y cuantización | No |
| `01_exploracion_dronerf.ipynb` | Archivos, etiquetas y ejemplos temporales | Opcional |
| `02_analisis_tiempo_fft_psd.ipynb` | Comparación temporal, FFT y Welch | Sí |
| `03_ventanas_y_bandas.ipynb` | Ventaneo, zero padding y bandas | No |
| `04_filtros_digitales.ipynb` | FIR, IIR y Transformada Z | Opcional |
| `05_caracteristicas_y_validacion.ipynb` | Características y validación | No |
| `06_resultados_finales.ipynb` | Resultados, caso de estudio y límites | No |

Los notebooks ya incluyen los gráficos y las tablas obtenidos, por lo que se pueden revisar
desde GitHub sin ejecutarlos. Los notebooks 03, 05 y 06 también pueden ejecutarse con las
muestras y los resultados incluidos en esta carpeta.

## Entorno

Se recomienda Python 3.13. Desde la carpeta principal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,docs]"
jupyter notebook
```

## Dataset DroneRF

Los archivos CSV originales no se incluyen porque ocupan aproximadamente 14 GB. El archivo
`data/manifests/dronerf_demo_v2_manifest.json` registra la selección de datos utilizada, las
etiquetas y la separación entre desarrollo, evaluación y muestras de demostración.

Para ejecutar los notebooks 01, 02 y 04 con los datos reales:

```powershell
$env:DATA_DIR = "C:\ruta\a\DroneRF_demo_v2"
jupyter notebook
```

El dataset original se encuentra en:
https://data.mendeley.com/datasets/f4c2b4n755/1

## Reproducción del procesamiento final

Con el dataset ya preparado, el orden principal es:

```powershell
python scripts/construir_caracteristicas_demo.py --data-dir $env:DATA_DIR
python scripts/calibrar_detector_demo.py
python scripts/exportar_resultados_demo.py
python scripts/evaluar_ruido_demo.py
python scripts/auditar_robustez_detector.py
```

Los parámetros usados en el análisis final fueron `N=1024`, ventana Hann, 50 % de
solapamiento, 100 ventanas por parte y 20 bandas relativas.

## Demostración local

La carpeta `muestras_demo/` contiene señales reservadas que no se usaron para ajustar el
detector. Para abrir la aplicación:

```powershell
python scripts/ejecutar_demo.py
```

Luego se abre `http://127.0.0.1:7860`.

El dataset completo debe descargarse por separado desde su fuente pública. No es necesario
para leer la memoria, recorrer los resultados guardados ni ejecutar la demostración con las
muestras incluidas.
