---
title: Emotion Smart Door
emoji: 🚪
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: 3.12
pinned: false
license: mit
---

# Puerta Inteligente Emocional

Aplicación desarrollada con Python, Keras y Hugging Face Spaces. Utiliza una
arquitectura híbrida CNN + LLM: una red neuronal convolucional clasifica la
expresión facial y un modelo de lenguaje genera una respuesta contextual. La
puerta simulada se abre cuando la CNN detecta felicidad con suficiente confianza.

## Flujo CNN + LLM

1. La CNN local clasifica la emoción y calcula su confianza.
2. Las reglas de Python deciden si la puerta se abre.
3. `Qwen/Qwen2.5-7B-Instruct`, servido desde Hugging Face, genera una respuesta
   breve según la emoción y decisión.
4. Si el LLM no está disponible, se muestra una respuesta local de respaldo.

El LLM solamente genera texto: nunca decide ni modifica el acceso.

La interfaz utiliza un tema oscuro adaptable a dispositivos móviles.

## Tecnologías

- Hugging Face Spaces para alojar y ejecutar la aplicación.
- Hugging Face Inference API y Qwen2.5 para generar respuestas con un LLM.
- Gradio para la interfaz web y captura desde cámara.
- Python, Keras y OpenCV para detección facial e inferencia.
- Modelo CNN entrenado con imágenes FER2013 y RAF-DB.

## Arquitectura

```text
Cámara o imagen
      |
      v
OpenCV detecta y recorta el rostro
      |
      v
CNN Keras clasifica una de las siete emociones
      |
      +--> Reglas Python controlan la puerta simulada
      |
      v
LLM Qwen genera una respuesta contextual
      |
      v
Gradio muestra resultado, probabilidades y puerta
```

La CNN reconoce `enojo`, `disgusto`, `miedo`, `felicidad`, `neutral`,
`tristeza` y `sorpresa`. La puerta solamente se abre cuando detecta felicidad
con una confianza mínima de 45 %.

## Cómo se construyó

1. Se entrenó una CNN con imágenes de expresiones faciales de FER2013 y RAF-DB.
2. La arquitectura entrenada se guardó en `Data/emociondetector.json` y sus
   pesos en `Data/emociondetector.h5`.
3. OpenCV utiliza un clasificador Haar Cascade para encontrar el rostro dentro
   de cada imagen.
4. Python convierte el rostro a escala de grises, lo redimensiona a `48x48`,
   normaliza sus valores y lo entrega a la CNN.
5. La CNN devuelve las probabilidades de las siete emociones.
6. Python abre la puerta simulada únicamente cuando la emoción principal es
   felicidad y supera el umbral configurado.
7. El resultado de la CNN y la decisión se envían al LLM
   `Qwen/Qwen2.5-7B-Instruct` mediante Hugging Face Inference API.
8. El LLM crea un mensaje breve y amable, pero nunca modifica la decisión de
   acceso.
9. Gradio construye la interfaz y Hugging Face Spaces aloja la aplicación.

## Ejecución local

```bash
pip install -r requirements.txt
python app.py
```

Para activar las respuestas generadas por el LLM se debe definir `HF_TOKEN`
como variable de entorno. Sin ella, la aplicación utiliza respuestas locales de
respaldo y mantiene funcionando la CNN.

## Puntos para la exposición

- Es una solución híbrida: la CNN entiende imágenes y el LLM genera lenguaje.
- OpenCV localiza el rostro antes de clasificarlo.
- Python conecta los modelos, aplica reglas y controla la simulación.
- El LLM no controla la puerta, evitando que una respuesta generativa cambie
  una decisión del sistema.
- Hugging Face se utiliza para alojar la aplicación y ejecutar el LLM.
- Wokwi no fue necesario porque la puerta se representa visualmente en Gradio.

La simulación tiene fines educativos y no debe utilizarse como un sistema real
de seguridad o control de acceso.
