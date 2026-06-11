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

La simulación tiene fines educativos y no debe utilizarse como un sistema real
de seguridad o control de acceso.
