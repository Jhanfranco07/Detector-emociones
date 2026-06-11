import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from huggingface_hub import InferenceClient
from keras.models import model_from_json


BASE_DIR = Path(__file__).resolve().parent
MODEL_JSON = BASE_DIR / "Data" / "emociondetector.json"
MODEL_WEIGHTS = BASE_DIR / "Data" / "emociondetector.h5"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

LABELS = {
    0: "Enojo",
    1: "Disgusto",
    2: "Miedo",
    3: "Felicidad",
    4: "Neutral",
    5: "Tristeza",
    6: "Sorpresa",
}

RESPONSES = {
    "Enojo": "Acceso en pausa. Respira unos segundos e inténtalo nuevamente.",
    "Disgusto": "La puerta permanece cerrada. Intenta cambiar tu expresión.",
    "Miedo": "Todo está bien. La puerta permanece cerrada hasta detectar felicidad.",
    "Felicidad": "¡Bienvenido! Felicidad detectada: puerta abierta.",
    "Neutral": "La puerta está cerrada. Regálanos una sonrisa para ingresar.",
    "Tristeza": "Ánimo, hoy puede mejorar. Sonríe para abrir la puerta.",
    "Sorpresa": "¡Sorpresa detectada! Sonríe para confirmar el acceso.",
}

EMOTION_STYLES = {
    "Enojo": {"color": "#fb7185", "accent": "#e11d48", "face": "angry"},
    "Disgusto": {"color": "#a3e635", "accent": "#65a30d", "face": "disgust"},
    "Miedo": {"color": "#c084fc", "accent": "#9333ea", "face": "fear"},
    "Felicidad": {"color": "#34d399", "accent": "#059669", "face": "happy"},
    "Neutral": {"color": "#94a3b8", "accent": "#64748b", "face": "neutral"},
    "Tristeza": {"color": "#60a5fa", "accent": "#2563eb", "face": "sad"},
    "Sorpresa": {"color": "#fbbf24", "accent": "#d97706", "face": "surprise"},
}

with MODEL_JSON.open("r", encoding="utf-8") as model_file:
    emotion_model = model_from_json(model_file.read())
emotion_model.load_weights(MODEL_WEIGHTS)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def generate_llm_response(emotion: str, confidence: float, is_open: bool) -> tuple[str, bool]:
    fallback = RESPONSES[emotion]
    if emotion == "Felicidad" and not is_open:
        fallback = (
            f"Se detectó felicidad con {confidence:.1%} de confianza. "
            "Sonríe un poco más para superar el 45% y abrir la puerta."
        )

    token = os.getenv("HF_TOKEN")
    if not token:
        return fallback, False

    access_state = "autorizado" if is_open else "en espera"
    prompt = (
        f"La CNN detectó la emoción {emotion} con {confidence:.0%} de confianza. "
        f"El acceso está {access_state}. Genera una respuesta amable y natural para "
        "la persona, en español y de máximo 22 palabras. No diagnostiques su estado "
        "psicológico y no cambies la decisión de acceso."
    )
    try:
        client = InferenceClient(model=LLM_MODEL, token=token, timeout=20)
        result = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres el asistente breve y cordial de una puerta inteligente. "
                        "La CNN y las reglas del sistema deciden el acceso."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=60,
            temperature=0.65,
        )
        response = result.choices[0].message.content.strip().strip('"')
        return response or fallback, bool(response)
    except Exception:
        return fallback, False


def face_svg(expression: str, color: str) -> str:
    mouths = {
        "happy": '<path d="M47 72 Q64 91 81 72" class="face-line"/>',
        "sad": '<path d="M48 83 Q64 65 80 83" class="face-line"/>',
        "angry": '<path d="M48 79 Q64 71 80 79" class="face-line"/>',
        "disgust": '<path d="M47 78 Q55 70 63 78 T81 78" class="face-line"/>',
        "fear": '<ellipse cx="64" cy="79" rx="9" ry="13" class="face-line"/>',
        "surprise": '<circle cx="64" cy="79" r="10" class="face-line"/>',
        "neutral": '<path d="M49 78 L79 78" class="face-line"/>',
    }
    brows = {
        "angry": '<path d="M39 47 L55 52 M89 47 L73 52" class="face-line"/>',
        "fear": '<path d="M39 45 Q47 38 55 45 M73 45 Q81 38 89 45" class="face-line"/>',
        "sad": '<path d="M39 48 Q47 41 55 48 M73 48 Q81 41 89 48" class="face-line"/>',
    }.get(expression, "")
    eyes = (
        '<path d="M39 59 Q47 66 55 59 M73 59 Q81 66 89 59" class="face-line"/>'
        if expression == "happy"
        else '<circle cx="47" cy="59" r="4"/><circle cx="81" cy="59" r="4"/>'
    )
    return f"""
    <svg class="emotion-face" viewBox="0 0 128 128" aria-label="Rostro {expression}">
      <circle cx="64" cy="64" r="52" fill="{color}" opacity=".16"/>
      <circle cx="64" cy="64" r="45" fill="{color}" opacity=".32" stroke="{color}" stroke-width="3"/>
      {brows}{eyes}{mouths[expression]}
    </svg>
    """


def door_html(is_open: bool) -> str:
    state_class = "open" if is_open else "closed"
    state_text = "PUERTA ABIERTA" if is_open else "PUERTA CERRADA"
    state_icon = "ACCESO AUTORIZADO" if is_open else "ACCESO EN ESPERA"
    return f"""
    <div class="door-card">
      <div class="section-heading">
        <span class="section-kicker">SIMULACIÓN EN TIEMPO REAL</span>
        <strong>Control de acceso</strong>
      </div>
      <div class="door-scene">
        <div class="door-status {state_class}">
          <span class="status-dot"></span>{state_icon}
        </div>
        <div class="door-wall">
          <div class="access-light {state_class}"></div>
          <div class="door-frame">
            <div class="door {state_class}">
              <div class="door-panel"></div>
              <div class="door-panel"></div>
              <div class="handle"></div>
            </div>
          </div>
          <div class="floor-light {state_class}"></div>
        </div>
        <strong class="door-state-text">{state_text}</strong>
      </div>
    </div>
    """


def dashboard_html(
    emotion=None, confidence=0.0, scores=None, notice=None, used_llm=False
) -> str:
    scores = scores or {}
    emotion = emotion or "Neutral"
    style = EMOTION_STYLES[emotion]
    title = emotion if scores else "Esperando análisis"
    subtitle = notice or "Captura o sube una imagen para comenzar."
    response_source = (
        f"RESPUESTA GENERADA POR LLM · {LLM_MODEL.split('/')[-1]}"
        if used_llm
        else "RESPUESTA LOCAL DE RESPALDO"
    )
    badge = f"{confidence:.0%} DE CONFIANZA" if scores else "SISTEMA PREPARADO"
    bars = ""
    for label, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        bar_style = EMOTION_STYLES[label]
        bars += f"""
        <div class="score-row">
          <div class="score-meta"><span>{label}</span><strong>{value:.0%}</strong></div>
          <div class="score-track"><div class="score-fill" style="width:{value * 100:.1f}%;
            background:{bar_style['color']}"></div></div>
        </div>
        """
    if not bars:
        bars = """
        <div class="empty-analysis">
          <span></span><span></span><span></span>
          <p>Las probabilidades aparecerán aquí.</p>
        </div>
        """
    return f"""
    <div class="insight-card" style="--emotion:{style['color']};--accent:{style['accent']}">
      <div class="section-heading">
        <span class="section-kicker">LECTURA DE LA IA</span>
        <strong>Estado emocional</strong>
      </div>
      <div class="emotion-summary">
        {face_svg(style['face'], style['color'])}
        <div class="emotion-copy">
          <span class="confidence-badge">{badge}</span>
          <h2>{title}</h2>
        </div>
      </div>
      <div class="ai-response">
        <span>{response_source}</span>
        <p>{subtitle}</p>
      </div>
      <div class="score-list">{bars}</div>
      <div class="privacy-note">
        <span class="privacy-icon">i</span>
        La imagen se procesa para esta demostración y no se almacena.
      </div>
    </div>
    """


def analyze_emotion(image: np.ndarray):
    if image is None:
        return None, dashboard_html(), door_html(False)

    annotated = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return (
            annotated,
            dashboard_html(
                notice="No se detectó un rostro. Mira de frente a la cámara."
            ),
            door_html(False),
        )

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    face = cv2.resize(gray[y : y + height, x : x + width], (48, 48))
    features = face.astype("float32").reshape(1, 48, 48, 1) / 255.0
    probabilities = emotion_model.predict(features, verbose=0)[0]

    emotion_index = int(np.argmax(probabilities))
    emotion = LABELS[emotion_index]
    confidence = float(probabilities[emotion_index])
    is_open = emotion == "Felicidad" and confidence >= 0.45

    color = (40, 190, 90) if is_open else (235, 80, 70)
    cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 3)
    caption = f"{emotion}: {confidence:.1%}"
    cv2.putText(
        annotated,
        caption,
        (x, max(y - 12, 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    decision, used_llm = generate_llm_response(emotion, confidence, is_open)

    scores = {
        label: float(probabilities[index])
        for index, label in LABELS.items()
    }
    return (
        annotated,
        dashboard_html(emotion, confidence, scores, decision, used_llm),
        door_html(is_open),
    )


CSS = """
:root {--page:#07111f;--card:#0e1b2e;--card-2:#0b1728;--line:rgba(148,163,184,.18);
  --muted:#91a4bd;--text:#f8fafc;--soft-text:#dbeafe;--step:#17243a;--step-line:#334155;
  --media:#07111f;--header:rgba(7,17,31,.92);--shadow:#02061755;--response:#07111f88;}
body, .gradio-container {background:var(--page) !important;color:var(--text)!important;transition:background .25s,color .25s;}
.gradio-container {max-width:1440px !important;padding:0 24px 48px !important;color:var(--text) !important;}
.main-header {margin:0 -24px 24px;padding:22px 30px;border-bottom:1px solid var(--line);
  background:var(--header);backdrop-filter:blur(16px);transition:background .25s;}
.header-inner {max-width:1380px;margin:auto;display:flex;align-items:center;justify-content:space-between;gap:24px;}
.brand {display:flex;align-items:center;gap:14px}.brand-mark {width:46px;height:46px;display:grid;place-items:center;
  border-radius:14px;background:linear-gradient(145deg,#2563eb,#7c3aed);box-shadow:0 12px 30px #2563eb55;}
.brand-mark svg {width:25px}.brand h1 {font-size:1.15rem;margin:0;letter-spacing:-.02em;color:var(--text)}
.brand p {margin:3px 0 0;color:var(--muted);font-size:.82rem}
.header-actions {display:flex;align-items:center;justify-content:flex-end;gap:10px;min-height:40px}
.live-pill {height:38px;box-sizing:border-box;display:flex;align-items:center;justify-content:center;gap:9px;padding:0 14px;
  border:1px solid #34d39955;border-radius:99px;color:#6ee7b7;font-size:.75rem;font-weight:800;
  letter-spacing:.08em;background:#064e3b33}
.live-pill span,.status-dot {width:8px;height:8px;border-radius:50%;background:#34d399;box-shadow:0 0 12px #34d399}
.workflow {display:flex;align-items:center;justify-content:center;gap:10px;margin-bottom:24px;color:var(--muted);
  font-size:.76rem;font-weight:800;letter-spacing:.06em}.workflow b {color:var(--soft-text);background:var(--step);
  border:1px solid var(--line);padding:8px 13px;border-radius:99px;box-shadow:0 6px 18px var(--shadow)}
.workflow b span {color:#60a5fa;margin-right:4px}.workflow i {width:28px;height:1px;background:var(--step-line)}
.section-heading {display:flex;flex-direction:column;gap:4px;margin-bottom:16px}.section-heading strong {font-size:1.08rem;color:var(--text)}
.section-kicker {color:#60a5fa;font-size:.67rem;font-weight:900;letter-spacing:.14em}
.media-card,.insight-card,.door-card {height:100%;box-sizing:border-box;border:1px solid var(--line);
  background:linear-gradient(145deg,var(--card),var(--card-2));border-radius:20px;padding:20px;box-shadow:0 20px 50px var(--shadow)}
.media-card {min-height:460px!important;padding:0!important;overflow:hidden}.media-card > div:first-child {margin:0!important}
.media-card img {width:100%!important;height:410px!important;max-height:410px!important;object-fit:contain!important;background:var(--media)!important}
.analyze-btn {min-height:52px!important;margin-top:14px!important;border:0!important;border-radius:14px!important;
  background:linear-gradient(90deg,#2563eb,#7c3aed)!important;font-weight:800!important;box-shadow:0 12px 28px #2563eb40}
.emotion-summary {display:flex;align-items:center;gap:20px;padding:18px;border-radius:16px;
  border:1px solid color-mix(in srgb,var(--emotion) 25%,transparent);background:color-mix(in srgb,var(--emotion) 7%,transparent)}
.emotion-face {width:132px;min-width:132px;filter:drop-shadow(0 12px 22px #02061788)}
.face-line {fill:none;stroke:var(--emotion);stroke-width:5;stroke-linecap:round}
.emotion-face circle:not(:first-child):not(:nth-child(2)) {fill:var(--emotion)}
.emotion-copy h2 {font-size:2rem;line-height:1;margin:12px 0 8px;color:var(--emotion)}
.emotion-copy p {color:var(--soft-text);line-height:1.5;margin:0;max-width:390px}
.confidence-badge {font-size:.64rem;font-weight:900;letter-spacing:.1em;color:var(--emotion);
  border:1px solid color-mix(in srgb,var(--emotion) 40%,transparent);border-radius:99px;padding:6px 9px}
.ai-response {margin-top:16px;padding:15px 16px;border-left:3px solid var(--emotion);border-radius:5px 12px 12px 5px;
  background:var(--response)}.ai-response span {color:var(--emotion);font-size:.61rem;font-weight:900;letter-spacing:.11em}
.ai-response p {margin:7px 0 0;color:var(--soft-text);line-height:1.5;font-size:.88rem}
.score-list {margin-top:20px;display:grid;gap:10px}.score-meta {display:flex;justify-content:space-between;
  font-size:.76rem;color:var(--soft-text)}.score-meta strong {color:var(--text)}.score-track {height:5px;border-radius:99px;
  background:var(--step);overflow:hidden;margin-top:5px}.score-fill {height:100%;border-radius:99px;transition:width .6s ease}
.empty-analysis {height:156px;display:flex;align-items:center;justify-content:center;gap:7px;color:var(--muted)}
.empty-analysis span {width:7px;height:7px;border-radius:50%;background:#475569}.empty-analysis p {margin-left:8px;font-size:.82rem}
.privacy-note {display:flex;align-items:center;gap:9px;margin-top:20px;padding-top:15px;border-top:1px solid var(--line);
  color:var(--muted);font-size:.7rem}.privacy-icon {width:18px;height:18px;border:1px solid #64748b;border-radius:50%;
  display:grid;place-items:center;font-weight:bold}
.door-scene {min-height:390px;display:flex;flex-direction:column;align-items:center;justify-content:center;
  border-radius:16px;background:radial-gradient(circle at 50% 30%,#20314c,#081221 68%);perspective:900px;overflow:hidden}
.door-status {display:flex;align-items:center;gap:9px;margin-bottom:16px;padding:7px 12px;border-radius:99px;color:#fda4af;
  background:#88133755;border:1px solid #fb718555;font-size:.66rem;font-weight:900;letter-spacing:.1em}
.door-status.open {color:#6ee7b7;background:#064e3b88;border-color:#34d39966}.door-status.closed .status-dot {background:#fb7185;box-shadow:0 0 12px #fb7185}
.door-wall {position:relative;width:250px;height:265px;padding-top:15px;background:linear-gradient(90deg,#25344b,#34445d,#25344b);
  border-radius:8px 8px 0 0;display:flex;justify-content:center;transform-style:preserve-3d}
.access-light {position:absolute;right:18px;top:20px;width:9px;height:9px;border-radius:50%;background:#fb7185;box-shadow:0 0 16px #fb7185}
.access-light.open {background:#34d399;box-shadow:0 0 18px #34d399}.door-frame {width:170px;height:250px;padding:8px 8px 0;
  background:#111827;border-radius:5px 5px 0 0;transform-style:preserve-3d}.door {position:relative;width:100%;height:100%;
  box-sizing:border-box;background:linear-gradient(120deg,#9a5b20,#5f3515);border:5px solid #3f2613;transform-origin:left;
  transition:transform 1s ease,filter .5s;box-shadow:inset 0 0 30px #0005}.door.open {transform:rotateY(-68deg);filter:brightness(1.2)}
.door-panel {height:35%;margin:15px;border:3px solid #c07a31;box-shadow:inset 0 0 15px #0005}.handle {position:absolute;right:12px;top:52%;
  width:10px;height:10px;border-radius:50%;background:#fde68a;box-shadow:0 0 10px #facc15}.floor-light {position:absolute;bottom:-22px;
  width:170px;height:35px;background:#fb718520;filter:blur(15px)}.floor-light.open {background:#34d39955}.door-state-text {margin-top:18px;font-size:.8rem;letter-spacing:.12em}
footer {display:none!important}
@media(max-width:800px){
  .gradio-container{padding:0 12px 32px!important}.main-header{margin:0 -12px 18px;padding:16px 14px}
  .header-inner{align-items:center;gap:10px}.brand{gap:10px}.brand-mark{width:40px;height:40px;border-radius:12px}
  .brand h1{font-size:1rem}.brand p{font-size:.69rem;max-width:190px}.live-pill{display:none}.header-actions{min-height:40px}
  .workflow{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin:8px 0 22px}
  .workflow b{display:flex;align-items:center;justify-content:flex-start;min-height:42px;padding:9px 10px;border-radius:12px;
    font-size:.67rem;line-height:1.2;letter-spacing:.035em;color:var(--text);border-color:#60a5fa66}
  .workflow b span{display:grid;place-items:center;min-width:22px;height:22px;margin-right:7px;border-radius:7px;
    color:white;background:#2563eb}.workflow i{display:none}.section-heading{margin:8px 0 10px}
  .section-kicker{font-size:.62rem}.emotion-summary{flex-direction:column;text-align:center}.emotion-copy h2{font-size:1.6rem}
  .emotion-face{width:110px;min-width:110px}.door-card,.insight-card{padding:14px}.door-scene{min-height:350px}
  .media-card{min-height:360px!important}.media-card img{height:310px!important;max-height:310px!important}
}
"""

with gr.Blocks(title="EmotionGate AI") as demo:
    gr.HTML(
        """
        <header class="main-header"><div class="header-inner">
          <div class="brand">
            <div class="brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
              <path d="M4 21V4a1 1 0 0 1 1-1h11v18M4 21h16M12 12h.01"/></svg></div>
            <div><h1>EmotionGate AI</h1><p>Sistema híbrido de inteligencia artificial · CNN + LLM</p></div>
          </div>
          <div class="header-actions">
            <div class="live-pill"><span></span>SISTEMA OPERATIVO</div>
          </div>
        </div></header>
        <div class="workflow">
          <b><span>01</span>CAPTURA</b><i></i><b><span>02</span>EMOCIÓN · CNN</b><i></i>
          <b><span>03</span>RESPUESTA · LLM</b><i></i><b><span>04</span>ACCESO</b>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=6):
            gr.HTML('<div class="section-heading"><span class="section-kicker">ENTRADA VISUAL</span><strong>Captura facial</strong></div>')
            input_image = gr.Image(
                label="",
                sources=["webcam", "upload"],
                type="numpy",
                elem_classes=["media-card"],
            )
            analyze_button = gr.Button("Analizar emoción y verificar acceso", variant="primary", elem_classes=["analyze-btn"])
        with gr.Column(scale=6):
            gr.HTML('<div class="section-heading"><span class="section-kicker">VISIÓN COMPUTACIONAL</span><strong>Detección procesada</strong></div>')
            output_image = gr.Image(label="", interactive=False, elem_classes=["media-card"])
    with gr.Row():
        with gr.Column(scale=7):
            dashboard = gr.HTML(dashboard_html())
        with gr.Column(scale=5):
            door = gr.HTML(door_html(False))

    analyze_button.click(
        fn=analyze_emotion,
        inputs=input_image,
        outputs=[output_image, dashboard, door],
    )
    input_image.change(
        fn=analyze_emotion,
        inputs=input_image,
        outputs=[output_image, dashboard, door],
    )
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(),
        css=CSS,
        ssr_mode=False,
    )
