"""
Detector de Características Físicas - Software de Enfermería
=============================================================
Contextura : MobileNetV2 entrenado con fotos reales
Postura    : Análisis geométrico con MediaPipe PoseLandmarker

INSTALACIÓN:
    pip install "mediapipe>=0.10.14" ultralytics opencv-python numpy torch torchvision

USO (una vez tengas el modelo entrenado):
    python detector_fisico_enfermeria.py --modo camara
    python detector_fisico_enfermeria.py --modo imagen --fuente paciente.jpg

Si aún no tienes el modelo entrenado, el sistema usa la
lógica geométrica de respaldo automáticamente.
"""

import cv2
import numpy as np
import math
import urllib.request
import os
import json
from dataclasses import dataclass
from typing import Optional

from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions


# ─────────────────────────────────────────────────────────────
#  Índices BlazePose
# ─────────────────────────────────────────────────────────────
NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28

POSE_CONNECTIONS = [
    (NOSE, LEFT_SHOULDER), (NOSE, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),   (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),     (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),   (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]


# ─────────────────────────────────────────────────────────────
#  Descarga del modelo MediaPipe
# ─────────────────────────────────────────────────────────────
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
MODEL_PATH = "pose_landmarker_lite.task"


def descargar_modelo(url=MODEL_URL, destino=MODEL_PATH):
    if os.path.exists(destino):
        print(f"[INFO] Modelo MediaPipe encontrado: {destino}")
        return destino
    print("[INFO] Descargando modelo pose (~29 MB)...")
    urllib.request.urlretrieve(url, destino)
    print(f"[INFO] Guardado en: {destino}")
    return destino


# ─────────────────────────────────────────────────────────────
#  Clasificador CNN (MobileNetV2)
# ─────────────────────────────────────────────────────────────

def cargar_clasificador_cnn(
    ruta_modelo: str = "modelo_contextura.pth",
    ruta_clases: str = "clases_contextura.json",
):
    """
    Carga el modelo entrenado si existe.
    Retorna (modelo, clases, transform) o None si no está disponible.
    """
    if not os.path.exists(ruta_modelo):
        print(f"[AVISO] Modelo CNN no encontrado ({ruta_modelo}). "
              f"Se usará lógica geométrica de respaldo.")
        return None

    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        # Cargar clases
        if os.path.exists(ruta_clases):
            with open(ruta_clases) as f:
                clases = json.load(f)
        else:
            clases = ["delgado", "contextura_media", "sobrepeso", "obesidad"]

        # Reconstruir arquitectura idéntica a entrenar_contextura.py
        modelo = models.mobilenet_v2(weights=None)
        in_features = modelo.classifier[1].in_features
        modelo.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, len(clases)),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo.load_state_dict(torch.load(ruta_modelo, map_location=device))
        modelo.to(device)
        modelo.eval()

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        print(f"[INFO] Modelo CNN cargado: {ruta_modelo}  |  clases: {clases}")
        return modelo, clases, tf, device

    except Exception as e:
        print(f"[AVISO] No se pudo cargar el modelo CNN: {e}. Usando lógica geométrica.")
        return None


def predecir_cnn(recorte_bgr, cnn_pack):
    """Clasifica un recorte de persona con el modelo CNN."""
    import torch
    modelo, clases, tf, device = cnn_pack
    img_rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
    tensor  = tf(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = modelo(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()
    return clases[idx], round(probs[idx].item(), 2)


# ─────────────────────────────────────────────────────────────
#  Estructura de resultado
# ─────────────────────────────────────────────────────────────
@dataclass
class CaracteristicasFisicas:
    contextura: str
    postura: str
    confianza_contextura: float
    confianza_postura: float
    metodo_contextura: str       # "CNN" o "geometrico"
    bbox: tuple


# ─────────────────────────────────────────────────────────────
#  Dibujo del esqueleto con OpenCV puro
# ─────────────────────────────────────────────────────────────

def dibujar_esqueleto(imagen, lms,
                      color_punto=(0, 255, 200),
                      color_linea=(0, 200, 255)):
    h, w = imagen.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for i, j in POSE_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            vi = getattr(lms[i], "visibility", 1.0) or 0.0
            vj = getattr(lms[j], "visibility", 1.0) or 0.0
            if vi > 0.3 and vj > 0.3:
                cv2.line(imagen, pts[i], pts[j], color_linea, 2, cv2.LINE_AA)
    for idx, pt in enumerate(pts):
        v = getattr(lms[idx], "visibility", 1.0) or 0.0
        if v > 0.3:
            cv2.circle(imagen, pt, 4, color_punto, -1, cv2.LINE_AA)
            cv2.circle(imagen, pt, 4, (0, 0, 0),   1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  Detector principal
# ─────────────────────────────────────────────────────────────
class DetectorFisicoEnfermeria:

    CONF_YOLO  = 0.50
    MARGEN_ROI = 0.08

    COLORES = {
        "delgado":           (255, 200,  50),
        "contextura_media":  ( 50, 220,  80),
        "sobrepeso":         ( 50, 180, 255),
        "obesidad":          ( 30,  30, 255),
        "postura_correcta":  ( 50, 220,  80),
        "postura_encorvada": ( 30, 100, 255),
    }

    def __init__(
        self,
        modelo_yolo: str = "yolov8n.pt",
        modelo_pose: str = MODEL_PATH,
        modelo_cnn:  str = "modelo_contextura.pth",
        clases_cnn:  str = "clases_contextura.json",
    ):
        modelo_pose = descargar_modelo(destino=modelo_pose)

        print("[INFO] Cargando YOLOv8...")
        self.yolo = YOLO(modelo_yolo)

        print("[INFO] Iniciando MediaPipe PoseLandmarker...")
        base_opts = mp_python.BaseOptions(model_asset_path=modelo_pose)
        opts = PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(opts)

        # Intentar cargar CNN (opcional)
        self.cnn_pack = cargar_clasificador_cnn(modelo_cnn, clases_cnn)
        print("[INFO] Detector listo.\n")

    # ── Contextura geométrica (respaldo) ─────────────────────

    def _ratio_torso(self, lms, rw, rh):
        ls  = lms[LEFT_SHOULDER];  rs  = lms[RIGHT_SHOULDER]
        lh  = lms[LEFT_HIP];       rh_ = lms[RIGHT_HIP]
        ancho = max(abs(rs.x - ls.x) * rw, abs(rh_.x - lh.x) * rw)
        alto  = abs(((ls.y + rs.y) / 2) - ((lh.y + rh_.y) / 2)) * rh
        return ancho / alto if alto > 1e-3 else 0.35

    def _contextura_geometrica(self, lms, bbox, rw, rh, fw, fh):
        x1, y1, x2, y2 = bbox
        r = self._ratio_torso(lms, rw, rh)
        area_rel = ((x2 - x1) * (y2 - y1)) / max(fw * fh, 1)
        r += 0.07 if area_rel > 0.25 else (-0.04 if area_rel < 0.08 else 0.0)

        if r < 0.38:
            return "delgado",          round(min(1.0, (0.38 - r) / 0.15 + 0.6), 2)
        elif r < 0.52:
            d = min(r - 0.38, 0.52 - r)
            return "contextura_media", round(min(1.0, d / 0.07 * 0.35 + 0.65), 2)
        elif r < 0.65:
            return "sobrepeso",        round(min(1.0, (r - 0.52) / 0.13 + 0.6), 2)
        else:
            return "obesidad",         round(min(1.0, (r - 0.65) / 0.15 + 0.65), 2)

    # ── Postura geométrica ────────────────────────────────────

    def _angulo_columna(self, lms, rw, rh):
        ls = lms[LEFT_SHOULDER];  rs = lms[RIGHT_SHOULDER]
        lh = lms[LEFT_HIP];       rh_ = lms[RIGHT_HIP]
        hx = ((ls.x + rs.x) / 2) * rw;  hy = ((ls.y + rs.y) / 2) * rh
        cx = ((lh.x + rh_.x) / 2) * rw; cy = ((lh.y + rh_.y) / 2) * rh
        dx, dy = hx - cx, hy - cy
        return math.degrees(math.atan2(abs(dx), abs(dy))) if abs(dy) > 1e-3 else 0.0

    def _angulo_cuello(self, lms):
        n  = lms[NOSE]
        ls = lms[LEFT_SHOULDER]; rs = lms[RIGHT_SHOULDER]
        dx = n.x - (ls.x + rs.x) / 2
        dy = (ls.y + rs.y) / 2 - n.y
        return math.degrees(math.atan2(abs(dx), abs(dy))) if abs(dy) > 1e-3 else 0.0

    def _clasificar_postura(self, lms, rw, rh):
        sc = min(1.0, self._angulo_columna(lms, rw, rh) / 25.0)
        sn = min(1.0, self._angulo_cuello(lms) / 40.0)
        score = sc * 0.6 + sn * 0.4
        if score > 0.45:
            return "postura_encorvada", round(min(1.0, 0.5 + score * 0.5), 2)
        return "postura_correcta", round(min(1.0, 0.5 + (1 - score) * 0.5), 2)

    # ── Etiquetas ─────────────────────────────────────────────

    def _dibujar_etiquetas(self, frame, x1, y1, car: CaracteristicasFisicas):
        fuente, escala, grosor = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        metodo_tag = f"[{car.metodo_contextura}]"
        items = [
            (f"{car.contextura} {car.confianza_contextura:.0%} {metodo_tag}",
             self.COLORES.get(car.contextura, (200, 200, 200))),
            (f"{car.postura} {car.confianza_postura:.0%}",
             self.COLORES.get(car.postura, (200, 200, 200))),
        ]
        for i, (txt, col) in enumerate(items):
            (tw, th), _ = cv2.getTextSize(txt, fuente, escala, grosor)
            by = max(y1 - 10 - i * (th + 8), th + 4)
            ov = frame.copy()
            cv2.rectangle(ov, (x1, by - th - 4), (x1 + tw + 6, by + 2), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, txt, (x1 + 3, by - 2),
                        fuente, escala, col, grosor, cv2.LINE_AA)

    # ── Pipeline principal ────────────────────────────────────

    def analizar_frame(self, frame: np.ndarray):
        fh, fw = frame.shape[:2]
        yolo_res = self.yolo(frame, classes=[0], conf=self.CONF_YOLO, verbose=False)
        detecciones: list[CaracteristicasFisicas] = []

        for res in yolo_res:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                mx = int((x2 - x1) * self.MARGEN_ROI)
                my = int((y2 - y1) * self.MARGEN_ROI)
                rx1 = max(0, x1 - mx);  ry1 = max(0, y1 - my)
                rx2 = min(fw, x2 + mx); ry2 = min(fh, y2 + my)

                recorte = frame[ry1:ry2, rx1:rx2].copy()
                if recorte.size == 0:
                    continue
                rh, rw = recorte.shape[:2]

                # ── MediaPipe pose ──
                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB),
                )
                result = self.landmarker.detect(mp_img)

                if not result.pose_landmarks:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
                    cv2.putText(frame, "Sin pose", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    continue

                lms = result.pose_landmarks[0]

                # ── Contextura: CNN si está disponible, geométrico si no ──
                if self.cnn_pack is not None:
                    contextura, conf_c = predecir_cnn(recorte, self.cnn_pack)
                    metodo = "CNN"
                else:
                    contextura, conf_c = self._contextura_geometrica(
                        lms, (rx1, ry1, rx2, ry2), rw, rh, fw, fh)
                    metodo = "geometrico"

                # ── Postura: siempre geométrica ──
                postura, conf_p = self._clasificar_postura(lms, rw, rh)

                car = CaracteristicasFisicas(
                    contextura=contextura, postura=postura,
                    confianza_contextura=conf_c, confianza_postura=conf_p,
                    metodo_contextura=metodo,
                    bbox=(x1, y1, x2, y2),
                )
                detecciones.append(car)

                dibujar_esqueleto(recorte, lms)
                frame[ry1:ry2, rx1:rx2] = recorte

                color_bbox = self.COLORES.get(contextura, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bbox, 2)
                self._dibujar_etiquetas(frame, x1, y1, car)

        return frame, detecciones

    def cerrar(self):
        self.landmarker.close()


# ─────────────────────────────────────────────────────────────
#  Modos de uso
# ─────────────────────────────────────────────────────────────

def procesar_camara(modelo_yolo="yolov8n.pt", camara=0):
    detector = DetectorFisicoEnfermeria(modelo_yolo)
    cap = cv2.VideoCapture(camara)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {camara}")
    print("[INFO] 'q' salir  |  's' guardar captura")
    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_out, dets = detector.analizar_frame(frame)
            cv2.putText(frame_out, f"Personas: {len(dets)}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("Detector Fisico - Enfermeria", frame_out)
            for d in dets:
                print(f"  [{d.metodo_contextura}] {d.contextura} "
                      f"({d.confianza_contextura:.0%}) | "
                      f"{d.postura} ({d.confianza_postura:.0%})")
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                cv2.imwrite(f"captura_{n:04d}.jpg", frame_out)
                print(f"[INFO] captura_{n:04d}.jpg guardada")
                n += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.cerrar()


def procesar_imagen(ruta: str, modelo_yolo="yolov8n.pt") -> dict:
    detector = DetectorFisicoEnfermeria(modelo_yolo)
    frame = cv2.imread(ruta)
    if frame is None:
        raise FileNotFoundError(f"No se encontró: {ruta}")
    frame_out, dets = detector.analizar_frame(frame)
    base, ext = os.path.splitext(ruta)
    salida = f"{base}_analizado{ext}"
    cv2.imwrite(salida, frame_out)
    print(f"[INFO] Guardado: {salida}")
    resultados = []
    for d in dets:
        resultados.append({
            "contextura": d.contextura, "confianza": d.confianza_contextura,
            "metodo":     d.metodo_contextura,
            "postura":    d.postura,    "confianza_postura": d.confianza_postura,
            "bbox":       d.bbox,
        })
        print(f"  [{d.metodo_contextura}] {d.contextura} "
              f"({d.confianza_contextura:.0%}) | "
              f"{d.postura} ({d.confianza_postura:.0%})")
    detector.cerrar()
    return {"imagen_salida": salida, "detecciones": resultados}


def procesar_video(ruta: str, modelo_yolo="yolov8n.pt", guardar=True):
    detector = DetectorFisicoEnfermeria(modelo_yolo)
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir: {ruta}")
    writer = None
    if guardar:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base, _ = os.path.splitext(ruta)
        sal = f"{base}_analizado.mp4"
        writer = cv2.VideoWriter(sal, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"[INFO] Guardando en: {sal}")
    try:
        n = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_out, _ = detector.analizar_frame(frame)
            if writer:
                writer.write(frame_out)
            cv2.imshow("Video - Enfermeria", frame_out)
            n += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        print(f"[INFO] Frames procesados: {n}")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        detector.cerrar()


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Detector de características físicas — Software de Enfermería")
    parser.add_argument("--modo",   choices=["camara", "imagen", "video"], default="camara")
    parser.add_argument("--fuente", default="0")
    parser.add_argument("--modelo", default="yolov8n.pt")
    args = parser.parse_args()

    if args.modo == "camara":
        procesar_camara(modelo_yolo=args.modelo, camara=int(args.fuente))
    elif args.modo == "imagen":
        procesar_imagen(ruta=args.fuente, modelo_yolo=args.modelo)
    elif args.modo == "video":
        procesar_video(ruta=args.fuente, modelo_yolo=args.modelo)
