# 🏥 Detector de Características Físicas — Software de Enfermería

Detecta automáticamente **contextura corporal** (delgado, contextura media, sobrepeso, obesidad)
y **postura** (correcta, encorvada) usando YOLOv8 + MediaPipe + MobileNetV2.

---

## ⚙️ Requisitos del sistema

| Componente | Versión recomendada |
|---|---|
| Python | 3.9 / 3.10 / 3.11 / 3.12 (**NO** 3.13) |
| Sistema operativo | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| RAM mínima | 4 GB |
| Cámara | Webcam USB o integrada |

---

## 📦 Instalación paso a paso

### 1. Crear entorno virtual (recomendado)

```bash
# Crear entorno
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux / macOS
source venv/bin/activate
```

---

### 2. Instalar dependencias principales

```bash
# OpenCV CON interfaz gráfica (importante: NO instalar headless)
pip install opencv-python

# YOLO (detección de personas)
pip install ultralytics

# MediaPipe Tasks API (pose landmarks)
pip install "mediapipe>=0.10.14"

# PyTorch (clasificador CNN de contextura)
pip install torch torchvision

# Utilidades
pip install numpy pillow matplotlib
```

> ⚠️ Si ya tienes `opencv-python-headless` instalado, reemplázalo:
> ```bash
> pip uninstall opencv-python-headless -y
> pip install opencv-python
> ```

---

### 3. Instalar dependencias para descarga del dataset (solo una vez)

```bash
pip install roboflow
```

---

### 4. Verificar instalación

```bash
python -c "import cv2; import mediapipe; import ultralytics; import torch; print('Todo OK')"
```

---

## 📁 Archivos del proyecto

```
proyecto/
├── detector_fisico_enfermeria.py   ← detector principal
├── reorganizar_y_entrenar.py       ← entrena el modelo con el dataset
├── modelo_contextura.pth           ← modelo entrenado (se genera al entrenar)
├── clases_contextura.json          ← clases del modelo (se genera al entrenar)
└── pose_landmarker_lite.task       ← modelo MediaPipe (se descarga automático)
```

---

## 🚀 Uso

### Ejecutar el detector (cámara en tiempo real)
```bash
python detector_fisico_enfermeria.py --modo camara
```

### Analizar una imagen
```bash
python detector_fisico_enfermeria.py --modo imagen --fuente foto.jpg
```

### Analizar un video
```bash
python detector_fisico_enfermeria.py --modo video --fuente video.mp4
```

### Cambiar cámara (si tienes varias)
```bash
python detector_fisico_enfermeria.py --modo camara --fuente 1
```

### Usar modelo YOLO más preciso (más lento)
```bash
python detector_fisico_enfermeria.py --modo camara --modelo yolov8s.pt
```

---

## 🔁 Re-entrenar el modelo

Si quieres mejorar la precisión con más épocas:
```bash
python reorganizar_y_entrenar.py --epocas 40 --solo-entrenar
```

---

## ❗ Errores comunes

| Error | Solución |
|---|---|
| `No module named mediapipe.framework` | Versión antigua de mediapipe. Ejecuta: `pip install "mediapipe>=0.10.14"` |
| `No module named mediapipe.python` | Igual que arriba |
| `The function is not implemented` en OpenCV | Tienes headless. Ejecuta: `pip uninstall opencv-python-headless -y && pip install opencv-python` |
| `No se pudo abrir la cámara 0` | Cambia el índice: `--fuente 1` |
| Python 3.13 no compatible | Usa Python 3.10 o 3.11 |

---

## 📋 requirements.txt

```
opencv-python>=4.8.0
ultralytics>=8.0.0
mediapipe>=0.10.14
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.0.0
matplotlib>=3.7.0
roboflow>=1.0.0
```
