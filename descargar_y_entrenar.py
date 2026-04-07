"""
Reorganizar dataset Roboflow (formato multiclass CSV) y entrenar clasificador
=============================================================================
Lee los _classes.csv con formato one-hot y organiza las imágenes en carpetas
por clase, luego entrena MobileNetV2.

USO:
    python reorganizar_y_entrenar.py

INSTALACIÓN:
    pip install torch torchvision pillow matplotlib

El dataset debe estar en ./roboflow_raw/ (ya descargado).
"""

import os
import csv
import shutil
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import matplotlib.pyplot as plt


# ── Mapeo de columnas CSV → tus 4 categorías ─────────────────
MAPEO = {
    "Under-weight":     "delgado",
    "Normal-weight":    "contextura_media",
    "Overweight":       "sobrepeso",
    "Mild-obesity":     "obesidad",
    "Moderate-obesity": "obesidad",
    "Severe-obesity":   "obesidad",
}

TUS_CLASES = ["delgado", "contextura_media", "sobrepeso", "obesidad"]
IMG_SIZE   = 224
BATCH      = 32


# ─────────────────────────────────────────────────────────────
#  PASO 1 — Leer CSV one-hot y reorganizar imágenes en carpetas
# ─────────────────────────────────────────────────────────────

def reorganizar(ruta_raw: str = "roboflow_raw", ruta_salida: str = "dataset"):
    print("=" * 55)
    print("  PASO 1: Reorganizando imágenes por clase")
    print("=" * 55)

    # Roboflow usa train/ valid/ — mapeamos valid → val
    splits = {
        "train": "train",
        "valid": "val",
    }

    conteo     = {c: {"train": 0, "val": 0} for c in TUS_CLASES}
    sin_clase  = 0

    for split_origen, split_destino in splits.items():
        ruta_split = os.path.join(ruta_raw, split_origen)
        ruta_csv   = os.path.join(ruta_split, "_classes.csv")

        if not os.path.exists(ruta_csv):
            print(f"  [AVISO] No encontrado: {ruta_csv}")
            continue

        with open(ruta_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for fila in reader:
                filename = fila["filename"].strip()
                ruta_img = os.path.join(ruta_split, filename)

                if not os.path.exists(ruta_img):
                    continue

                # Determinar clase por columna con valor "1"
                clase_csv = None
                for col, val in fila.items():
                    if col == "filename":
                        continue
                    if val.strip() == "1":
                        clase_csv = col
                        break

                # Ignorar imágenes sin clase (todos en 0)
                if clase_csv is None:
                    sin_clase += 1
                    continue

                clase_dest = MAPEO.get(clase_csv)
                if clase_dest is None:
                    print(f"  [AVISO] Clase no mapeada: '{clase_csv}'")
                    continue

                # Copiar imagen a dataset/<split>/<clase>/
                carpeta_dest = Path(ruta_salida) / split_destino / clase_dest
                carpeta_dest.mkdir(parents=True, exist_ok=True)

                destino = carpeta_dest / filename
                if not destino.exists():
                    shutil.copy2(ruta_img, destino)

                conteo[clase_dest][split_destino] += 1

    # Resumen
    print(f"\n  {'Clase':<20} {'Train':>8} {'Val':>8}")
    print("  " + "-" * 38)
    total_train, total_val = 0, 0
    for clase in TUS_CLASES:
        t = conteo[clase]["train"]
        v = conteo[clase]["val"]
        total_train += t
        total_val   += v
        print(f"  {clase:<20} {t:>8} {v:>8}")
    print("  " + "-" * 38)
    print(f"  {'TOTAL':<20} {total_train:>8} {total_val:>8}")
    print(f"\n  Imágenes sin clase (ignoradas): {sin_clase}")
    print(f"\n  ✓ Dataset listo en: {ruta_salida}/")


# ─────────────────────────────────────────────────────────────
#  PASO 2 — Entrenar MobileNetV2
# ─────────────────────────────────────────────────────────────

def crear_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def construir_modelo(num_clases: int, device):
    modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in modelo.features.parameters():
        param.requires_grad = False
    in_features = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_clases),
    )
    return modelo.to(device)


def entrenar(ruta_dataset: str, epocas: int, ruta_salida: str, device):
    print("\n" + "=" * 55)
    print("  PASO 2: Entrenando clasificador")
    print("=" * 55)

    train_tf, val_tf = crear_transforms()
    train_ds = datasets.ImageFolder(
        os.path.join(ruta_dataset, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(
        os.path.join(ruta_dataset, "val"),   transform=val_tf)

    print(f"\n  Clases detectadas: {train_ds.classes}")
    print(f"  Train: {len(train_ds)} imágenes  |  Val: {len(val_ds)} imágenes\n")

    # Guardar mapeo de clases
    with open("clases_contextura.json", "w", encoding="utf-8") as f:
        json.dump(train_ds.classes, f, ensure_ascii=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    modelo      = construir_modelo(len(train_ds.classes), device)
    criterio    = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()), lr=1e-3)
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizador, step_size=7, gamma=0.5)

    historial = {"train_loss": [], "val_loss": [], "val_acc": []}
    mejor_acc = 0.0

    for epoca in range(1, epocas + 1):
        # Entrenamiento
        modelo.train()
        loss_train = 0.0
        for imgs, etiquetas in train_loader:
            imgs, etiquetas = imgs.to(device), etiquetas.to(device)
            optimizador.zero_grad()
            loss = criterio(modelo(imgs), etiquetas)
            loss.backward()
            optimizador.step()
            loss_train += loss.item() * imgs.size(0)
        loss_train /= len(train_loader.dataset)

        # Validación
        modelo.eval()
        loss_val, correctos = 0.0, 0
        with torch.no_grad():
            for imgs, etiquetas in val_loader:
                imgs, etiquetas = imgs.to(device), etiquetas.to(device)
                salidas  = modelo(imgs)
                loss_val += criterio(salidas, etiquetas).item() * imgs.size(0)
                correctos += (salidas.argmax(1) == etiquetas).sum().item()
        loss_val /= len(val_loader.dataset)
        acc = correctos / len(val_loader.dataset)

        historial["train_loss"].append(loss_train)
        historial["val_loss"].append(loss_val)
        historial["val_acc"].append(acc)

        marca = " ← MEJOR" if acc > mejor_acc else ""
        print(f"  Época {epoca:>2}/{epocas}  "
              f"loss={loss_train:.4f}  val_loss={loss_val:.4f}  "
              f"acc={acc:.2%}{marca}")

        if acc > mejor_acc:
            mejor_acc = acc
            torch.save(modelo.state_dict(), ruta_salida)

        scheduler.step()

    print(f"\n  ✓ Mejor accuracy: {mejor_acc:.2%}")
    print(f"  ✓ Modelo guardado: {ruta_salida}")
    print(f"  ✓ Clases guardadas: clases_contextura.json")
    return historial


def graficar(historial):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(historial["train_loss"], label="Train")
    ax1.plot(historial["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Época"); ax1.legend(); ax1.grid(True)
    ax2.plot(historial["val_acc"], color="green")
    ax2.set_title("Accuracy validación"); ax2.set_xlabel("Época"); ax2.grid(True)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    plt.savefig("entrenamiento_resultado.png", dpi=120)
    print("  ✓ Gráfica guardada: entrenamiento_resultado.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas",       type=int, default=20)
    parser.add_argument("--salida",       default="modelo_contextura.pth")
    parser.add_argument("--raw",          default="roboflow_raw",
                        help="Carpeta donde está el dataset descargado")
    parser.add_argument("--solo-entrenar", action="store_true",
                        help="Saltar reorganización y usar ./dataset ya existente")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    if not args.solo_entrenar:
        reorganizar(ruta_raw=args.raw, ruta_salida="dataset")

    historial = entrenar("dataset", args.epocas, args.salida, device)
    graficar(historial)

    print("\n" + "=" * 55)
    print("  LISTO. Siguiente paso:")
    print("  python detector_fisico_enfermeria.py --modo camara")
    print("=" * 55)