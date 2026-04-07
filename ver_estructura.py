"""
Diagnóstico: muestra la estructura exacta del dataset descargado
Ejecuta este script y comparte el resultado.
"""
import os

def mostrar_estructura(ruta, nivel=0, max_archivos=5):
    if not os.path.exists(ruta):
        print(f"[ERROR] No existe: {ruta}")
        return
    
    items = sorted(os.listdir(ruta))
    carpetas = [i for i in items if os.path.isdir(os.path.join(ruta, i))]
    archivos = [i for i in items if os.path.isfile(os.path.join(ruta, i))]
    
    indent = "  " * nivel
    
    for carpeta in carpetas:
        print(f"{indent}📁 {carpeta}/")
        mostrar_estructura(os.path.join(ruta, carpeta), nivel + 1, max_archivos)
    
    # Mostrar solo los primeros N archivos para no saturar
    for archivo in archivos[:max_archivos]:
        print(f"{indent}📄 {archivo}")
    if len(archivos) > max_archivos:
        print(f"{indent}   ... y {len(archivos) - max_archivos} archivos más")

print("=" * 60)
print("ESTRUCTURA DE roboflow_raw:")
print("=" * 60)
mostrar_estructura("roboflow_raw", max_archivos=3)

print("\n" + "=" * 60)
print("NOMBRES EXACTOS DE CARPETAS (para mapeo de clases):")
print("=" * 60)
for raiz, carpetas, archivos in os.walk("roboflow_raw"):
    nivel = raiz.replace("roboflow_raw", "").count(os.sep)
    if nivel <= 2:
        indent = "  " * nivel
        n_imgs = len([f for f in archivos if f.lower().endswith(
            ('.jpg','.jpeg','.png','.webp','.bmp'))])
        if n_imgs > 0 or not archivos:
            print(f"{indent}{os.path.basename(raiz)}/ "
                  f"{'(' + str(n_imgs) + ' imágenes)' if n_imgs else ''}")
