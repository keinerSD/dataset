"""
Ver qué clases hay en el _classes.csv de Roboflow
"""
import csv, os

for split in ["train", "valid", "test"]:
    ruta_csv = os.path.join("roboflow_raw", split, "_classes.csv")
    if not os.path.exists(ruta_csv):
        print(f"[NO ENCONTRADO] {ruta_csv}")
        continue

    print(f"\n=== {split}/_classes.csv (primeras 10 filas) ===")
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            print(row)
            if i >= 9:
                break

    # Contar valores únicos en cada columna
    print(f"\n  Columnas únicas:")
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {}
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, set()).add(v.strip())
    for k, vs in cols.items():
        print(f"    '{k}': {sorted(vs)[:20]}")
