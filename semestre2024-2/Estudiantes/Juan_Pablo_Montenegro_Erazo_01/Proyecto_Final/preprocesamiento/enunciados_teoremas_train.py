import json

# Abrir y cargar el archivo JSON
with open("corpus_teoremas.json", "r", encoding="utf-8") as archivo:
    teoremas = json.load(archivo)

# Filtrar los teoremas que tengan demostración
# Se descartan aquellos en los que el string de "teorema" contenga "No hay demo"
teoremas_filtrados = [t for t in teoremas if "No hay demo" not in t.get("teorema", "")]

# Guardar el dataset modificado en el mismo archivo o en uno nuevo
with open("corpus_teoremas.json", "w", encoding="utf-8") as archivo:
    json.dump(teoremas_filtrados, archivo, indent=2, ensure_ascii=False)

print(f"Se han guardado {len(teoremas_filtrados)} teoremas con demostración.")
