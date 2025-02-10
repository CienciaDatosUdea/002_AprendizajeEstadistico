import json
import random

# Abrir y cargar el contenido del archivo JSON
with open('corpus_total.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Separar el string en palabras
palabras = data["corpus"].split()

# Mezclar aleatoriamente las palabras
random.shuffle(palabras)

# Volver a unir las palabras en un string
data["corpus"] = " ".join(palabras)

# Guardar el diccionario modificado en el archivo JSON
with open('corpus_total.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

print("Archivo modificado exitosamente.")
