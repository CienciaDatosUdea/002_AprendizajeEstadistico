import json
import re

with open("/home/juan/Demostenes/preprocesamiento/entrenamiento_base.json", "r", encoding="utf-8") as file:
    datos = json.load(file)

# Iterar sobre cada diccionario en la lista
for item in datos:
    if "teorema" in item:
        # Reemplazar los espacios consecutivos (2 o más) por un solo espacio
        item["teorema"] = re.sub(r' {2,}', ' ', item["teorema"])

# Guardar los cambios en el archivo JSON
with open('entrenamiento_base.json', 'w', encoding='utf-8') as archivo:
    json.dump(datos, archivo, ensure_ascii=False, indent=4)

"""
with open('/home/juan/Demostenes/leandojo_benchmark_4/random/train.json', 'r', encoding="utf-8") as file:
    teoremas_entrenamiento = json.load(file)  # Carga la lista de diccionarios


def eliminar_etiquetas(texto): # Elimina las etiquetas <a> y </a> de un string, conservando el texto interno.

    if isinstance(texto, str):

        pattern = r'<a>(.*?)</a>'
        return re.sub(pattern, r'\1', texto)

    else:
        return texto


def unir_tacticas(lista_tacticas : list): # lista_tacticas = teoremas_entrenamiento[indice]['traced_tactics']

    numero_tacticas = len(lista_tacticas)
    tacticas_unidas = ""

    for tactica in lista_tacticas:
        tacticas_unidas = tacticas_unidas + eliminar_etiquetas(tactica['annotated_tactic'][0]) + "\n"

    return tacticas_unidas

indice = 2
"""

"""
lista_tacticas = teoremas_entrenamiento[indice]['traced_tactics']

print(unir_tacticas(lista_tacticas))
print("full_name: ", teoremas_entrenamiento[indice]['full_name'])

"""

"""
claves_tacticas = set()

for tactica in teoremas_entrenamiento[indice]['traced_tactics']:
    claves_tacticas = claves_tacticas | tactica.keys()

print("claves: ", teoremas_entrenamiento[indice].keys())
print("url: ", teoremas_entrenamiento[indice]['url'])
print("file_path: ", teoremas_entrenamiento[indice]['file_path'])
print("commit: ", teoremas_entrenamiento[indice]['commit'])
print("full_name: ", teoremas_entrenamiento[indice]['full_name'])
print("start: ", teoremas_entrenamiento[indice]['start'])
print("end: ", teoremas_entrenamiento[indice]['end'])
print("claves de las tacticas: ", claves_tacticas)
print("")
print("estado_inicial:")
print(teoremas_entrenamiento[indice]['traced_tactics'][0]['state_before'])
print()

estado = 0

ultimo_estado = teoremas_entrenamiento[indice]['traced_tactics'][estado]['state_after']


while ultimo_estado != "no goals":
    print(f"{estado} estado after:")
    print(ultimo_estado)
    print()
    estado = estado + 1
    ultimo_estado = teoremas_entrenamiento[indice]['traced_tactics'][estado]['state_after']


#with open('corpus.json', "r", encoding="utf-8") as file:
    #diccionario = json.load(file)
    #print(diccionario)
"""

"""
for premisa in lista_diccionarios[0]['premises']:
    print(premisa['full_name'])


print(len(lista_diccionarios))
print()

numero_premisas = 0

for linea in lista_diccionarios:
    premisas_linea = len(linea['premises']) # lista_diccionarios[i]['premises'] es una lista de premisas
    numero_premisas = numero_premisas + premisas_linea

print(numero_premisas)


#print(teoremas_entrenamiento[7])
teorema = teoremas_entrenamiento[7]
print(print(teorema['file_path']))
print()
print(teorema['traced_tactics'][0]['state_before'])
print()

for tactica in teorema['traced_tactics']:
    print(tactica['tactic'])
"""