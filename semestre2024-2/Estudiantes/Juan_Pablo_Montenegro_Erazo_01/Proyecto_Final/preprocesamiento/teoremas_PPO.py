import json
import re
import random
from pathlib import Path

# 1) Lee train.json (LeanDojo).
# 2) Omite los teoremas que no tienen demostración. Asumimos que "no demostración"
#    significa que "traced_tactics" está vacío (es decir, len(traced_tactics) == 0).
# 3) Para cada teorema con demostración, extraemos "full_name", "file_path" y construimos
#    la clave "teorema" como antes (con theorem/lemma aleatorio, variables detectadas y la meta).
# 4) Se guarda el resultado en "entrenamiento_PPO.json".

ruta_train_json = Path("/home/juan/Demostenes/leandojo_benchmark_4/random/val.json")
ruta_salida = Path("validacion_PPO.json")

def extraer_tipo_de_linea(linea_var: str) -> str:
    """
    Dada una línea como 'n : ℕ' o 'p₁ p₂ : P',
    retorna la parte derecha tras ':', por ejemplo 'ℕ' o 'P'.
    """
    parts = linea_var.split(":", 1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()

def detectar_variables_en_tipo(var_type: str, lista_variables: list) -> set:
    """
    Dado un 'var_type' (ej. 'l ⊕ m → α') y una lista de posibles variables (['l','m','α']),
    busca cuáles aparecen en var_type usando regex de palabra completa.
    """
    encontradas = set()
    for candidate in lista_variables:
        if re.search(rf"\b{re.escape(candidate)}\b", var_type):
            encontradas.add(candidate)
    return encontradas

def recolectar_meta_completa(lineas, indice_inicial) -> str:
    """
    Desde la línea 'indice_inicial' (que contiene '⊢'), vamos acumulando
    todas las líneas que pertenezcan a la meta hasta que hallemos una línea que
    indique final (vacía, 'case...', 'no goals', o nueva definición de variable).
    Unimos y removemos '⊢'.
    """
    meta_lineas = []
    n = len(lineas)
    pos = indice_inicial

    while pos < n:
        line_strip = lineas[pos].rstrip("\n")
        if not line_strip.strip():
            break
        if line_strip.strip().startswith("case"):
            break
        if "no goals" in line_strip:
            break
        # Si no es la primera línea, detecta algo tipo "x y : ℕ"
        if pos != indice_inicial:
            if re.match(r"^\s*\S+\s+.*:.*$", line_strip):
                break

        meta_lineas.append(line_strip)
        pos += 1

    meta_unida = " ".join(meta_lineas)
    meta_unida = meta_unida.replace("⊢", "").strip()
    return meta_unida

# Cargar train.json
with open(ruta_train_json, "r", encoding="utf-8") as file:
    teoremas_entrenamiento = json.load(file)

resultado = []

for entry in teoremas_entrenamiento:
    # Excluir teoremas sin demostración: traced_tactics vacío
    traced_tactics = entry.get("traced_tactics", [])
    if not traced_tactics:
        continue  # Este teorema no tiene demostración, lo omitimos

    full_name = entry.get("full_name", "UnknownName")
    file_path = entry.get("file_path", "UnknownFile")

    # Al azar elegimos 'theorem' o 'lemma'
    palabra_clave = random.choice(["theorem", "lemma"])

    # Obtener el primer trazo
    primer_trazo = traced_tactics[0]
    state_before = primer_trazo.get("state_before", "")

    # Dividimos en líneas y buscamos la línea con '⊢'
    lineas = state_before.split("\n")
    indice_enunciado = None
    for i, linea in enumerate(lineas):
        if "⊢" in linea:
            indice_enunciado = i
            break

    if indice_enunciado is None:
        enunciado_lean = "/* no se encontró ⊢ en state_before */"
        indice_enunciado = len(lineas)
    else:
        enunciado_lean = recolectar_meta_completa(lineas, indice_enunciado)

    # Parsear las variables definidas antes de la meta
    variables_contexto = {}
    for linea in lineas[:indice_enunciado]:
        linea = linea.strip()
        if not linea:
            continue
        if ":" in linea:
            left_part, right_part = linea.split(":", 1)
            left_part = left_part.strip()
            right_part = right_part.strip()
            var_names = left_part.split()
            for v_name in var_names:
                variables_contexto[v_name] = f"{v_name} : {right_part}"

    # Detectar las variables que aparecen en el enunciado
    usadas_directo = set()
    for var_name in variables_contexto:
        patron = rf"\b{re.escape(var_name)}\b"
        if re.search(patron, enunciado_lean):
            usadas_directo.add(var_name)

    # Construir grafo de dependencias var -> {vars en su tipo}
    dependencias = {v: set() for v in variables_contexto}
    todos_los_vars = list(variables_contexto.keys())
    for var_name, definicion_linea in variables_contexto.items():
        var_type = extraer_tipo_de_linea(definicion_linea)
        refs = detectar_variables_en_tipo(var_type, todos_los_vars)
        dependencias[var_name] = refs

    # Cierre transitivo
    usados_final = set()
    cola = list(usadas_directo)
    while cola:
        actual = cola.pop()
        if actual not in usados_final:
            usados_final.add(actual)
            for y in dependencias[actual]:
                if y not in usados_final:
                    cola.append(y)

    # Construir "(var : tipo)" para esas variables
    variables_str_list = []
    for var_name in sorted(usados_final):
        definicion_linea = variables_contexto[var_name]
        left, right = definicion_linea.split(":", 1)
        left, right = left.strip(), right.strip()
        variables_str_list.append(f"({left} : {right})")

    variables_str = ""
    if variables_str_list:
        variables_str = " " + " ".join(variables_str_list)

    # Construir el string final
    teorema_str = f"{palabra_clave} {full_name}{variables_str} : {enunciado_lean} := by\n"

    # Crear la entrada con las claves solicitadas
    teorema_filtrado = {
        "full_name": full_name,
        "file_path": file_path,
        "teorema": teorema_str
    }

    resultado.append(teorema_filtrado)

# Guardar el nuevo JSON
with open(ruta_salida, "w", encoding="utf-8") as f_out:
    json.dump(resultado, f_out, indent=2, ensure_ascii=False)

print(f"¡Listo! Se generó el archivo", ruta_salida, "con", len(resultado), "teoremas con demostración.")
