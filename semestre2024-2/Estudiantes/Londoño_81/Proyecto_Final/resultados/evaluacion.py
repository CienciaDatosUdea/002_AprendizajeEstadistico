import json
from lean_dojo import *

import os
import torch
import dill
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Cargar el diccionario con los objetos de inferencia
with open("/home/juan/Demostenes/Demostenes/objetos_inferencia.pkl", "rb") as f:
    objetos_inferencia = dill.load(f)

# Extraer las variables, clases y funciones necesarias
DIM_EMBEDDING = objetos_inferencia["DIM_EMBEDDING"]
NUM_HEADS = objetos_inferencia["NUM_HEADS"]
NUM_CAPAS = objetos_inferencia["NUM_CAPAS"]
DIM_FFN = objetos_inferencia["DIM_FFN"]
VENTANA_CONTEXTO = objetos_inferencia["VENTANA_CONTEXTO"]
TOKENS_ESPECIALES = objetos_inferencia["TOKENS_ESPECIALES"]
#RUTA_MODELO = objetos_inferencia["RUTA_MODELO"]
tokenizer = objetos_inferencia["tokenizer"]
TransformerLM = objetos_inferencia["TransformerLM"]
genera_continuacion = objetos_inferencia["genera_continuacion"]
genera_continuacion_tiempo_real = objetos_inferencia["genera_continuacion_tiempo_real"]

RUTA_MODELO = "/home/juan/Demostenes/Demostenes/demostenes_prueba.pth"

print("Objetos de inferencia cargados correctamente.")


def correr_demostenes(entrada = "theorem ejemplo: ", tiempo_real = False, longitud_max = 156, temperatura = 0.5):
    """
    Ejecuta la inferencia usando el modelo entrenado.
    
    Parámetros:
      - entrada: cadena de texto con el prompt de entrada.
      - tiempo_real: si es True, muestra la generación token a token en tiempo real.
      - longitud_max: número máximo de tokens a generar.
      - temperatura: temperatura para el muestreo.
      
    Devuelve la secuencia generada.
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtener el tamaño del vocabulario a partir del tokenizer
    if hasattr(tokenizer, "get_vocab_size"):
        tam_vocab = tokenizer.get_vocab_size()

    else:
        tam_vocab = len(tokenizer.get_vocab())

    # Inicializar el modelo con la misma arquitectura usada durante el entrenamiento
    modelo = TransformerLM(tam_vocab, DIM_EMBEDDING, NUM_HEADS, NUM_CAPAS, DIM_FFN, VENTANA_CONTEXTO)
    modelo.to(dispositivo)

    # Cargar los pesos entrenados
    if os.path.exists(RUTA_MODELO):
        estado_modelo = torch.load(RUTA_MODELO, map_location=dispositivo)
        modelo.load_state_dict(estado_modelo)

    else:
        print(f"No se encontró el archivo {RUTA_MODELO}. Verifica la ruta.")
        return None

    # Realizar la generación según el modo seleccionado
    if tiempo_real:
        texto_generado = genera_continuacion_tiempo_real(modelo, tokenizer, entrada, dispositivo,
                                                         longitud_max_generacion=longitud_max,
                                                         temperatura=temperatura)
    else:
        texto_generado = genera_continuacion(modelo, tokenizer, entrada, dispositivo,
                                             longitud_max_generacion=longitud_max,
                                             temperatura=temperatura)

    return texto_generado




with open("/home/juan/Demostenes/resultados/prueba_PPO.json", "r", encoding="utf-8") as file:
    teoremas_prueba = json.load(file)


repo = LeanGitRepo("https://github.com/leanprover-community/mathlib4", "29dcec074de168ac2bf835a77ef68bbe069194c5")

lineas_correctas = 0

#print(len(teoremas_prueba))
for indice in range(len(teoremas_prueba)):

	if indice % 10 == 0:
		print(indice)
		print(f"Lineas lineas_correctas: {lineas_correctas}")

	full_name = teoremas_prueba[indice]["full_name"]
	file_path = teoremas_prueba[indice]["file_path"]
	teorema = teoremas_prueba[indice]["teorema"]

	demos = correr_demostenes(entrada = f"{teorema}\n", tiempo_real=False, longitud_max=156, temperatura=0.5)
	lineas = demos.splitlines()

	#theorem = Theorem(repo, "Mathlib/Data/Matrix/Block.lean", "Matrix.toBlocks₁₁_diagonal")
	theorem = Theorem(repo, file_path, full_name)


	with Dojo(theorem) as (dojo, init_state):

	  estado_previo = init_state

	  for linea in lineas:
	  	siguiente_estado = dojo.run_tac(estado_previo, linea)

	  	if isinstance(siguiente_estado, LeanError) == False:
	  		lineas_correctas = lineas_correctas + 1
	  		estado_previo = siguiente_estado

	  	else:
	  		break


print(f"Lineas lineas_correctas: {lineas_correctas}")