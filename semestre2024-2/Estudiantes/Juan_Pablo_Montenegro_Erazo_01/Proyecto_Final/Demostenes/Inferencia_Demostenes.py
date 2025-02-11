import os
import torch
import dill
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Cargar el diccionario con los objetos de inferencia
with open("objetos_inferencia.pkl", "rb") as f:
    objetos_inferencia = dill.load(f)

# Extraer las variables, clases y funciones necesarias
DIM_EMBEDDING = objetos_inferencia["DIM_EMBEDDING"]
NUM_HEADS = objetos_inferencia["NUM_HEADS"]
NUM_CAPAS = objetos_inferencia["NUM_CAPAS"]
DIM_FFN = objetos_inferencia["DIM_FFN"]
VENTANA_CONTEXTO = objetos_inferencia["VENTANA_CONTEXTO"]
TOKENS_ESPECIALES = objetos_inferencia["TOKENS_ESPECIALES"]
RUTA_MODELO = objetos_inferencia["RUTA_MODELO"]
tokenizer = objetos_inferencia["tokenizer"]
TransformerLM = objetos_inferencia["TransformerLM"]
genera_continuacion = objetos_inferencia["genera_continuacion"]
genera_continuacion_tiempo_real = objetos_inferencia["genera_continuacion_tiempo_real"]

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
    print(f"Usando dispositivo: {dispositivo}")

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
        print(f"Modelo cargado desde: {RUTA_MODELO}")
    else:
        print(f"No se encontró el archivo {RUTA_MODELO}. Verifica la ruta.")
        return None

    # Realizar la generación según el modo seleccionado
    if tiempo_real:
        print("\nGeneración en tiempo real:")
        texto_generado = genera_continuacion_tiempo_real(modelo, tokenizer, entrada, dispositivo,
                                                         longitud_max_generacion=longitud_max,
                                                         temperatura=temperatura)
    else:
        print("\nGeneración usando genera_continuacion:")
        texto_generado = genera_continuacion(modelo, tokenizer, entrada, dispositivo,
                                             longitud_max_generacion=longitud_max,
                                             temperatura=temperatura)
        print("\n=== Continuación generada ===")
        print(texto_generado)

    return texto_generado


entrada = "theorem union_comm {α : Type _} (A B : Set α) : A ∪ B = B ∪ A := by\n"
tiempo_real = False  # Generación en tiempo real
longitud_max = 100
temperatura = 1.0

correr_demostenes(entrada=entrada, tiempo_real=tiempo_real, longitud_max=longitud_max, temperatura=temperatura)
