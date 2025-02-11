import os
import json
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import matplotlib.pyplot as plt

import dill

# ==============================
# Parámetros
# ==============================
BATCH_SIZE = 16            # Número de teoremas con los que se entrena
EPOCAS = 4                 # Número de épocas
LEARNING_RATE = 5e-4       # Tasa de aprendizaje
DIM_EMBEDDING = 512        # Dimensión de los embeddings
NUM_HEADS = 4              # Número de "heads" en el multi-head attention
NUM_CAPAS = 2              # Número de capas del Transformer
DIM_FFN = 512              # Dimensión del feed-forward network
VENTANA_CONTEXTO = 256     # Longitud máxima de secuencia (se trunca si es mayor)

# ==============================
# Rutas de archivos
# ==============================
DIRECTORIO_PRE = "/home/juan/Demostenes/preprocesamiento"
RUTA_TOKENIZER = os.path.join(DIRECTORIO_PRE, "tokenizer_Demostenes.json")
RUTA_TEOREMAS = os.path.join(DIRECTORIO_PRE, "entrenamiento_base.json")
RUTA_MODELO = "demostenes_prueba.pth"
DIRECTORIO_RES = "/home/juan/Demostenes/resultados"

# ============================================
# Tokens especiales definidos en el tokenizer
# ============================================
TOKENS_ESPECIALES = {
    "unk": "[UNK]",
    "pad": "[PAD]",
    "cls": "[CLS]",
    "sep": "[SEP]",
    "end": " <|findelteorema|>"  # Nota: tiene un espacio inicial
}

# ==============================
# Carga del tokenizer
# ==============================
tokenizer = Tokenizer.from_file(RUTA_TOKENIZER) # El tokenizer fue configurado con el pre-tokenizer Split(" ", behavior="merged_with_next")

ID_PAD = tokenizer.token_to_id(TOKENS_ESPECIALES["pad"])

# ==============================
# Definición del dataset
# ==============================
class TeoremasDataset(Dataset): # Dataset que carga el archivo JSON con la lista de teoremas. Cada entrada se tokeniza usando el tokenizer cargado.

    def __init__(self, ruta_datos, tokenizer, max_length=VENTANA_CONTEXTO):

        with open(ruta_datos, 'r', encoding='utf-8') as f:
            datos = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.muestras = []

        for item in datos:

            texto_teorema = item["teorema"]
            codificacion = tokenizer.encode(texto_teorema)
            ids_tokens = codificacion.ids
            # Trunca la secuencia si excede la longitud máxima
            if len(ids_tokens) > max_length:
                ids_tokens = ids_tokens[:max_length]

            self.muestras.append(ids_tokens)
    
    def __len__(self):
        return len(self.muestras)
    
    def __getitem__(self, indice):
        return torch.tensor(self.muestras[indice], dtype=torch.long)


def agrupa_lote(lote): # Función para agrupar secuencias de longitudes variables. Se realiza padding al máximo largo de la tanda utilizando el token [PAD].

    lote_agrupado = nn.utils.rnn.pad_sequence(lote, batch_first=True, padding_value=ID_PAD)
    mascara_atencion = (lote_agrupado != ID_PAD).long()
    return lote_agrupado, mascara_atencion

# ==============================
# Arquitectura del modelo
# ==============================
class TransformerLM(nn.Module): # Transformer decoder-only

    def __init__(self, tam_vocab, dim_embedding, num_heads, num_capas, dim_ffn, ventana_contexto, dropout=0.1):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(tam_vocab, dim_embedding)
        self.pos_embedding = nn.Embedding(ventana_contexto, dim_embedding)
        self.dropout = nn.Dropout(dropout)
        
        lista_auxiliar = [nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads, dim_feedforward=dim_ffn, dropout=dropout) for _ in range(num_capas)]
        self.layers = nn.ModuleList(lista_auxiliar)
        self.ln = nn.LayerNorm(dim_embedding)
        self.head = nn.Linear(dim_embedding, tam_vocab)
        self.ventana_contexto = ventana_contexto

    def forward(self, x, mascara_atencion=None):
        batch_size, seq_length = x.size()
        posiciones = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.embedding(x) + self.pos_embedding(posiciones)
        x = self.dropout(x)
        # Reordenar para cumplir con el formato requerido (seq_length, batch_size, dim_embedding)
        x = x.transpose(0, 1)
        # Creación de la máscara causal: cada token solo puede atender a posiciones anteriores o iguales
        mascara_causal = torch.triu(torch.full((seq_length, seq_length), float('-inf'), device=x.device),
                                      diagonal=1)
        for capa in self.layers:
            x = capa(x, src_mask=mascara_causal)

        x = x.transpose(0, 1)  # Volvemos a (batch_size, seq_length, dim_embedding)
        x = self.ln(x)
        logits = self.head(x)

        return logits

# =========================================================
# Función de entrenamiento (registrando pérdida por batch)
# =========================================================
def entrena_modelo(modelo, cargador_datos, optimizador, dispositivo, epocas):

    modelo.train()
    funcion_perdida = nn.CrossEntropyLoss(ignore_index=ID_PAD)
    perdidas_por_lote = []  # Lista para almacenar la pérdida de cada lote
    
    for epoca in range(epocas):
        print(f"\n=== Época {epoca+1}/{epocas} ===")

        for indice_lote, (entradas, mascara_atencion) in enumerate(cargador_datos):
            entradas = entradas.to(dispositivo)
            optimizador.zero_grad()
            salidas = modelo(entradas)
            # Para LM causal: la predicción es la secuencia desplazada una posición a la izquierda
            logits = salidas[:, :-1, :].contiguous()   # (batch, seq_length-1, tam_vocab)
            etiquetas = entradas[:, 1:].contiguous()       # (batch, seq_length-1)
            perdida = funcion_perdida(logits.view(-1, logits.size(-1)), etiquetas.view(-1))
            perdida.backward()
            optimizador.step()
            
            valor_perdida = perdida.item()
            perdidas_por_lote.append(valor_perdida)

            if (indice_lote+1) % 100 == 0 or indice_lote == 0:
                print(f"Lote {indice_lote+1}/{len(cargador_datos)} - Pérdida: {valor_perdida:.4f}")

    return perdidas_por_lote

# ============================================
# Función de inferencia (generación de texto)
# ============================================
def genera_continuacion(modelo, tokenizer, entrada, dispositivo, longitud_max_generacion=100, temperatura=1.0):
    """
    Dado un prompt, genera tokens hasta alcanzar el token especial " <|findelteorema|>"
    o hasta generar longitud_max_generacion tokens. Se utiliza muestreo con temperatura.
    La decodificación se realiza manualmente para evitar que se inserten espacios extra.
    """
    modelo.eval()
    codificacion = tokenizer.encode(entrada)
    ids_entrada = codificacion.ids
    generado = ids_entrada.copy()
    
    tensor_entrada = torch.tensor([ids_entrada], dtype=torch.long, device=dispositivo)
    
    with torch.no_grad():
        for _ in range(longitud_max_generacion):

            salidas = modelo(tensor_entrada)
            logits = salidas[0, -1, :] / temperatura
            probs = torch.softmax(logits, dim=-1)
            siguiente_token = torch.multinomial(probs, num_samples=1).item()
            generado.append(siguiente_token)
            tensor_entrada = torch.tensor([generado], dtype=torch.long, device=dispositivo)
            token_str = tokenizer.id_to_token(siguiente_token)

            if token_str == TOKENS_ESPECIALES["end"]:
                break
    
    decodificado = "".join([tokenizer.id_to_token(tid) for tid in generado])

    if decodificado.endswith(TOKENS_ESPECIALES["end"]):
        decodificado = decodificado[:-len(TOKENS_ESPECIALES["end"])]

    return decodificado

# =====================================
# Función de generación en tiempo real
# =====================================
def genera_continuacion_tiempo_real(modelo, tokenizer, entrada, dispositivo, longitud_max_generacion=100, temperatura=1.0):
    """
    Función que genera tokens a partir de un prompt y muestra cada token en tiempo real en la consola.
    La generación se detiene cuando se genera el token especial de fin (" <|findelteorema|>")
    o se alcanza el límite.
    """
    modelo.eval()
    codificacion = tokenizer.encode(entrada)
    ids_entrada = codificacion.ids
    generado = ids_entrada.copy()
    
    tensor_entrada = torch.tensor([ids_entrada], dtype=torch.long, device=dispositivo)
    
    # Imprime el prompt inicialmente
    print(entrada, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(longitud_max_generacion):

            salidas = modelo(tensor_entrada)
            logits = salidas[0, -1, :] / temperatura
            probs = torch.softmax(logits, dim=-1)
            siguiente_token = torch.multinomial(probs, num_samples=1).item()
            generado.append(siguiente_token)
            tensor_entrada = torch.tensor([generado], dtype=torch.long, device=dispositivo)
            token_str = tokenizer.id_to_token(siguiente_token)

            if token_str == TOKENS_ESPECIALES["end"]:
                break
            # Imprime el token generado en tiempo real (sin salto de línea)
            print(token_str, end="", flush=True)

    print("")  # Salto de línea final
    decodificado = "".join([tokenizer.id_to_token(tid) for tid in generado])

    if decodificado.endswith(TOKENS_ESPECIALES["end"]):
        decodificado = decodificado[:-len(TOKENS_ESPECIALES["end"])]

    return decodificado

# ===========================================================
# Carga de datos, entrenamiento, guardado del modelo y visualización de pérdidas
# ===========================================================
def entrenar_demostenes():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")
    
    # Preparar dataset y cargador de datos
    dataset = TeoremasDataset(RUTA_TEOREMAS, tokenizer, max_length=VENTANA_CONTEXTO)
    cargador_datos = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=agrupa_lote)
    
    # Obtener tamaño del vocabulario (según el tokenizer)
    if hasattr(tokenizer, "get_vocab_size"):
        tam_vocab = tokenizer.get_vocab_size()

    else:
        tam_vocab = len(tokenizer.get_vocab())

    print(f"Tamaño del vocabulario: {tam_vocab}")
    
    # Inicializar el modelo y el optimizador
    modelo = TransformerLM(tam_vocab, DIM_EMBEDDING, NUM_HEADS, NUM_CAPAS, DIM_FFN, VENTANA_CONTEXTO).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LEARNING_RATE)
    
    print("=== Inicio del entrenamiento ===")
    perdidas_por_lote = entrena_modelo(modelo, cargador_datos, optimizador, dispositivo, EPOCAS)
    print("=== Entrenamiento completado ===")
    
    # Guardar el modelo entrenado
    torch.save(modelo.state_dict(), RUTA_MODELO)
    print(f"Modelo guardado en: {RUTA_MODELO}")
    
    # Graficar la pérdida por lote durante el entrenamiento
    plt.figure()
    lotes = range(1, len(perdidas_por_lote) + 1)
    plt.plot(lotes, perdidas_por_lote, linestyle='-')
    plt.xlabel("Lote")
    plt.ylabel("Pérdida")
    plt.title("Pérdida por lote durante el entrenamiento")
    os.makedirs(DIRECTORIO_RES, exist_ok=True)
    ruta_grafico = os.path.join(DIRECTORIO_RES, "perdida_Demostenes.png")
    plt.savefig(ruta_grafico)
    print(f"Gráfico de pérdidas guardado en: {ruta_grafico}")
    plt.show()
    
    # Ejemplo de generación usando la función original
    entrada = "theorem ejemplo: "
    print("\nGeneración usando genera_continuacion:")
    print(entrada)
    continuacion = genera_continuacion(modelo, tokenizer, entrada, dispositivo)
    print("=== Continuación generada ===")
    print(continuacion)
    
    # Ejemplo de generación en tiempo real
    print("\nGeneración en tiempo real usando genera_continuacion_tiempo_real:")
    genera_continuacion_tiempo_real(modelo, tokenizer, entrada, dispositivo)


entrenar_demostenes()

# ======================================================
# Guardado de objetos de inferencia
# ======================================================
objetos_inferencia = {
    "DIM_EMBEDDING": DIM_EMBEDDING,
    "NUM_HEADS": NUM_HEADS,
    "NUM_CAPAS": NUM_CAPAS,
    "DIM_FFN": DIM_FFN,
    "VENTANA_CONTEXTO": VENTANA_CONTEXTO,
    "TOKENS_ESPECIALES": TOKENS_ESPECIALES,
    "RUTA_MODELO" : RUTA_MODELO,
    "tokenizer": tokenizer,
    "TransformerLM": TransformerLM,
    "genera_continuacion": genera_continuacion,
    "genera_continuacion_tiempo_real": genera_continuacion_tiempo_real
}

with open("objetos_inferencia.pkl", "wb") as f:
    dill.dump(objetos_inferencia, f)

print("Se han guardado los objetos de inferencia en 'objetos_inferencia.pkl'.")
