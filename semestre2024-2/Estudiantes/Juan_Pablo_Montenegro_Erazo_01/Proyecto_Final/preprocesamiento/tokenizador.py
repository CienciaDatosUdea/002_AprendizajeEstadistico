import json
import random

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import Split

lista_especiales = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", " <|findelteorema|>"]


with open("/home/juan/Demostenes/preprocesamiento/corpus_premisas.json", "r", encoding="utf-8") as file:
    corpus_premisas = json.load(file)


corpus_premisas = corpus_premisas["corpus"]

lista_palabras = corpus_premisas.split() # 1️) Dividir el texto en palabras
random.shuffle(lista_palabras) # 2️) Mezclar aleatoriamente las palabras
corpus_premisas = " ".join(lista_palabras) # 3️) Volver a unir las palabras en un solo string
corpus_premisas = [corpus_premisas]

tokenizer = Tokenizer(BPE(unk_token="[UNK]")) # Crear tokenizer BPE con un token desconocido [UNK]
tokenizer.pre_tokenizer = Split(" ", behavior="merged_with_next") # Pre-tokenizador para que pueda juntar simbolos

trainer_1 = BpeTrainer(vocab_size=10000, special_tokens=lista_especiales, initial_alphabet=["\n"]) # Entrenador para la segunda base con vocabulario de 5000 tokens
tokenizer.train_from_iterator(corpus_premisas, trainer_1) # Entrenar BPE con la segunda base

tokenizer.save("tokenizer_Demostenes.json") # Guardar el tokenizer entrenado con 5000 tokens
print("Finalizó la primera fase")

# -------------------------------------------------------------------


with open("/home/juan/Demostenes/preprocesamiento/entrenamiento_base.json", "r", encoding="utf-8") as file:
    teoremas_entrenamiento = json.load(file)

print("comenzó la segunda fase")

teoremas_entrenamiento = [diccionario["teorema"] for diccionario in teoremas_entrenamiento]

nueva_lista = []
contador = 0

for teorema in teoremas_entrenamiento:
    nueva_lista = nueva_lista + teorema.splitlines(keepends=True) # Separamos por \n
    contador = contador + 1

    if contador == 2:
        print(nueva_lista)

    if contador % 1000 == 0:
        print(contador)


teoremas_entrenamiento = nueva_lista
random.shuffle(teoremas_entrenamiento) # 2️) Mezclar aleatoriamente los teoremas
print(teoremas_entrenamiento[:2])

tokenizer = Tokenizer.from_file("tokenizer_Demostenes.json") # Cargar el tokenizer entrenado
tokenizer.pre_tokenizer = Split(" ", behavior="merged_with_next") # Pre-tokenizador para que pueda juntar simbolos

#trainer_2 = trainers.BpeTrainer(vocab_size=16000, special_tokens = lista_especiales)
trainer_2 = BpeTrainer(vocab_size = 16000, special_tokens = lista_especiales)

tokenizer.train_from_iterator(teoremas_entrenamiento, trainer_2) # Continúa el entrenamiento con el nuevo corpus.

tokenizer.save("tokenizer_Demostenes.json") # Guarda el tokenizer resultante.


"""
# Obtener el vocabulario completo
vocab = tokenizer.get_vocab()  # Devuelve un diccionario {token_string: token_id}

# Ordenar el vocabulario por ID
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

# Mostrar los primeros 100 tokens
print("🔹 ID  |  Token")
print("-------------------")
for token, token_id in sorted_vocab[100:200]:  # Tomar solo los primeros 100

    if " " in token or "." in token:
        print(f"{token_id:3d} |{token}")

    if token_id == len(sorted_vocab) - 1:
        print(f"El ultimo token es: {token}")


print(f"Hay {len(sorted_vocab)} tokens")
"""

tokenizer = Tokenizer.from_file("tokenizer_Demostenes.json") # Cargar el tokenizer entrenado

# Lista de frases de prueba
test_sentences = ["intro x left Or.inl Or.inr or.inl by cases Nat.le_of_not_gt",
    "intro x left\n Or.inl Or.inr \n or.inl by cases\n",
    "Este tokenizador fue entrenado con BPE en dos fases."]

# Tokenizar cada frase y mostrar resultados
for sentence in test_sentences:
    encoded = tokenizer.encode(sentence)
    print(f"Texto: {sentence}")
    print(f"Token IDs: {encoded.ids}")
    print(f"Tokens: {encoded.tokens}")

    # Decodificación de una secuencia de tokens
    decoded_text = tokenizer.decode(encoded.ids)
    print(f"Texto reconstruido: {decoded_text}\n")