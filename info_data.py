import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from VOCAB2 import load_vocab

# --------- NUM OF UNK -----------

# Cargar el archivo CSV
df = pd.read_csv('cleaned_dataset.csv')

# Convertir las cadenas de texto en listas de enteros, si es necesario
df['Numericalized_Caption'] = df['Numericalized_Caption'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Verificar la conversión
print(df['Numericalized_Caption'].head())  # Ahora debe ser una lista de enteros

# Verificar si hay valores nulos o vacíos en 'Numericalized_Caption'
print(f"Valores nulos en Numericalized_Caption: {df['Numericalized_Caption'].isnull().sum()}")  # Número de valores nulos
print(f"Listas válidas en Numericalized_Caption: {df['Numericalized_Caption'].apply(lambda x: isinstance(x, list)).sum()}")  # Cuántas son listas válidas

# Definir los tokens especiales
pad_token = 0  # Suponiendo que <PAD> tiene índice 0
unk_token = 3  # Suponiendo que <UNK> tiene índice 3

# Contar los tokens <PAD> y <UNK>
pad_count = sum(df['Numericalized_Caption'].apply(lambda x: x.count(pad_token) if isinstance(x, list) else 0))
unk_count = sum(df['Numericalized_Caption'].apply(lambda x: x.count(unk_token) if isinstance(x, list) else 0))

# Calcular el total de palabras (tokens) en el dataset
total_tokens = sum(df['Numericalized_Caption'].apply(lambda x: len(x) if isinstance(x, list) else 0))

# Calcular el porcentaje de <PAD> y <UNK>
pad_percentage = (pad_count / total_tokens) * 100 if total_tokens > 0 else 0
unk_percentage = (unk_count / total_tokens) * 100 if total_tokens > 0 else 0

# Imprimir los resultados
print(f"Total tokens (palabras): {total_tokens}")
print(f"Tokens <PAD>: {pad_count} ({pad_percentage:.2f}%)")
print(f"Tokens <UNK>: {unk_count} ({unk_percentage:.2f}%)")

# -------- PADDING METRICS ----------

# Calcular las longitudes de las secuencias
df['Sequence_Length'] = df['Numericalized_Caption'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Estadísticas descriptivas
min_length = df['Sequence_Length'].min()
max_length = df['Sequence_Length'].max()
mean_length = df['Sequence_Length'].mean()
std_length = df['Sequence_Length'].std()

print(f"Longitud mínima: {min_length}")
print(f"Longitud máxima: {max_length}")
print(f"Longitud promedio: {mean_length:.2f}")
print(f"Desviación estándar: {std_length:.2f}")

# Percentiles para decidir longitud máxima óptima
percentiles = np.percentile(df['Sequence_Length'], [50, 75, 90, 95, 99])
print("Percentiles de longitud:")
for i, p in enumerate([50, 75, 90, 95, 99]):
    print(f"{p}% de las frases tienen {percentiles[i]} tokens o menos")

# Graficar un histograma de las longitudes
plt.figure(figsize=(10, 6))
plt.hist(df['Sequence_Length'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title("Distribución de las longitudes de las secuencias")
plt.xlabel("Longitud de la secuencia")
plt.ylabel("Frecuencia")
plt.grid(axis='y', alpha=0.75)
plt.show()

# -------- FRECUENCIA DE PALABRAS -----
vocab_path = "vocab.json"
vocab = load_vocab(vocab_path)

# Crear una lista de todas las palabras en el dataset original (antes de numericalizar)
df['Title'] = df['Title'].fillna("").astype(str).str.lower()
all_words = [word for caption in df['Title'] for word in caption.split()]
word_frequencies = Counter(all_words)

# Proporción de palabras cubiertas por el vocabulario
vocab_covered = set(vocab.stoi.keys())
unique_words = set(all_words)
coverage = len(vocab_covered.intersection(unique_words)) / len(unique_words) * 100
print(f"Porcentaje de palabras cubiertas por el vocabulario: {coverage:.2f}%")

# Palabras más comunes fuera del vocabulario
unknown_words = [word for word in all_words if word not in vocab_covered]
unknown_count = Counter(unknown_words)

print("Palabras más comunes fuera del vocabulario:")
print(unknown_count.most_common(10))

# Gráfica de distribución de frecuencias de palabras
frequencies = list(word_frequencies.values())
plt.figure(figsize=(10, 6))
plt.hist(frequencies, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.yscale('log')  # Escala logarítmica para visualizar mejor
plt.title("Distribución de Frecuencias de Palabras")
plt.xlabel("Frecuencia de las palabras")
plt.ylabel("Número de palabras")
plt.grid(axis='y', alpha=0.75)
plt.show()

# -------- ANALISIS DE FRASES --------

# Palabras iniciales más comunes
start_words = [caption.split()[0] for caption in df['Title'] if len(caption.split()) > 0]
start_word_count = Counter(start_words)

print("Palabras iniciales más comunes:")
print(start_word_count.most_common(10))

# Palabras finales más comunes
end_words = [caption.split()[-1] for caption in df['Title'] if len(caption.split()) > 0]
end_word_count = Counter(end_words)

print("Palabras finales más comunes:")
print(end_word_count.most_common(10))

# Análisis de diversidad sintáctica
unique_phrases = set(df['Title'].tolist())
print(f"Número total de captions: {len(df['Title'])}")
print(f"Número de captions únicas: {len(unique_phrases)}")
print(f"Porcentaje de captions únicas: {(len(unique_phrases) / len(df['Title'])) * 100:.2f}%")
