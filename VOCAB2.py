# -*- coding: utf-8 -*-

#VOCAB2.PY
import pandas as pd
from collections import Counter
import json
import os
import torch
from torchvision.transforms import functional as F
from PIL import Image

class Vocabulary:
    def __init__(self, freq_thrseshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = len(self.itos)

        for sentence in sentence_list:
            if isinstance(sentence, str): 
                for word in sentence.split(' '):
                    frequencies[word] += 1
                    if frequencies[word] == self.freq_threshold:
                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1

    def numericalize(self, text):
        if not isinstance(text, str):
            text = ""  
        tokenized_text = text.split(' ')
        return [
            self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text
        ]

def build_vocab_from_csv(csv_path, freq_threshold, vocab_path):
    df = pd.read_csv(csv_path)
    
    df['Title'] = df['Title'].fillna("").astype(str)

    captions = df['Title'].tolist()
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(captions)

    with open(vocab_path, 'w') as f:
        json.dump({"itos": vocab.itos, "stoi": vocab.stoi}, f)

    print(f"vocabulary saved in --> {vocab_path}")


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab = Vocabulary(freq_threshold=1)  
    vocab.itos = {int(k): v for k, v in vocab_data["itos"].items()}
    vocab.stoi = vocab_data["stoi"]
    return vocab


def beam_search_decoding(model, image_tensor, vocab, device, beam_width=3, max_length=50, repetition_penalty=1.2):
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)
        encoder_out = encoder_out.permute(0, 2, 3, 1).reshape(encoder_out.size(0), -1, 2048)
        h, c = model.init_hidden_state(1)

        sequences = [[vocab.stoi["<SOS>"]]]
        scores = [0.0]

        for _ in range(max_length):
            all_candidates = []

            for seq, score in zip(sequences, scores):
                if seq[-1] == vocab.stoi["<EOS>"]:
                    all_candidates.append((seq, score))
                    continue

                input_word = torch.tensor([seq[-1]]).to(device)
                embed = model.embedding(input_word)
                context, _ = model.attention(encoder_out, h)
                h, c = model.decoder(torch.cat([embed, context], dim=1), (h, c))
                preds = model.fc(h)
                top_k_probs, top_k_indices = torch.topk(torch.softmax(preds, dim=-1), beam_width)
                top_k_indices = top_k_indices.squeeze(0)

                for i in range(beam_width):
                    next_word = top_k_indices[i].item()
                    # Penalización por repetición
                    penalty = seq.count(next_word) * repetition_penalty
                    candidate_score = score - torch.log(top_k_probs[0, i]).item() * penalty
                    candidate = seq + [next_word]
                    all_candidates.append((candidate, candidate_score))

            ordered = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
            sequences, scores = zip(*ordered)

        final_seq = sequences[0]
        return ' '.join([vocab.itos[idx] for idx in final_seq if idx not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]]])


"""
def generate_caption(model, image_tensor, vocab, device, max_length=50):
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)
        encoder_out = encoder_out.permute(0, 2, 3, 1).reshape(encoder_out.size(0), -1, 2048)
        h, c = model.init_hidden_state(1)

        word_idx = vocab.stoi["<SOS>"]
        caption = []
        prev_word = None
        for _ in range(max_length):
            embed = model.embedding(torch.tensor([word_idx]).to(device))
            context, _ = model.attention(encoder_out, h)
            h, c = model.decoder(torch.cat([embed, context], dim=1), (h, c))
            word_idx = torch.argmax(model.fc(h), dim=1).item()
            word = vocab.itos[word_idx]
            
            # Evitar repeticiones consecutivas
            if word == prev_word:
                continue
            if word == "<EOS>":
                break
            caption.append(word)
            prev_word = word
    return ' '.join(caption)
"""

"""def beam_search_decoding(model, image_tensor, vocab, device, beam_width=3, max_length=50):
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)
        encoder_out = encoder_out.permute(0, 2, 3, 1).reshape(encoder_out.size(0), -1, 2048)
        h, c = model.init_hidden_state(1)

        # Inicialización del beam con secuencias vacías
        sequences = [[vocab.stoi["<SOS>"]]]  # Cada secuencia comienza con <SOS>
        scores = [0.0]  # Puntuación acumulada para cada secuencia

        for _ in range(max_length):
            all_candidates = []
            
            for seq, score in zip(sequences, scores):
                if seq[-1] == vocab.stoi["<EOS>"]:  # Si la secuencia termina en <EOS>, no se expande más
                    all_candidates.append((seq, score))
                    continue

                # Generar predicciones para la siguiente palabra
                input_word = torch.tensor([seq[-1]]).to(device)
                embed = model.embedding(input_word)
                context, _ = model.attention(encoder_out, h)
                h, c = model.decoder(torch.cat([embed, context], dim=1), (h, c))
                preds = model.fc(h)

                # Obtener las top-k palabras
                top_k_probs, top_k_indices = torch.topk(torch.softmax(preds, dim=-1), beam_width)
                top_k_indices = top_k_indices.squeeze(0)  # Asegurarse de que sea un tensor unidimensional

                for i in range(beam_width):
                    candidate = seq + [top_k_indices[i].item()]  # Expandir la secuencia
                    candidate_score = score - torch.log(top_k_probs[0, i]).item()  # Minimizar log-probabilidad
                    all_candidates.append((candidate, candidate_score))

            # Seleccionar los mejores candidatos
            ordered = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
            sequences, scores = zip(*ordered)  # Actualizar secuencias y puntuaciones

        # Devolver la mejor secuencia
        final_seq = sequences[0]
        return ' '.join([vocab.itos[idx] for idx in final_seq if idx not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]]])
"""
