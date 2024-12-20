import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from UTILS2 import clean_dataset, ImageCaptionDataset, get_transforms, preprocess_captions, CaptionCollate
from model_con_self_attention import ImageCaptioningModel
from VOCAB2 import build_vocab_from_csv, load_vocab, beam_search_decoding


def train_model(dataset_csv, images_path, model_save_path, vocab_path, max_images=10000, epochs=25, batch_size=16, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(vocab_path)

    dataset = ImageCaptionDataset(dataset_csv, images_path, vocab, transform=get_transforms())
    print(f"Tamaño del dataset: {len(dataset)}")  # Verificar tamaño antes de crear el Subset
    
    if max_images:
        max_images = min(max_images, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(max_images))
    print(f"Tamaño final del dataset: {len(dataset)}")
    
    collate_fn = CaptionCollate(pad_idx=vocab.stoi["<PAD>"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = ImageCaptioningModel(
        vocab_size=len(vocab), embed_dim=512, attention_dim=512,
        num_heads=8, ff_dim=2048, num_layers=6
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, captions) in enumerate(dataloader):
            print(f"Procesando batch {batch_idx + 1}/{len(dataloader)}")  # Debug por batch
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()
            predictions = model(images, captions)
            loss = criterion(predictions.view(-1, predictions.size(-1)), captions[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"EPOCH [{epoch + 1}/{epochs}], LOSS: {epoch_loss / len(dataloader)}")
    torch.save({"model_state_dict": model.state_dict()}, model_save_path)
    print(f"MODEL SAVED IN {model_save_path}")


def beam_search_decoding(model, image_tensor, vocab, device, beam_width=3, max_length=20, repetition_penalty=1.2):
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)
        encoder_out = encoder_out.permute(0, 2, 3, 1).reshape(encoder_out.size(0), -1, 2048)
        encoder_out = model.encoder_proj(encoder_out)  # Proyección al tamaño embed_dim
        memory = encoder_out.permute(1, 0, 2)  # Memory necesita secuencia por batch

        sequences = [[vocab.stoi["<SOS>"]]]
        scores = [0.0]

        for _ in range(max_length):
            all_candidates = []

            for seq, score in zip(sequences, scores):
                if seq[-1] == vocab.stoi["<EOS>"]:
                    all_candidates.append((seq, score))
                    continue

                input_word = torch.tensor([seq[-1]]).to(device)
                embed = model.decoder.embedding(input_word)  # Corregido para usar el embedding del decodificador
                embed = embed.unsqueeze(0)  # Añadir dimensión de batch
                tgt = embed.permute(1, 0, 2)  # Convertir a secuencia por batch

                decoder_output = model.decoder.transformer_decoder(tgt, memory)
                preds = model.decoder.fc_out(decoder_output.permute(1, 0, 2))[:, -1, :]

                top_k_probs, top_k_indices = torch.topk(torch.softmax(preds, dim=-1), beam_width)
                for i in range(beam_width):
                    next_word = top_k_indices[0, i].item()
                    penalty = seq.count(next_word) * repetition_penalty
                    candidate_score = score - torch.log(top_k_probs[0, i]).item() * penalty
                    candidate = seq + [next_word]
                    all_candidates.append((candidate, candidate_score))

            ordered = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
            sequences, scores = zip(*ordered)

        final_seq = sequences[0]
        return ' '.join([vocab.itos[idx] for idx in final_seq if idx not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]]])

    
def generate_captions(model_path, dataset_csv, images_path, results_path, vocab_path, beam_width=3, max_results=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(vocab_path)

    # Cargar el modelo completo
    model = ImageCaptioningModel(
        vocab_size=len(vocab), 
        embed_dim=512, 
        attention_dim=512,
        num_heads=8, 
        ff_dim=2048, 
        num_layers=6
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Dataset y Subset
    dataset = ImageCaptionDataset(dataset_csv, images_path, vocab, transform=get_transforms())
    subset_indices = range(min(len(dataset), max_results))  # Limitar a max_results
    os.makedirs(results_path, exist_ok=True)

    # Generar captions y guardar imágenes con margen blanco
    for subset_idx in subset_indices:
        image_tensor, _ = dataset[subset_idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Usar beam search decoding para generar la caption
        caption = beam_search_decoding(model, image_tensor, vocab, device, beam_width=beam_width)

        # Limitar la caption a 22 palabras
        caption_words = caption.split()[:22]  # Tomar las primeras 10 palabras
        caption_text = " ".join(caption_words)

        # Cargar la imagen original
        image_path = os.path.join(images_path, dataset.data.iloc[subset_idx]['Image_Name'] + '.jpg')
        original_image = Image.open(image_path).convert("RGB")

        # Crear margen blanco arriba
        margin_height = 100
        new_image = Image.new("RGB", (original_image.width, original_image.height + margin_height), (255, 255, 255))
        new_image.paste(original_image, (0, margin_height))

        # Dibujar la caption en el margen blanco
        draw = ImageDraw.Draw(new_image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            font = ImageFont.load_default()

        text_color = (0, 0, 0)
        max_line_length = 40  # Longitud máxima de una línea
        lines = [caption_text[i:i + max_line_length] for i in range(0, len(caption_text), max_line_length)]

        y_offset = 10
        for line in lines:
            draw.text((10, y_offset), line, font=font, fill=text_color)
            y_offset += 30

        # Guardar la imagen con la caption
        output_path = os.path.join(results_path, f"{dataset.data.iloc[subset_idx]['Image_Name']}_caption.jpg")
        new_image.save(output_path)
    print(f"CAPTIONS GENERATED SAVED IN --> {results_path}")

if __name__ == "__main__":
    dataset_images_path = "/export/fhome/vlia09/DATASETS/Food Images/"
    dataset_captions_path = "/export/fhome/vlia09/DATASETS/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    test_csv = "test.csv"
    train_csv = "train.csv"
    val_csv = "val.csv"
    vocab_path = "vocab.json"
    results_path = "results_with_self_attention"
    model_save_path = "/export/fhome/vlia09/Challenge3/model_2.pth"

    # PREPARAR DATASET
    clean_dataset(dataset_images_path, dataset_captions_path, test_csv, train_csv, val_csv)
    build_vocab_from_csv(train_csv, freq_threshold=3, vocab_path=vocab_path)
    preprocess_captions(train_csv, load_vocab(vocab_path))
    
    

    # ENTRENAR MODELO
    train_model(train_csv, dataset_images_path, model_save_path, vocab_path, max_images=13000, epochs=300, batch_size=64, lr=1e-4)

    # GENERAR CAPTIONS
    generate_captions(model_save_path, train_csv, dataset_images_path, results_path, vocab_path, beam_width=5, max_results=20)