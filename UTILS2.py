# -*- coding: utf-8 -*-

# UTILS2.PY
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

def clean_dataset(dataset_images_path, dataset_captions_path, test_csv, train_csv, val_csv, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    df = pd.read_csv(dataset_captions_path)
    image_files = set(os.listdir(dataset_images_path))
    valid_images = df['Image_Name'].apply(lambda x: x + '.jpg').isin(image_files)
    cleaned_df = df[valid_images]
    
    np.random.seed(42) 
    shuffled_indices = np.random.permutation(len(cleaned_df))

    train_end = int(len(cleaned_df) * train_ratio)
    val_end = train_end + int(len(cleaned_df) * validation_ratio)

    train_indices = shuffled_indices[:train_end]
    validation_indices = shuffled_indices[train_end:val_end]
    test_indices = shuffled_indices[val_end:]

    train_df = cleaned_df.iloc[train_indices]
    validation_df = cleaned_df.iloc[validation_indices]
    test_df = cleaned_df.iloc[test_indices]

    train_df.to_csv(train_csv, index=False)
    validation_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Corregir los nombres de las variables en el print
    print(f"Dataset cleaned and split saved in: \nTrain --> {train_csv}\nValidation --> {val_csv}\nTest --> {test_csv}")

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_path, images_path, vocab, transform=None):
        self.data = pd.read_csv(csv_path)
        self.images_path = images_path
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            image_path = os.path.join(self.images_path, row['Image_Name'] + '.jpg')
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            caption = row['Title']
            numericalized_caption = torch.tensor(self.vocab.numericalize(caption), dtype=torch.long)
            return image, numericalized_caption
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            raise IndexError("Invalid dataset entry")
    

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class CaptionCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        images = torch.stack(images)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions

def preprocess_captions(dataset_csv, vocab):
    df = pd.read_csv(dataset_csv)
    df['Title'] = df['Title'].fillna("").astype(str)
    df['Numericalized_Caption'] = df['Title'].apply(lambda x: vocab.numericalize(str(x)))
    df.to_csv(dataset_csv, index=False)
    

