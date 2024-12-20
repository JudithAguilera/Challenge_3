import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # encoder_dim debe ser 512
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        print(f"ENCODER OUTPUT SHAPE: {encoder_out.shape}")
        print(f"DECODER HIDDEN SHAPE: {decoder_hidden.shape}")

        batch_size, num_features, encoder_dim = encoder_out.shape
        encoder_out_flat = encoder_out.view(batch_size * num_features, encoder_dim)

        att1 = self.encoder_att(encoder_out_flat)  # Proyectar encoder_out_flat
        att1 = att1.view(batch_size, num_features, -1)  # Restaurar dimensiones originales

        att2 = self.decoder_att(decoder_hidden)

        att = torch.tanh(att1 + att2.unsqueeze(1))
        alpha = torch.softmax(self.full_att(att), dim=1)
        context = (encoder_out * alpha).sum(dim=1)
        return context, alpha

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, vocab_size):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, vocab_size)
        )

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        tgt = tgt.permute(1, 0, 2)  # Convertir a secuencia por batch
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output.permute(1, 0, 2))
        return output
    
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, attention_dim, num_heads, ff_dim, num_layers, encoder_dim=2048):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.attention = Attention(encoder_dim=512, decoder_dim=512, attention_dim=attention_dim)
        self.encoder_proj = nn.Linear(encoder_dim, embed_dim)  # Proyección de dimensiones
        self.decoder = TransformerDecoder(embed_dim, num_heads, ff_dim, num_layers, vocab_size)

    def forward(self, images, captions):
        # Encoder
        encoder_out = self.encoder(images)
        encoder_out = encoder_out.permute(0, 2, 3, 1).reshape(encoder_out.size(0), -1, 2048)
        #print(f"ENCODER OUTPUT SHAPE:  {encoder_out.shape}")

        # Proyección a la dimensión embed_dim
        encoder_out = self.encoder_proj(encoder_out)
        #print(f"PROJECTED ENCODER OUTPUT SHAPE:  {encoder_out.shape}")

        # Inicializar decoder_hidden para atención
        batch_size = captions.size(0)
        decoder_hidden = torch.zeros(batch_size, encoder_out.size(2)).to(encoder_out.device)
        #print(f"DECODER HIDDEN SHAPE: {decoder_hidden.shape}")

        # Aplicar atención
        context, alpha = self.attention(encoder_out, decoder_hidden)

        # Decoder
        memory = encoder_out.permute(1, 0, 2)  # Memory necesita secuencia por batch
        tgt = captions[:, :-1]  # Entrada para decodificador: sin el token <EOS>
        output = self.decoder(tgt, memory)
        return output