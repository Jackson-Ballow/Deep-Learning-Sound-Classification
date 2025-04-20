import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Hybrid CNN-Transformer
class AudioTransformer(nn.Module):

    def __init__(self, num_classes=2, input_shape=(128, 128), 
                 embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(AudioTransformer, self).__init__()

        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.input_projection = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            dummy_output = self.input_projection(dummy_input)
            self.seq_length = dummy_output.size(2)
            self.feature_dim = dummy_output.size(1) * dummy_output.size(3)
        
        self.embedding = nn.Linear(self.feature_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, self.seq_length, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)  # [batch_size, channels, reduced_height, width]
        batch_size, channels, seq_len, width = x.size()
        x = x.permute(0, 2, 1, 3)  # [batch_size, seq_len, channels, width]
        x = x.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, channels*width]
        
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # [batch_size, embed_dim]
        x = self.classifier(x)  # [batch_size, num_classes]
        
        return x


# Vision Transformer
class AudioTransformerV2(nn.Module):

    def __init__(self, num_classes=2, input_shape=(128, 128), 
                 embed_dim=128, num_heads=4, num_layers=3, 
                 patch_size=16, dropout=0.1):
        super(AudioTransformerV2, self).__init__()
        
        height, width = input_shape
        self.patch_size = patch_size
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        # x shape: [batch_size, 1, height, width]
        batch_size = x.size(0)
        
        x = self.patch_embedding(x)  # [batch_size, embed_dim, grid_h, grid_w]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1+num_patches, embed_dim]
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x[:, 0]  # [batch_size, embed_dim]
        x = self.norm(x)
        x = self.classifier(x)  # [batch_size, num_classes]
        
        return x