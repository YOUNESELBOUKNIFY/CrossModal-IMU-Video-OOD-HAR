import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Convertit les séries temporelles univariées en patches et les projette
    """
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int, stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        
        # Projection linéaire pour chaque patch
        self.projection = nn.Linear(patch_size, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - série univariée
        Returns:
            patches: (batch_size, num_patches, embed_dim)
        """
        batch_size, seq_len = x.shape
        
        # Extraire les patches avec stride
        patches = []
        for i in range(0, seq_len - self.patch_size + 1, self.stride):
            patch = x[:, i:i+self.patch_size]
            patches.append(patch)
        
        # Stack patches: (batch_size, num_patches, patch_size)
        patches = torch.stack(patches, dim=1)
        
        # Projeter dans l'espace d'embedding
        patches = self.projection(patches)  # (batch_size, num_patches, embed_dim)
        
        return patches


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel pour préserver l'ordre temporel des patches
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        
        # Créer l'encodage positionnel
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Ajouter dimension batch
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        """
        return x + self.pe[:, :x.size(1), :]


class PatchTST(nn.Module):
    """
    PatchTST : Patch Time Series Transformer
    Traite chaque canal de la série multivariée de manière indépendante
    """
    def __init__(self, 
                 seq_len: int = 250,           # Longueur de la séquence (5s à 50Hz)
                 patch_size: int = 16,          # Taille des patches
                 stride: int = 16,              # Stride pour extraire les patches
                 num_channels: int = 6,         # Nombre de canaux IMU (3 acc + 3 gyro)
                 embed_dim: int = 128,          # Dimension d'embedding
                 num_heads: int = 8,            # Nombre de têtes d'attention
                 num_layers: int = 3,           # Nombre de couches Transformer
                 dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        
        # Calculer le nombre de patches
        self.num_patches = (seq_len - patch_size) // stride + 1
        
        # Patch embedding pour chaque canal (indépendant)
        self.patch_embedding = PatchEmbedding(patch_size, num_channels, embed_dim, stride)
        
        # CLS token (token spécial pour la représentation globale)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=self.num_patches + 1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Normalisation
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_channels, seq_len) - données IMU multivariées
        
        Returns:
            cls_embedding: (batch_size, embed_dim) - représentation globale
            all_embeddings: (batch_size, num_channels, num_patches+1, embed_dim) - tous les embeddings
        """
        batch_size, num_channels, seq_len = x.shape
        
        # Traiter chaque canal indépendamment
        channel_embeddings = []
        
        for i in range(num_channels):
            # Extraire le canal i
            channel_data = x[:, i, :]  # (batch_size, seq_len)
            
            # Créer les patches et projeter
            patches = self.patch_embedding(channel_data)  # (batch_size, num_patches, embed_dim)
            
            # Ajouter le CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            patches = torch.cat([cls_tokens, patches], dim=1)  # (batch_size, num_patches+1, embed_dim)
            
            # Ajouter l'encodage positionnel
            patches = self.pos_encoder(patches)
            
            # Passer dans le Transformer
            encoded = self.transformer_encoder(patches)  # (batch_size, num_patches+1, embed_dim)
            
            channel_embeddings.append(encoded)
        
        # Stack les embeddings de tous les canaux
        all_embeddings = torch.stack(channel_embeddings, dim=1)  # (batch_size, num_channels, num_patches+1, embed_dim)
        
        # Extraire le CLS token de chaque canal et faire la moyenne
        cls_tokens = all_embeddings[:, :, 0, :]  # (batch_size, num_channels, embed_dim)
        cls_embedding = cls_tokens.mean(dim=1)   # (batch_size, embed_dim)
        
        # Normalisation finale
        cls_embedding = self.norm(cls_embedding)
        
        return cls_embedding, all_embeddings


class IMUEncoder(nn.Module):
    """
    Wrapper pour l'encodeur IMU avec projection head pour le contrastive learning
    """
    def __init__(self, 
                 seq_len: int = 250,
                 patch_size: int = 16,
                 stride: int = 16,
                 num_channels: int = 6,
                 embed_dim: int = 128,
                 projection_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        # Encodeur PatchTST
        self.encoder = PatchTST(
            seq_len=seq_len,
            patch_size=patch_size,
            stride=stride,
            num_channels=num_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Projection head pour le contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )
        
    def forward(self, x, return_projection=True):
        """
        Args:
            x: (batch_size, num_channels, seq_len)
            return_projection: Si True, retourne la projection; sinon l'embedding
        
        Returns:
            embedding ou projection selon return_projection
        """
        # Obtenir l'embedding CLS
        cls_embedding, _ = self.encoder(x)
        
        if return_projection:
            # Projeter pour le contrastive learning
            projection = self.projection_head(cls_embedding)
            return projection
        else:
            # Retourner l'embedding brut
            return cls_embedding


# Test du modèle
if __name__ == "__main__":
    # Paramètres
    batch_size = 4
    num_channels = 6
    seq_len = 250  # 5 secondes à 50 Hz
    
    # Créer des données synthétiques
    x = torch.randn(batch_size, num_channels, seq_len)
    
    # Créer le modèle
    model = IMUEncoder(
        seq_len=250,
        patch_size=16,
        stride=16,
        num_channels=6,
        embed_dim=128,
        projection_dim=256,
        num_heads=8,
        num_layers=3
    )
    
    # Forward pass
    projection = model(x, return_projection=True)
    embedding = model(x, return_projection=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Projection shape: {projection.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"\nNombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")