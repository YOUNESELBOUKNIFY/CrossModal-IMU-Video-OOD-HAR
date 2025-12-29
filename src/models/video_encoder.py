import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights


class VideoEncoder(nn.Module):
    """
    Encodeur vidéo basé sur Vision Transformer (ViT)
    Traite 10 frames d'un segment de 5 secondes
    """
    def __init__(self, 
                 num_frames: int = 10,
                 img_size: int = 224,
                 embed_dim: int = 768,
                 projection_dim: int = 256,
                 pretrained: bool = True):
        super().__init__()
        
        self.num_frames = num_frames
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Utiliser un Vision Transformer pré-entraîné
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.vit = models.vit_b_16(weights=weights)
        else:
            self.vit = models.vit_b_16(weights=None)
        
        # Retirer la tête de classification
        self.vit.heads = nn.Identity()
        
        # Temporal aggregation : moyenne ou attention sur les frames
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Projection head pour le contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )
        
        # Query learnable pour l'agrégation temporelle
        self.temporal_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x, return_projection=True):
        """
        Args:
            x: (batch_size, num_frames, 3, 224, 224) - frames vidéo
            return_projection: Si True, retourne la projection; sinon l'embedding
        
        Returns:
            embedding ou projection selon return_projection
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape pour traiter toutes les frames d'un coup
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extraire les features avec ViT
        frame_embeddings = self.vit(x)  # (batch_size * num_frames, embed_dim)
        
        # Reshape pour séparer batch et frames
        frame_embeddings = frame_embeddings.view(batch_size, num_frames, self.embed_dim)
        
        # Agrégation temporelle avec attention
        query = self.temporal_query.expand(batch_size, -1, -1)
        video_embedding, _ = self.temporal_attention(
            query, 
            frame_embeddings, 
            frame_embeddings
        )
        video_embedding = video_embedding.squeeze(1)  # (batch_size, embed_dim)
        
        if return_projection:
            # Projeter pour le contrastive learning
            projection = self.projection_head(video_embedding)
            return projection
        else:
            return video_embedding


class SimpleVideoEncoder(nn.Module):
    """
    Version simplifiée de l'encodeur vidéo (sans ViT pré-entraîné)
    Utilise une architecture CNN + pooling temporel
    Plus léger et rapide pour le développement initial
    """
    def __init__(self,
                 num_frames: int = 10,
                 img_size: int = 224,
                 embed_dim: int = 512,
                 projection_dim: int = 256):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Utiliser ResNet18 comme backbone CNN
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Retirer la couche fc
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Dimension de sortie de ResNet18 : 512
        cnn_output_dim = 512
        
        # Projection vers embed_dim
        self.feature_projection = nn.Linear(cnn_output_dim, embed_dim)
        
        # Agrégation temporelle simple (moyenne)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head pour contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )
        
    def forward(self, x, return_projection=True):
        """
        Args:
            x: (batch_size, num_frames, 3, 224, 224)
            return_projection: Si True, retourne la projection
        
        Returns:
            embedding ou projection
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape pour traiter toutes les frames
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extraire features CNN
        features = self.cnn_backbone(x)  # (batch_size * num_frames, 512, 1, 1)
        features = features.view(batch_size * num_frames, -1)  # (batch_size * num_frames, 512)
        
        # Projeter vers embed_dim
        features = self.feature_projection(features)  # (batch_size * num_frames, embed_dim)
        
        # Reshape pour séparer batch et frames
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, embed_dim)
        
        # Agrégation temporelle (moyenne)
        features = features.transpose(1, 2)  # (batch_size, embed_dim, num_frames)
        video_embedding = self.temporal_pool(features)  # (batch_size, embed_dim, 1)
        video_embedding = video_embedding.squeeze(-1)  # (batch_size, embed_dim)
        
        if return_projection:
            projection = self.projection_head(video_embedding)
            return projection
        else:
            return video_embedding


# Test des modèles
if __name__ == "__main__":
    # Paramètres
    batch_size = 2
    num_frames = 10
    C, H, W = 3, 224, 224
    
    # Créer des données synthétiques
    x = torch.randn(batch_size, num_frames, C, H, W)
    
    print("Test de SimpleVideoEncoder:")
    simple_model = SimpleVideoEncoder(
        num_frames=10,
        embed_dim=512,
        projection_dim=256
    )
    
    projection = simple_model(x, return_projection=True)
    embedding = simple_model(x, return_projection=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Projection shape: {projection.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Nombre de paramètres: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    print("\n" + "="*60)
    print("Test de VideoEncoder (avec ViT):")
    
    # Note: Ce modèle est plus lourd et nécessite plus de mémoire
    vit_model = VideoEncoder(
        num_frames=10,
        embed_dim=768,
        projection_dim=256,
        pretrained=False  # False pour éviter de télécharger les poids
    )
    
    projection_vit = vit_model(x, return_projection=True)
    embedding_vit = vit_model(x, return_projection=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Projection shape: {projection_vit.shape}")
    print(f"Embedding shape: {embedding_vit.shape}")
    print(f"Nombre de paramètres: {sum(p.numel() for p in vit_model.parameters()):,}")