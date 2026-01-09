"""
Loss functions pour IMU-Video Cross-modal Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidContrastiveLoss(nn.Module):
    """
    Sigmoid-based contrastive loss comme dans l'article
    Ref: Sigmoid Loss for Language Image Pre-Training (SigLIP)
    """
    
    def __init__(self, init_temperature=10.0, init_bias=-10.0, learnable=True):
        super().__init__()
        
        if learnable:
            self.temperature = nn.Parameter(torch.tensor(init_temperature).log())
            self.bias = nn.Parameter(torch.tensor(init_bias))
        else:
            self.register_buffer('temperature', torch.tensor(init_temperature).log())
            self.register_buffer('bias', torch.tensor(init_bias))
    
    def forward(self, imu_embeds, video_embeds):
        """
        Args:
            imu_embeds: (batch, dim) - normalized
            video_embeds: (batch, dim) - normalized
        Returns:
            loss: scalar
        """
        batch_size = imu_embeds.shape[0]
        
        # Compute similarity matrix
        # (batch, batch)
        logits = imu_embeds @ video_embeds.T
        
        # Scale by temperature and add bias
        t = self.temperature.exp()
        logits = logits * t + self.bias
        
        # Create labels: diagonal elements are positive pairs (1), others negative (-1)
        labels = 2 * torch.eye(batch_size, device=logits.device) - 1
        
        # Sigmoid loss: -log(sigmoid(z_ij * logits_ij))
        # Équivalent à: log(1 + exp(-z_ij * logits_ij))
        loss = F.binary_cross_entropy_with_logits(
            logits * labels,
            (labels + 1) / 2,  # convert {-1, 1} to {0, 1}
            reduction='mean'
        )
        
        return loss


class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE (NT-Xent) contrastive loss
    Alternative au sigmoid loss
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, imu_embeds, video_embeds):
        """
        Args:
            imu_embeds: (batch, dim) - normalized
            video_embeds: (batch, dim) - normalized
        """
        batch_size = imu_embeds.shape[0]
        
        # Similarity matrix
        logits = imu_embeds @ video_embeds.T / self.temperature  # (batch, batch)
        
        # Labels: diagonal elements
        labels = torch.arange(batch_size, device=logits.device)
        
        # Cross entropy loss
        loss_i2v = F.cross_entropy(logits, labels)
        loss_v2i = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i2v + loss_v2i) / 2
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss pour classification avec déséquilibre de classes
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes) - logits
            targets: (batch,) - class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy avec label smoothing
    """
    
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes) - logits
            targets: (batch,) - class indices
        """
        num_classes = inputs.shape[1]
        log_probs = F.log_softmax(inputs, dim=1)
        
        # One-hot encoding with smoothing
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets_one_hot = targets_one_hot * (1 - self.epsilon) + self.epsilon / num_classes
        
        loss = -(targets_one_hot * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_name, **kwargs):
    """Factory pour créer une loss function"""
    
    if loss_name == 'sigmoid_contrastive':
        return SigmoidContrastiveLoss(**kwargs)
    elif loss_name == 'infonce':
        return InfoNCELoss(**kwargs)
    elif loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    else:
        raise ValueError(f"Loss function inconnue: {loss_name}")


def test_losses():
    """Test des loss functions"""
    
    # Test Sigmoid Contrastive
    print("Test Sigmoid Contrastive Loss...")
    loss_fn = SigmoidContrastiveLoss()
    imu = F.normalize(torch.randn(8, 256), dim=1)
    video = F.normalize(torch.randn(8, 256), dim=1)
    loss = loss_fn(imu, video)
    print(f"Loss: {loss.item():.4f}")
    
    # Test InfoNCE
    print("\nTest InfoNCE Loss...")
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(imu, video)
    print(f"Loss: {loss.item():.4f}")
    
    # Test Focal Loss
    print("\nTest Focal Loss...")
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 32)
    targets = torch.randint(0, 32, (8,))
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_losses()