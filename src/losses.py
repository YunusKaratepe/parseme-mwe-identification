"""
Custom loss functions for MWE identification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focal Loss down-weights easy examples and focuses on hard negatives.
    This is particularly useful for MWE identification where most tokens are "O" (not MWE).
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100):
        """
        Args:
            alpha: Weighting factor (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
                   - gamma=0: equivalent to CrossEntropyLoss
                   - gamma=2: standard focal loss
                   - Higher gamma: more focus on hard examples
            ignore_index: Index to ignore in loss calculation (default: -100)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] or [batch_size, seq_len, num_classes]
            targets: [batch_size] or [batch_size, seq_len]
        
        Returns:
            Focal loss value
        """
        # Flatten inputs and targets if needed
        if inputs.dim() == 3:
            # [batch_size, seq_len, num_classes] -> [batch_size * seq_len, num_classes]
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)
        
        # Get log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Get probabilities
        probs = torch.exp(log_probs)
        
        # Create mask for valid targets (not ignore_index)
        valid_mask = targets != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Filter valid targets
        valid_targets = targets[valid_mask]
        valid_log_probs = log_probs[valid_mask]
        valid_probs = probs[valid_mask]
        
        # Get probabilities of true class
        true_class_probs = valid_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        true_class_log_probs = valid_log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        # focal_loss = -alpha * (1 - p_t)^gamma * log(p_t)
        focal_weight = self.alpha * torch.pow(1 - true_class_probs, self.gamma)
        focal_loss = -focal_weight * true_class_log_probs
        
        return focal_loss.mean()


def get_loss_function(loss_type='ce', ignore_index=-100, **kwargs):
    """
    Factory function to get loss function
    
    Args:
        loss_type: 'ce' for CrossEntropyLoss, 'focal' for FocalLoss
        ignore_index: Index to ignore in loss calculation
        **kwargs: Additional arguments for loss function (e.g., alpha, gamma for focal loss)
    
    Returns:
        Loss function
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose 'ce' or 'focal'")


if __name__ == '__main__':
    # Test focal loss
    print("Testing Focal Loss...")
    
    # Simulate predictions and targets
    batch_size = 4
    seq_len = 10
    num_classes = 3  # O, B-MWE, I-MWE
    
    # Random logits
    logits = torch.randn(batch_size, seq_len, num_classes)
    
    # Mostly "O" labels (class 0), simulating MWE imbalance
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets[0, 2] = 1  # B-MWE
    targets[0, 3] = 2  # I-MWE
    targets[1, 5] = 1  # B-MWE
    
    # Test CrossEntropyLoss
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    ce_value = ce_loss(logits.view(-1, num_classes), targets.view(-1))
    print(f"CrossEntropyLoss: {ce_value.item():.4f}")
    
    # Test FocalLoss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0, ignore_index=-100)
    focal_value = focal_loss(logits, targets)
    print(f"FocalLoss (gamma=2.0): {focal_value.item():.4f}")
    
    # Test with different gamma
    focal_loss_high = FocalLoss(alpha=1.0, gamma=4.0, ignore_index=-100)
    focal_value_high = focal_loss_high(logits, targets)
    print(f"FocalLoss (gamma=4.0): {focal_value_high.item():.4f}")
    
    print("\n✓ Focal Loss implementation tested successfully!")
