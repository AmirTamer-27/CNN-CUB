"""Cross-entropy plus cosine alignment on the same class logits."""
import torch.nn.functional as F


def combined_loss(scores, labels, cosine_weight=1.0):
    ce_loss = F.cross_entropy(scores, labels)
    targets = F.one_hot(labels, num_classes=scores.shape[1]).to(scores.dtype)
    similarity = F.cosine_similarity(scores, targets, dim=1)
    cosine_loss = (1 - similarity).mean()
    total_loss = ce_loss + cosine_weight * cosine_loss
    return total_loss, ce_loss, cosine_loss
