import torch


@torch.inference_mode()
def accuracy(logits, labels, mask) -> float:
    if not bool(mask.any()):
        raise ValueError("cannot evaluate an empty split")
    return float((logits[mask].argmax(dim=-1) == labels[mask]).float().mean())
