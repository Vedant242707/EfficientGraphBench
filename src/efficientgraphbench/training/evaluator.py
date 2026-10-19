import torch


@torch.inference_mode()
def classification_metrics(logits, labels, mask=None):
    """Single-label test metrics; undefined divisions are zero, never NaN.

    Precision/recall/F1 aliases are support-weighted. Macro includes classes
    present in either test targets or predictions; absent-in-both classes are
    excluded. Counts require O(C) storage rather than a C by C confusion matrix.
    """
    if mask is not None:
        logits, labels = logits[mask], labels[mask]
    if labels.numel() == 0:
        raise ValueError("cannot evaluate an empty split")
    classes = logits.shape[-1]
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("test logits are not finite")
    if bool(((labels < 0) | (labels >= classes)).any()):
        raise ValueError("test labels must be valid class indices")
    predicted = logits.argmax(dim=-1)
    support = torch.bincount(labels, minlength=classes).cpu().double()
    predicted_count = torch.bincount(predicted, minlength=classes).cpu().double()
    tp = torch.bincount(labels[predicted == labels], minlength=classes).cpu().double()
    precision = tp / predicted_count.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + predicted_count).clamp_min(1)
    active = (support + predicted_count) > 0
    weights = support / support.sum()
    micro = float(tp.sum() / support.sum())
    result = {"accuracy": micro}
    for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        result[name] = result[f"{name}_weighted"] = float((values * weights).sum())
        result[f"{name}_macro"] = float(values[active].mean())
        result[f"{name}_micro"] = micro
    result["classification_per_class"] = [
        {
            "class_id": i,
            "support": int(support[i]),
            "predicted": int(predicted_count[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i in range(classes)
    ]
    result["classification_protocol"] = {
        "task": "single_label_multiclass",
        "average": "weighted",
        "macro_classes": "test target/prediction union",
        "zero_division": 0,
        "split": "test",
        "checkpoint": "best validation accuracy",
    }
    return result


@torch.inference_mode()
def accuracy(logits, labels, mask) -> float:
    if not bool(mask.any()):
        raise ValueError("cannot evaluate an empty split")
    selected = labels[mask]
    return int((logits[mask].argmax(dim=-1) == selected).sum()) / selected.numel()
