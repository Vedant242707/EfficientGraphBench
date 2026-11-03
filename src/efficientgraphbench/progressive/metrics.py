"""Public definitions for the Progressive selection protocol."""

LABELS = {
    "val_accuracy": "Validation accuracy",
    "val_f1_macro": "Validation macro-F1",
    "val_f1_micro": "Validation micro-F1",
}


def metric_label(name):
    return LABELS.get(name, "Validation score (metric not recorded)")


def selection_protocol(name):
    definitions = {
        "val_accuracy": "Fraction of validation nodes correctly classified.",
        "val_f1_macro": "Unweighted mean of per-class 2TP/(target count + prediction count) "
        "over classes in the validation target/prediction union. Undefined divisions are zero.",
        "val_f1_micro": "Pool counts across validation classes. For single-label multiclass "
        "classification this equals validation accuracy.",
    }
    return {
        "version": 1,
        "metric": name,
        "label": metric_label(name),
        "definition": definitions[name],
        "predictions": "argmax over class logits in eval mode at the last epoch of this stage",
        "population": "stage nodes whose original validation mask is true",
        "aggregation": "arithmetic mean of seed scores; sample SD across seeds (0 for one seed)",
        "eligibility": "selection requires one successful finite score for every expected seed",
        "direction": "higher is better; ties ordered by model identifier",
    }
