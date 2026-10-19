import pytest
import torch

from efficientgraphbench import BenchmarkResults
from efficientgraphbench.results.classification import exceptional_notes
from efficientgraphbench.training.evaluator import classification_metrics


def metric(targets, predictions, classes=3):
    return classification_metrics(
        torch.nn.functional.one_hot(torch.tensor(predictions), classes).float(),
        torch.tensor(targets),
    )


def test_imbalanced_majority_only_predictions():
    result = metric([0] * 9 + [1], [0] * 10)
    assert result["accuracy"] == pytest.approx(0.9)
    assert result["precision"] == pytest.approx(0.81)
    assert result["recall"] == pytest.approx(0.9)
    assert result["f1"] == pytest.approx(0.9 * 18 / 19)
    assert result["f1_macro"] == pytest.approx(9 / 19)
    assert result["f1_micro"] == pytest.approx(0.9)
    row = dict(result, status="SUCCESS", model="gcn", dataset="toy", seed=7)
    notes = exceptional_notes([row])
    assert notes == []  # One tiny test set is insufficient evidence for an alert.


def test_perfect_absent_class_and_mask():
    result = metric([0, 1], [0, 1])
    assert result["f1_macro"] == 1
    assert exceptional_notes([dict(result, status="SUCCESS")]) == []
    logits = torch.eye(3)
    result = classification_metrics(
        logits, torch.tensor([0, 1, -1]), torch.tensor([True, True, False])
    )
    assert result["accuracy"] == 1
    with pytest.raises(ValueError, match="empty"):
        classification_metrics(logits, torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.bool))


def test_known_multiclass_confusion_counts():
    target, pred = [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 0, 2]
    result = metric(target, pred, 4)
    # Each class has one TP; true supports are (3,2,1), predicted supports (2,2,2).
    expected_by_average = {
        "macro": (0.5, 11 / 18, 47 / 90),
        "micro": (0.5, 0.5, 0.5),
        "weighted": (0.5, 0.5, 43 / 90),
    }
    for average, expected in expected_by_average.items():
        for name, value in zip(("precision", "recall", "f1"), expected[:3]):
            assert result[f"{name}_{average}"] == pytest.approx(value)


def test_summary_shows_macro_micro_and_keeps_legacy_missing():
    row = dict(metric([0, 1], [0, 1]), status="SUCCESS", dataset="toy", model="gcn")
    results = BenchmarkResults([row])
    assert results.summary().iloc[0].f1_macro == 1
    assert results.summary(detailed=True).iloc[0].f1_macro == 1
    assert "Macro gives" in results.metric_notes()
    legacy = BenchmarkResults([dict(status="SUCCESS", accuracy=0.8, model="gcn")])
    assert legacy.summary().iloc[0].f1 is None


def test_attention_requires_support_and_consistency():
    def records(total, minority):
        result = metric([0] * (total - minority) + [1] * minority, [0] * total)
        return [
            dict(result, status="SUCCESS", model="gcn", dataset="toy", seed=i) for i in range(10)
        ]

    assert exceptional_notes(records(37, 4)) == []
    supported = records(100, 20)
    notes = exceptional_notes(supported)
    assert len(notes) == 1
    assert "persistent macro/micro" in notes[0] and "zero recall" in notes[0]
    assert exceptional_notes(supported[:4]) == []
    assert exceptional_notes([supported[0]] * 10) == []
    good = metric([0] * 80 + [1] * 20, [0] * 80 + [1] * 20)
    for i in range(3):
        supported[i].update(good)
    assert exceptional_notes(supported) == []
