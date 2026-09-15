"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import evaluation


def test_BinaryClassificationEvaluator_find_best_f1_and_threshold() -> None:
    """Tests that the F1 score for the computed threshold is correct"""
    y_true = np.random.randint(0, 2, 1000)
    y_pred_cosine = np.random.randn(1000)
    (
        best_f1,
        best_precision,
        best_recall,
        threshold,
    ) = evaluation.BinaryClassificationEvaluator.find_best_f1_and_threshold(
        y_pred_cosine, y_true, high_score_more_similar=True
    )
    y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
    sklearn_f1score = f1_score(y_true, y_pred_labels)
    assert np.abs(best_f1 - sklearn_f1score) < 1e-6


def test_BinaryClassificationEvaluator_find_best_accuracy_and_threshold() -> None:
    """Tests that the Acc score for the computed threshold is correct"""
    y_true = np.random.randint(0, 2, 1000)
    y_pred_cosine = np.random.randn(1000)
    (
        max_acc,
        threshold,
    ) = evaluation.BinaryClassificationEvaluator.find_best_acc_and_threshold(
        y_pred_cosine, y_true, high_score_more_similar=True
    )
    y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
    sklearn_acc = accuracy_score(y_true, y_pred_labels)
    assert np.abs(max_acc - sklearn_acc) < 1e-6


@pytest.mark.parametrize("high_score_more_similar", [True, False])
@pytest.mark.parametrize(
    ("scores", "labels"),
    [
        pytest.param([0.5, 0.5], [1, 0], id="tied-positive-first"),
        pytest.param([0.5, 0.5], [0, 1], id="tied-negative-first"),
        pytest.param([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0], id="mixed-ties-positive-first"),
        pytest.param([0.9, 0.5, 0.5, 0.1], [1, 0, 1, 0], id="mixed-ties-negative-first"),
        pytest.param([0.9, 0.5, 0.1], [1, 1, 1], id="all-positive"),
        pytest.param([0.9, 0.5, 0.1], [0, 0, 0], id="all-negative"),
        pytest.param([0.5], [1], id="single-positive"),
        pytest.param([0.5], [0], id="single-negative"),
        pytest.param([0.9, 0.5, 0.1], [0, 0, 1], id="reversed-ranking"),
        pytest.param([np.nextafter(np.float32(1.0), np.inf, dtype=np.float32), 1.0], [1, 0], id="adjacent-scores"),
    ],
)
def test_BinaryClassificationEvaluator_best_threshold_matches_predictions(
    high_score_more_similar, scores, labels
) -> None:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels)
    if not high_score_more_similar:
        scores = -scores

    def predict(threshold):
        return scores >= threshold if high_score_more_similar else scores <= threshold

    # Each distinct score represents an attainable partition, plus the threshold
    # beyond every score that predicts no positives.
    no_positives_threshold = (
        np.nextafter(scores.max(), np.inf, dtype=scores.dtype)
        if high_score_more_similar
        else np.nextafter(scores.min(), -np.inf, dtype=scores.dtype)
    )
    candidate_thresholds = [*np.unique(scores), no_positives_threshold]
    accuracy, accuracy_threshold = evaluation.BinaryClassificationEvaluator.find_best_acc_and_threshold(
        scores, labels, high_score_more_similar
    )
    assert accuracy == pytest.approx(accuracy_score(labels, predict(accuracy_threshold)))
    assert accuracy == pytest.approx(
        max(accuracy_score(labels, predict(candidate)) for candidate in candidate_thresholds)
    )

    f1, precision, recall, f1_threshold = evaluation.BinaryClassificationEvaluator.find_best_f1_and_threshold(
        scores, labels, high_score_more_similar
    )
    predictions = predict(f1_threshold)
    assert f1 == pytest.approx(f1_score(labels, predictions, zero_division=0))
    assert precision == pytest.approx(precision_score(labels, predictions, zero_division=0))
    assert recall == pytest.approx(recall_score(labels, predictions, zero_division=0))
    assert f1 == pytest.approx(
        max(f1_score(labels, predict(candidate), zero_division=0) for candidate in candidate_thresholds)
    )


def test_BinaryClassificationEvaluator_threshold_with_integer_scores() -> None:
    scores = np.array([3, 2, 1])
    labels = np.array([1, 0, 0])
    assert evaluation.BinaryClassificationEvaluator.find_best_acc_and_threshold(scores, labels, True) == (
        1.0,
        2.5,
    )
    assert evaluation.BinaryClassificationEvaluator.find_best_f1_and_threshold(scores, labels, True) == (
        1.0,
        1.0,
        1.0,
        2.5,
    )


@pytest.mark.parametrize("similarity_fn_name", ["euclidean", "manhattan"])
@pytest.mark.parametrize("sentences_type", [list, tuple, np.array])
def test_BinaryClassificationEvaluator_distance_metrics_direction(similarity_fn_name: str, sentences_type) -> None:
    """The euclidean/manhattan metrics must treat smaller distances as more similar.

    ``pairwise_euclidean_sim``/``pairwise_manhattan_sim`` return negative distances (i.e.
    similarities), but were passed unnegated to the ``greater_is_better=False`` metric
    computations, which expect positive distances. That inverted the ranking, so accuracy,
    F1, precision, recall, AP and MCC were all computed for the opposite classifier
    ("larger distance means similar") and the reported thresholds were negative.
    """
    embeddings = {
        "a1": [0.0, 0.0],
        "a2": [1.0, 0.0],  # distance 1 to a1, similar pair (label 1)
        "b1": [0.0, 10.0],
        "b2": [0.0, 12.0],  # distance 2 to b1, similar pair (label 1)
        "c1": [0.0, 0.0],
        "c2": [10.0, 0.0],  # distance 10 to c1, dissimilar pair (label 0)
        "d1": [5.0, 5.0],
        "d2": [5.0, 25.0],  # distance 20 to d1, dissimilar pair (label 0)
    }

    class MockEncoder:
        def encode(self, sentences, **kwargs):
            return np.array([embeddings[sentence] for sentence in sentences], dtype=np.float32)

    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=sentences_type(["a1", "b1", "c1", "d1"]),
        sentences2=sentences_type(["a2", "b2", "c2", "d2"]),
        labels=[1, 1, 0, 0],
        similarity_fn_names=[similarity_fn_name],
    )
    scores = evaluator.compute_metrics(MockEncoder())[similarity_fn_name]

    # The similar pairs (distances 1 and 2) are perfectly separable from the
    # dissimilar pairs (distances 10 and 20), so every metric should be perfect.
    for metric in ["accuracy", "f1", "precision", "recall", "ap", "mcc"]:
        assert scores[metric] == pytest.approx(1.0), f"{metric}: {scores[metric]}"

    # The thresholds should be positive distances separating the similar from the
    # dissimilar pairs: (2 + 10) / 2 = 6 for both euclidean and manhattan.
    assert scores["accuracy_threshold"] == pytest.approx(6.0)
    assert scores["f1_threshold"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    "similarity_fn_names",
    [["dot", "euclidean"], ["cosine", "dot"], ["cosine", "dot", "euclidean", "manhattan"]],
)
def test_BinaryClassificationEvaluator_multiple_similarity_fn_names(
    stsb_bert_tiny_model: SentenceTransformer, similarity_fn_names: list[str]
) -> None:
    """Tests that the max_* aggregation does not assume that cosine was requested"""
    model = stsb_bert_tiny_model
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=["A man is eating food.", "A cat sits outside.", "The girl plays guitar."],
        sentences2=["A man eats something.", "The sky is blue.", "A woman plays a guitar."],
        labels=[1, 0, 1],
        similarity_fn_names=similarity_fn_names,
    )
    metrics = evaluator(model)

    assert evaluator.primary_metric == "max_ap"
    for metric in ["accuracy", "f1", "precision", "recall", "ap", "mcc"]:
        assert metrics[f"max_{metric}"] == max(metrics[f"{name}_{metric}"] for name in similarity_fn_names)


@pytest.mark.parametrize("similarity_fn_names", [["cosine"], ["cosine", "dot", "euclidean", "manhattan"]])
def test_BinaryClassificationEvaluator_csv_columns_are_aligned(
    stsb_bert_tiny_model: SentenceTransformer, tmp_path, similarity_fn_names: list[str]
) -> None:
    """The results CSV data row must have as many columns as the header row.

    The ``*_accuracy_threshold`` and ``*_f1_threshold`` metrics contain a second underscore, so a
    ``count("_") == 1`` filter dropped them from the data row while keeping them in the header,
    shifting every later column and leaving the trailing ``*_ap``/``*_mcc`` columns empty.
    """
    model = stsb_bert_tiny_model
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=["A man is eating food.", "A cat sits outside.", "The girl plays guitar."],
        sentences2=["A man eats something.", "The sky is blue.", "A woman plays a guitar."],
        labels=[1, 0, 1],
        similarity_fn_names=similarity_fn_names,
    )
    evaluator(model, output_path=str(tmp_path))

    with open(tmp_path / evaluator.csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 2
    assert len(rows[1]) == len(rows[0])
