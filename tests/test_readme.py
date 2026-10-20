import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.models.validation import validate_batch
from src.reporting.update_readme import (
    EXAMPLE_PROFILES, md_table, replace_block, results_markdown)

TEXT = "intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n"


def test_replace_block_swaps_only_the_marked_section_and_is_idempotent():
    once = replace_block(TEXT, "RESULTS", "new content")
    assert once.startswith("intro\n") and once.endswith("outro\n")
    assert "old" not in once and "new content" in once
    assert replace_block(once, "RESULTS", "new content") == once


def test_replace_block_fails_loudly_when_markers_are_missing():
    with pytest.raises(ValueError, match="markers"):
        replace_block("no markers here", "RESULTS", "x")


def fake_metrics():
    cm = {"tn": 600, "fp": 400, "fn": 40, "tp": 360}
    test = {"threshold": 0.3, "accuracy": 0.7, "precision": 0.47, "recall": 0.9, "f1": 0.6,
            "roc_auc": 0.84, "pr_auc": 0.66, "confusion_matrix": cm}
    row = {"accuracy": 0.7, "precision": 0.5, "recall": 0.8, "f1": 0.6, "roc_auc": 0.8, "pr_auc": 0.6}
    return ({"selected_model": "Test Model", "test_metrics": test, "train_churn_rate": 0.265,
             "split_sizes": {"train": 600, "val": 200, "test": 200},
             "dummy_baseline_test": {"accuracy": 0.735, "recall": 0.0},
             "validation_comparison": [{"model": "Test Model", **row}, {"model": "Dummy Baseline", **row}]},
            {"trained_at_utc": "2026-01-01", "versions": {"python": "3.13", "sklearn": "1.9"},
             "random_state": 42, "threshold_beta": 2.0})


def test_results_markdown_reports_the_numbers_it_was_given():
    metrics, meta = fake_metrics()
    importance = pd.DataFrame({"mean_abs_shap": [0.5], "permutation_pr_auc_drop": [None]},
                              index=["Contract"])
    md = results_markdown(metrics, meta, importance)
    for expected in ["Test Model", "**0.30**", "360 churners caught", "40 missed",
                     "400 false alarms", "73.5%", "600 train / 200 validation / 200 test",
                     "| Contract | 0.500 | - |"]:
        assert expected in md, expected


def test_every_markdown_table_row_has_the_same_number_of_columns():
    """Regression: a header such as 'Mean |SHAP|' once broke GitHub tables."""
    metrics, meta = fake_metrics()
    importance = pd.DataFrame({"mean_abs_shap": [0.5], "permutation_pr_auc_drop": [0.1]},
                              index=["Contract"])
    block: list[str] = []
    for line in results_markdown(metrics, meta, importance).splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
        elif block:
            assert len({row.count("|") for row in block}) == 1, block
            block = []
    assert md_table(["a", "b"], [["1", "2"]]).count("\n") == 2


def test_example_profiles_are_valid_customers():
    frame = pd.DataFrame(EXAMPLE_PROFILES.values())
    result = validate_batch(frame)
    assert result.errors == [] and len(result.valid) == len(EXAMPLE_PROFILES)


def test_readme_has_the_required_sections_and_result_markers():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for marker in ("RESULTS", "EXAMPLES"):
        assert f"<!-- {marker}:START -->" in text and f"<!-- {marker}:END -->" in text
    for heading in ["## Problem statement", "## Key features", "## Architecture", "## Dataset",
                    "## Installation", "## Train the model", "## Run the dashboard",
                    "## API usage", "## Model evaluation results", "## Screenshots",
                    "## Limitations", "## Future improvements"]:
        assert heading in text, heading
    assert "```mermaid" in text
