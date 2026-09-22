"""Dependency-free checks for the external-validation data and scoring rules."""

import unittest
from pathlib import Path

from .data import (
    EXPECTED_TARGET_COUNTS,
    TARGET_EMOTIONS,
    SourceExample,
    TargetExample,
    load_semeval,
    split_source_by_dialogue,
    target_metrics,
)


class DataTests(unittest.TestCase):
    def test_dialogue_split_never_separates_utterances(self):
        rows = [
            SourceExample(f"show:{group}", f"utterance {group}-{turn}", 1,
                          (1, 0), (0, -100))
            for group in range(20) for turn in range(3)
        ]
        train, dev = split_source_by_dialogue(rows, seed=42)
        self.assertEqual(len(train) + len(dev), len(rows))
        self.assertFalse({row.group for row in train} & {row.group for row in dev})
        self.assertEqual(
            [row.group for row in dev],
            [row.group for row in split_source_by_dialogue(rows, seed=42)[1]],
        )

    def test_official_macro_is_unweighted_across_emotions(self):
        rows, predictions = [], []
        for emotion in TARGET_EMOTIONS:
            for label in (0, 1, 2, 3):
                rows.append(TargetExample(f"{emotion}-{label}", "example", emotion, label))
                predictions.append(label if emotion != "sadness" else 3 - label)
        score = target_metrics(rows, predictions)
        self.assertAlmostEqual(score["official_macro_pearson"], 0.5)
        self.assertAlmostEqual(score["per_emotion"]["sadness"]["pearson"], -1.0)

    def test_official_archive_counts_and_split_boundaries(self):
        archive = Path(__file__).resolve().parents[1] / "data/SemEval2018-Task1-all-data.zip"
        if not archive.is_file():
            self.skipTest("Official archive not downloaded")
        splits = load_semeval(archive)
        for index, split in enumerate(("train", "dev", "test")):
            for emotion in TARGET_EMOTIONS:
                count = sum(row.emotion == emotion for row in splits[split])
                self.assertEqual(count, EXPECTED_TARGET_COUNTS[emotion][index])


if __name__ == "__main__":
    unittest.main()
