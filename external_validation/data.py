"""Strict loaders for the official English SemEval-2018 EI-oc data and MEISD."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

TARGET_EMOTIONS = ("anger", "fear", "joy", "sadness")
SOURCE_EMOTIONS = (
    "acceptance", "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
)
SOURCE_SENTIMENTS = ("negative", "neutral", "positive")
EXPECTED_TARGET_COUNTS = {
    "anger": (1701, 388, 1002),
    "fear": (2252, 389, 986),
    "joy": (1616, 290, 1105),
    "sadness": (1533, 397, 975),
}


@dataclass(frozen=True)
class TargetExample:
    uid: str
    text: str
    emotion: str
    label: int


@dataclass(frozen=True)
class SourceExample:
    group: str
    text: str
    sentiment: int
    emotions: tuple[int, ...]
    intensities: tuple[int, ...]  # -100 means no annotated intensity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_text(text: str) -> str:
    return " ".join(text.casefold().split())


def load_semeval(zip_path: Path) -> dict[str, list[TargetExample]]:
    """Load all four emotions and enforce the task's official split boundaries."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"Official SemEval archive missing: {zip_path}")
    splits: dict[str, list[TargetExample]] = {k: [] for k in ("train", "dev", "test")}
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for emotion in TARGET_EMOTIONS:
            spec = {
                "train": f"training/EI-oc-En-{emotion}-train.txt",
                "dev": f"development/2018-EI-oc-En-{emotion}-dev.txt",
                "test": f"test-gold/2018-EI-oc-En-{emotion}-test-gold.txt",
            }
            for split, suffix in spec.items():
                name = f"SemEval2018-Task1-all-data/English/EI-oc/{suffix}"
                if name not in names:
                    raise ValueError(f"Missing official EI-oc file: {name}")
                rows = csv.DictReader(
                    io.StringIO(archive.read(name).decode("utf-8-sig")), delimiter="\t"
                )
                if set(rows.fieldnames or ()) != {
                    "ID", "Tweet", "Affect Dimension", "Intensity Class"
                }:
                    raise ValueError(f"Unexpected EI-oc columns in {name}: {rows.fieldnames}")
                count = 0
                for row in rows:
                    if row["Affect Dimension"].strip().lower() != emotion:
                        raise ValueError(f"Wrong emotion in {name}: {row!r}")
                    match = re.match(r"^([0-3]):", row["Intensity Class"].strip())
                    if not match or not row["ID"].strip() or not row["Tweet"].strip():
                        raise ValueError(f"Invalid EI-oc row in {name}: {row!r}")
                    splits[split].append(TargetExample(
                        row["ID"].strip(), row["Tweet"].strip(), emotion,
                        int(match.group(1)),
                    ))
                    count += 1
                expected = EXPECTED_TARGET_COUNTS[emotion][("train", "dev", "test").index(split)]
                if count != expected:
                    raise ValueError(f"{name}: expected {expected} rows, got {count}")

    owner_by_id: dict[str, str] = {}
    owner_by_text: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            for key, owners, label in (
                (row.uid, owner_by_id, "ID"),
                (_normalise_text(row.text), owner_by_text, "text"),
            ):
                previous = owners.setdefault(key, split)
                if previous != split:
                    raise ValueError(f"Official SemEval {label} crosses {previous}/{split}: {key[:80]}")
    return splits


def load_meisd(csv_path: Path, target_texts: set[str] | None = None) -> tuple[list[SourceExample], Counter]:
    """Use raw, unaugmented MEISD; exclude ambiguous labels and exact target text overlap."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"MEISD source CSV missing: {csv_path}")
    excluded: Counter = Counter()
    result: list[SourceExample] = []
    sentiment_aliases = {"positve": "positive", "postive": "positive", "posit": "positive"}
    emotion_aliases = {"faer": "fear", "fera": "fear", "digust": "disgust", "sadnes": "sadness"}
    target_texts = target_texts or set()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"TV Series", "dialog_ids", "Utterances", "sentiment", "emotion", "intensity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"MEISD source columns missing: {required - set(reader.fieldnames or [])}")
        for row in reader:
            text = row["Utterances"].strip()
            sentiment = row["sentiment"].strip().lower()
            sentiment = sentiment_aliases.get(sentiment, sentiment)
            if not text or sentiment not in SOURCE_SENTIMENTS:
                excluded["missing_text_or_ambiguous_sentiment"] += 1
                continue
            if _normalise_text(text) in target_texts:
                excluded["exact_target_text_overlap"] += 1
                continue
            emotions = [0] * len(SOURCE_EMOTIONS)
            intensities = [-100] * len(SOURCE_EMOTIONS)
            ambiguous = False
            for offset in ("", "2", "3"):
                raw_emotion = (row.get(f"emotion{offset}") or "").strip().lower()
                raw_intensity = (row.get(f"intensity{offset}") or "").strip()
                if not raw_emotion:
                    continue
                emotion = emotion_aliases.get(raw_emotion, raw_emotion)
                if emotion not in SOURCE_EMOTIONS:
                    ambiguous = True
                    break
                index = SOURCE_EMOTIONS.index(emotion)
                emotions[index] = 1
                if raw_intensity in {"1", "2", "3"}:
                    intensities[index] = int(raw_intensity) - 1
                elif raw_intensity and raw_intensity not in {"`"}:
                    ambiguous = True
                    break
            if ambiguous or not any(label >= 0 for label in intensities):
                excluded["ambiguous_or_missing_intensity"] += 1
                continue
            series = row["TV Series"].strip()
            dialogue = row["dialog_ids"].strip()
            if not series or not dialogue:
                excluded["missing_group"] += 1
                continue
            result.append(SourceExample(
                f"{series}:{dialogue}", text, SOURCE_SENTIMENTS.index(sentiment),
                tuple(emotions), tuple(intensities),
            ))
    if not result:
        raise ValueError("No usable MEISD examples after label checks")
    return result, excluded


def split_source_by_dialogue(rows: list[SourceExample], seed: int, dev_fraction: float = 0.15):
    groups = sorted({row.group for row in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two source dialogues")
    random.Random(seed).shuffle(groups)
    n_dev = max(1, round(len(groups) * dev_fraction))
    dev_groups = set(groups[:n_dev])
    train = [row for row in rows if row.group not in dev_groups]
    dev = [row for row in rows if row.group in dev_groups]
    return train, dev


def _pearson(gold: list[int], predicted: list[int]) -> float:
    if len(gold) < 2:
        return 0.0
    mean_gold = sum(gold) / len(gold)
    mean_pred = sum(predicted) / len(predicted)
    cross = sum((a - mean_gold) * (b - mean_pred) for a, b in zip(gold, predicted))
    gold_norm = sum((a - mean_gold) ** 2 for a in gold)
    pred_norm = sum((b - mean_pred) ** 2 for b in predicted)
    return cross / math.sqrt(gold_norm * pred_norm) if gold_norm and pred_norm else 0.0


def macro_f1(gold: list[int], predicted: list[int], labels: range) -> float:
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(gold, predicted))
        fp = sum(a != label and b == label for a, b in zip(gold, predicted))
        fn = sum(a == label and b != label for a, b in zip(gold, predicted))
        scores.append((2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0)
    return sum(scores) / len(scores)


def target_metrics(rows: list[TargetExample], predictions: list[int]) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("Prediction count does not match target examples")
    per_emotion = {}
    for emotion in TARGET_EMOTIONS:
        pairs = [(row.label, pred) for row, pred in zip(rows, predictions) if row.emotion == emotion]
        gold = [a for a, _ in pairs]
        pred = [b for _, b in pairs]
        per_emotion[emotion] = {
            "n": len(gold),
            "pearson": _pearson(gold, pred),
            "macro_f1": macro_f1(gold, pred, range(4)),
            "mae": sum(abs(a - b) for a, b in pairs) / len(pairs),
        }
    return {
        "official_macro_pearson": sum(x["pearson"] for x in per_emotion.values()) / 4,
        "per_emotion": per_emotion,
    }
