"""Matched Chapter 5 representation-transfer check on SemEval-2018 EI-oc.

Run ``python -m external_validation.semeval_transfer --help`` from the repo root.
The primary experiment trains source STL and soft-sharing MTL on the same raw
MEISD rows, then compares their intensity encoders with an untouched BERT.
Every target arm receives a fresh four-class head and identical SemEval data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from .data import (
    SOURCE_EMOTIONS,
    TARGET_EMOTIONS,
    _normalise_text,
    load_meisd,
    load_semeval,
    macro_f1,
    sha256,
    split_source_by_dialogue,
    target_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ARCHIVE = (
    "https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/"
    "SemEval2018-Task1-all-data.zip"
)
ARMS = ("target_only", "meisd_stl", "meisd_soft_mtl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-zip", type=Path, default=ROOT / "data/SemEval2018-Task1-all-data.zip")
    parser.add_argument("--source-csv", type=Path, default=ROOT / "data/MEISD_text.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_semeval_transfer")
    parser.add_argument("--download", action="store_true", help="Fetch official archive from the task author's site")
    parser.add_argument("--check-data", action="store_true", help="Validate files and splits without ML dependencies")
    parser.add_argument("--model", default="bert-base-cased", help="Same base checkpoint for every arm")
    parser.add_argument("--source-seed", type=int, default=42)
    parser.add_argument("--target-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--source-epochs", type=int, default=3)
    parser.add_argument("--target-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--soft-sharing-lambda", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--historical-mtl-checkpoint", type=Path,
        help="Exploratory shortcut: old Chapter 5 checkpoint; source data are not matched to STL",
    )
    parser.add_argument(
        "--allow-nonmatched-source", action="store_true",
        help="Required with --historical-mtl-checkpoint; marks results exploratory",
    )
    return parser.parse_args(argv)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _validate_args(args):
    if args.historical_mtl_checkpoint and not args.allow_nonmatched_source:
        raise ValueError("Historical MTL training used different source data; pass --allow-nonmatched-source")
    if args.historical_mtl_checkpoint and args.model != "bert-base-cased":
        raise ValueError("The bundled historical MTL checkpoint uses bert-base-cased")
    if len(set(args.target_seeds)) != len(args.target_seeds):
        raise ValueError("Target seeds must be distinct")
    for name in ("source_epochs", "target_epochs", "batch_size", "max_length"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.soft_sharing_lambda < 0:
        raise ValueError("Learning rate must be positive; regularisation strengths nonnegative")


def _load_data(args):
    if args.download and not args.target_zip.exists():
        args.target_zip.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(OFFICIAL_ARCHIVE, args.target_zip)
    target = load_semeval(args.target_zip)
    all_target_texts = {
        _normalise_text(row.text) for rows in target.values() for row in rows
    }
    source, excluded = load_meisd(args.source_csv, all_target_texts)
    source_train, source_dev = split_source_by_dialogue(source, args.source_seed)
    audit = {
        "source_csv_sha256": sha256(args.source_csv),
        "target_zip_sha256": sha256(args.target_zip),
        "target_counts": {
            split: dict(Counter(row.emotion for row in rows))
            for split, rows in target.items()
        },
        "source_train_rows": len(source_train),
        "source_dev_rows": len(source_dev),
        "source_train_dialogues": len({row.group for row in source_train}),
        "source_dev_dialogues": len({row.group for row in source_dev}),
        "source_excluded": dict(excluded),
        "official_archive_url": OFFICIAL_ARCHIVE,
    }
    return target, source_train, source_dev, audit


def run_experiment(args, target, source_train, source_dev, audit):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Training needs PyTorch and Transformers; see external_validation/README.md"
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cpu":
        print("WARNING: full source MTL training on CPU may take many hours", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def seed_everything(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    class EncodedDataset(Dataset):
        def __init__(self, rows, source=False):
            self.rows = rows
            if source:
                encoded = tokenizer(
                    [row.text for row in rows], truncation=True, padding="max_length",
                    max_length=args.max_length, return_tensors="pt",
                )
                self.sentiments = torch.tensor([row.sentiment for row in rows], dtype=torch.long)
                self.emotions = torch.tensor([row.emotions for row in rows], dtype=torch.float)
                self.intensities = torch.tensor([row.intensities for row in rows], dtype=torch.long)
            else:
                encoded = tokenizer(
                    [row.emotion for row in rows], [row.text for row in rows],
                    truncation=True, padding="max_length", max_length=args.max_length,
                    return_tensors="pt",
                )
                self.labels = torch.tensor([row.label for row in rows], dtype=torch.long)
            self.inputs = {key: encoded[key] for key in ("input_ids", "attention_mask")}
            if "token_type_ids" in encoded:
                self.inputs["token_type_ids"] = encoded["token_type_ids"]
            self.source = source

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            item = {key: value[index] for key, value in self.inputs.items()}
            if self.source:
                item.update(sentiment=self.sentiments[index], emotion=self.emotions[index],
                            intensity=self.intensities[index])
            else:
                item["label"] = self.labels[index]
            return item

    class SourceModel(nn.Module):
        def __init__(self, multitask):
            super().__init__()
            self.multitask = multitask
            tasks = ("intensity", "emotion", "sentiment") if multitask else ("intensity",)
            self.encoders = nn.ModuleDict({task: AutoModel.from_pretrained(args.model) for task in tasks})
            hidden = self.encoders["intensity"].config.hidden_size
            self.projection = nn.Linear(hidden, hidden // 2)
            self.dropout = nn.Dropout(0.3)
            self.heads = nn.ModuleDict({
                "intensity": nn.Linear(hidden // 2, len(SOURCE_EMOTIONS) * 3),
                **({
                    "emotion": nn.Linear(hidden // 2, len(SOURCE_EMOTIONS)),
                    "sentiment": nn.Linear(hidden // 2, 3),
                } if multitask else {}),
            })

        def task_logits(self, task, batch):
            inputs = {key: batch[key] for key in ("input_ids", "attention_mask")}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]
            hidden = self.encoders[task](**inputs).last_hidden_state[:, 0]
            return self.heads[task](torch.relu(self.projection(self.dropout(hidden))))

        def soft_loss(self):
            encoders = list(self.encoders.values())
            if len(encoders) == 1:
                return torch.zeros((), device=device)
            total = torch.zeros((), device=device)
            for i in range(len(encoders)):
                for j in range(i + 1, len(encoders)):
                    for left, right in zip(encoders[i].parameters(), encoders[j].parameters()):
                        total = total + (left - right).square().sum()
            return args.soft_sharing_lambda * total

    class TargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(args.model)
            self.dropout = nn.Dropout(0.3)
            self.head = nn.Linear(self.encoder.config.hidden_size, 4)

        def forward(self, batch):
            inputs = {key: batch[key] for key in ("input_ids", "attention_mask")}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]
            hidden = self.encoder(**inputs).last_hidden_state[:, 0]
            return self.head(self.dropout(hidden))

    def batches(dataset, shuffle=False, seed=0):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, generator=generator)

    source_ds = EncodedDataset(source_train, source=True)
    source_dev_ds = EncodedDataset(source_dev, source=True)
    target_ds = {name: EncodedDataset(rows) for name, rows in target.items()}
    cross_entropy = nn.CrossEntropyLoss()
    masked_cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)
    bce = nn.BCEWithLogitsLoss()
    source_dir = args.output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    def source_predictions(model, dataset):
        model.eval()
        gold, predicted = [], []
        with torch.no_grad():
            for batch in batches(dataset):
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model.task_logits("intensity", batch).reshape(-1, len(SOURCE_EMOTIONS), 3)
                labels = batch["intensity"]
                mask = labels >= 0
                gold.extend(labels[mask].tolist())
                predicted.extend(logits.argmax(-1)[mask].tolist())
        return macro_f1(gold, predicted, range(3))

    def train_source(arm):
        out = source_dir / f"{arm}_encoder.pt"
        meta_path = source_dir / f"{arm}_meta.json"
        expected = {
            "source_sha256": audit["source_csv_sha256"],
            "target_sha256": audit["target_zip_sha256"],
            "source_seed": args.source_seed,
            "model": args.model,
            "max_length": args.max_length,
            "epochs": args.source_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "soft_sharing_lambda": args.soft_sharing_lambda,
        }
        if out.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if all(meta.get(key) == value for key, value in expected.items()):
                print(f"Reusing matched source encoder: {out}")
                return out
            raise ValueError(f"Existing source checkpoint has different settings: {out}")
        seed_everything(args.source_seed)
        model = SourceModel(multitask=(arm == "meisd_soft_mtl")).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        best_score = float("-inf")
        stale = 0
        for epoch in range(args.source_epochs):
            model.train()
            for batch in batches(source_ds, shuffle=True, seed=args.source_seed + epoch):
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                tasks = ("intensity", "emotion", "sentiment") if model.multitask else ("intensity",)
                for task in tasks:
                    logits = model.task_logits(task, batch)
                    if task == "intensity":
                        loss = masked_cross_entropy(logits.reshape(-1, 3), batch["intensity"].reshape(-1))
                        weight = 0.7 if model.multitask else 1.0
                    elif task == "emotion":
                        loss, weight = bce(logits, batch["emotion"]), 2.0
                    else:
                        loss, weight = cross_entropy(logits, batch["sentiment"]), 1.0
                    (weight * loss).backward()
                if model.multitask:
                    model.soft_loss().backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            score = source_predictions(model, source_dev_ds)
            print(f"source {arm} epoch {epoch + 1}: dev intensity macro-F1={score:.4f}")
            if score > best_score:
                best_score, stale = score, 0
                torch.save(model.encoders["intensity"].state_dict(), out)
                _write_json(meta_path, {**expected, "best_epoch": epoch + 1,
                                        "source_dev_intensity_macro_f1": score})
            else:
                stale += 1
                if stale >= args.early_stopping_patience:
                    break
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def historical_encoder():
        path = args.historical_mtl_checkpoint
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        prefix = "encoder_int."
        encoder_state = {key[len(prefix):]: value for key, value in state.items()
                         if key.startswith(prefix)}
        if not encoder_state:
            raise ValueError("Historical checkpoint has no encoder_int.* weights")
        out = source_dir / "meisd_soft_mtl_historical_encoder.pt"
        torch.save(encoder_state, out)
        _write_json(source_dir / "meisd_soft_mtl_historical_meta.json", {
            "historical_checkpoint": str(path.resolve()), "sha256": sha256(path),
            "source_data_matched_to_stl": False,
        })
        del state, encoder_state
        return out

    def predict_target(model, dataset):
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in batches(dataset):
                batch = {key: value.to(device) for key, value in batch.items()}
                predictions.extend(model(batch).argmax(-1).tolist())
        return predictions

    def train_target(arm, seed):
        seed_everything(seed)
        model = TargetModel()
        if arm != "target_only":
            encoder_state = torch.load(checkpoints[arm], map_location="cpu", weights_only=True, mmap=True)
            model.encoder.load_state_dict(encoder_state, strict=True)
            del encoder_state
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        best_score = float("-inf")
        best_state = args.output_dir / "target" / f"{arm}_seed{seed}_best.pt"
        best_state.parent.mkdir(parents=True, exist_ok=True)
        stale = 0
        for epoch in range(args.target_epochs):
            model.train()
            for batch in batches(target_ds["train"], shuffle=True, seed=seed + epoch):
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                cross_entropy(model(batch), batch["label"]).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            dev_predictions = predict_target(model, target_ds["dev"])
            score = target_metrics(target["dev"], dev_predictions)["official_macro_pearson"]
            print(f"target {arm} seed {seed} epoch {epoch + 1}: dev Pearson={score:.4f}")
            if score > best_score:
                best_score, stale = score, 0
                torch.save(model.state_dict(), best_state)
                best_epoch = epoch + 1
            else:
                stale += 1
                if stale >= args.early_stopping_patience:
                    break
        model.load_state_dict(torch.load(best_state, map_location=device, weights_only=True))
        test_predictions = predict_target(model, target_ds["test"])
        metrics = target_metrics(target["test"], test_predictions)
        metrics.update(arm=arm, seed=seed, best_epoch=best_epoch,
                       dev_official_macro_pearson=best_score)
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics, test_predictions

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "run_manifest.json", {
        **audit, "model": args.model, "source_seed": args.source_seed,
        "target_seeds": args.target_seeds, "source_epochs": args.source_epochs,
        "target_epochs": args.target_epochs, "batch_size": args.batch_size,
        "max_length": args.max_length, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "soft_sharing_lambda": args.soft_sharing_lambda,
        "device": device, "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "historical_nonmatched_mtl": bool(args.historical_mtl_checkpoint),
        "scope": "representation transfer to independent tweet intensity; not dialogue forecasting",
    })
    checkpoints = {
        "meisd_stl": train_source("meisd_stl"),
        "meisd_soft_mtl": historical_encoder() if args.historical_mtl_checkpoint
                          else train_source("meisd_soft_mtl"),
    }
    results = []
    prediction_path = args.output_dir / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arm", "seed", "id", "emotion", "gold", "predicted"))
        for seed in args.target_seeds:
            for arm in ARMS:
                metrics, predictions = train_target(arm, seed)
                results.append(metrics)
                for row, predicted in zip(target["test"], predictions):
                    writer.writerow((arm, seed, row.uid, row.emotion, row.label, predicted))
                stream.flush()
                _write_json(args.output_dir / "results_partial.json", results)
    _write_json(args.output_dir / "results.json", results)
    by_arm = {arm: [r["official_macro_pearson"] for r in results if r["arm"] == arm]
              for arm in ARMS}
    summary = {
        arm: {"mean": statistics.mean(scores),
              "sd": statistics.stdev(scores) if len(scores) > 1 else None,
              "per_seed": scores}
        for arm, scores in by_arm.items()
    }
    for arm in ARMS[1:]:
        summary[arm]["paired_delta_vs_target_only"] = [
            a - b for a, b in zip(by_arm[arm], by_arm["target_only"])
        ]
    summary["meisd_soft_mtl"]["paired_delta_vs_meisd_stl"] = [
        a - b for a, b in zip(by_arm["meisd_soft_mtl"], by_arm["meisd_stl"])
    ]
    summary["exploratory_nonmatched_source"] = bool(args.historical_mtl_checkpoint)
    _write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "summary_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "arm", "macro_pearson_mean", "macro_pearson_sd",
            *[f"{emotion}_pearson_mean" for emotion in TARGET_EMOTIONS],
            *[f"{emotion}_pearson_sd" for emotion in TARGET_EMOTIONS],
        ))
        for arm in ARMS:
            arm_results = [result for result in results if result["arm"] == arm]
            emotion_scores = [
                [result["per_emotion"][emotion]["pearson"] for result in arm_results]
                for emotion in TARGET_EMOTIONS
            ]
            writer.writerow((
                arm, summary[arm]["mean"], summary[arm]["sd"],
                *[statistics.mean(scores) for scores in emotion_scores],
                *[statistics.stdev(scores) if len(scores) > 1 else "" for scores in emotion_scores],
            ))
    print(json.dumps(summary, indent=2))


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    target, source_train, source_dev, audit = _load_data(args)
    print(json.dumps(audit, indent=2))
    if args.check_data:
        return 0
    run_experiment(args, target, source_train, source_dev, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
