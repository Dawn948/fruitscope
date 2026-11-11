"""用 E04 的冻结特征训练一个判断照片是否属于支持范围的二分类头。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
from zipfile import ZipFile

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.model import pooled_features
from src.predict import Predictor
from src.transforms import load_image


@dataclass(frozen=True)
class Example:
    image: Path | BytesIO
    split: str
    label: str
    category: str
    name: str


def kaggle_examples(split_file: Path):
    splits = json.loads(split_file.read_text(encoding="utf-8"))["splits"]
    for split, records in splits.items():
        for record in records:
            path = config.ROOT / record["path"]
            label = record["label"]
            yield Example(path, split, label, label, path.name)


def caltech_examples(path: Path, per_category: int):
    # 按物体类别切分，避免用训练见过的同类照片评估陌生类别。
    counts = Counter()
    with ZipFile(path) as bundle:
        member = next(n for n in bundle.namelist() if PurePosixPath(n).name == "101_ObjectCategories.tar.gz")
        with bundle.open(member) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for item in archive:
                parts = PurePosixPath(item.name).parts
                if len(parts) != 3 or not item.isfile() or not parts[2].lower().endswith((".jpg", ".jpeg")):
                    continue
                _, category, filename = parts
                if category in {"BACKGROUND_Google", "strawberry", "pizza"} or counts[category] >= per_category:
                    continue
                counts[category] += 1
                bucket = int.from_bytes(sha256(category.encode()).digest()[:4], "big") % 10
                split = "train" if bucket < 6 else "val" if bucket < 8 else "test"
                with archive.extractfile(item) as photo:
                    yield Example(BytesIO(photo.read()), split, config.OTHER_CLASS, category, filename)


def encode_examples(predictor: Predictor, examples, batch_size: int = 24):
    tensors, rows, vectors, predictions = [], [], [], []
    skipped = 0

    def encode_batch():
        with torch.no_grad():
            features = pooled_features(predictor.model, torch.stack(tensors))
            logits = predictor.model.classifier(features)
            vectors.append(features.cpu())
            predictions.append((logits / predictor.temperature).softmax(dim=1).cpu())
        tensors.clear()

    for example in examples:
        try:
            tensor = predictor.transform(load_image(example.image))
        except (OSError, ValueError) as exc:
            print(f"跳过无法读取的样本 {example.category}/{example.name}: {exc}")
            skipped += 1
            continue
        tensors.append(tensor)
        rows.append(example)
        if len(tensors) == batch_size:
            encode_batch()
    if tensors:
        encode_batch()
    print(f"提取特征 {len(rows)} 张，跳过 {skipped} 张", flush=True)
    return rows, torch.cat(vectors), torch.cat(predictions)


def fit_filter(features: torch.Tensor, positive: torch.Tensor) -> tuple[torch.Tensor, float]:
    mean, std = features.mean(0), features.std(0).clamp_min(0.05)
    inputs = (features - mean) / std
    classifier = nn.Linear(inputs.shape[1], 1)
    optimizer = torch.optim.LBFGS(classifier.parameters(), max_iter=100, line_search_fn="strong_wolfe")
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def closure():
        optimizer.zero_grad()
        losses = loss_fn(classifier(inputs).flatten(), positive.float())
        loss = (losses[positive].mean() + losses[~positive].mean()) / 2
        loss = loss + 0.005 * classifier.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    weight = classifier.weight.detach().flatten() / std
    return weight, float(classifier.bias.detach().item() - weight @ mean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--caltech", type=Path, required=True, help="Caltech 官方 caltech-101.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=10)
    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("请为新权重指定另一个文件")
    if args.per_category < 1:
        parser.error("--per-category 至少为 1")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(config.SEED)
    predictor = Predictor(args.checkpoint, device="cpu")
    fruit_rows, fruit_features, fruit_probs = encode_examples(predictor, kaggle_examples(args.split))
    object_rows, object_features, object_probs = encode_examples(
        predictor, caltech_examples(args.caltech, args.per_category)
    )
    rows = fruit_rows + object_rows
    features = torch.cat((fruit_features, object_features))
    probs = torch.cat((fruit_probs, object_probs))
    masks = {key: torch.tensor([row.split == key for row in rows]) for key in ("train", "val", "test")}
    targets = torch.tensor([predictor.classes.index(row.label) for row in rows])
    other = predictor.classes.index(config.OTHER_CLASS)
    positive = targets != other
    weight, bias = fit_filter(features[masks["train"]], positive[masks["train"]])
    scores = features @ weight + bias
    confidence, labels = probs.max(dim=1)
    accepted = (labels != other) & (confidence >= (predictor.threshold or 0))
    correct = labels == targets

    # 阈值只看验证集，保留所有 E04 已正确接纳的验证集水果。
    validation = masks["val"] & positive & accepted & correct
    if not validation.any():
        raise ValueError("验证集没有正确接纳的水果，无法选择阈值")
    threshold = float(scores[validation].min().item()) - 1e-5
    updated = accepted & (scores >= threshold)

    report = {"per_caltech_category": args.per_category, "threshold": threshold, "splits": {}}
    caltech = torch.arange(len(rows)) >= len(fruit_rows)
    for split, mask in masks.items():
        groups = {
            "fruits": mask & positive,
            "kaggle_other": mask & ~positive & ~caltech,
            "caltech_objects": mask & caltech,
        }
        report["splits"][split] = {
            name: {
                "total": int(group.sum()),
                "accepted_before": int((group & accepted).sum()),
                "accepted_after": int((group & updated).sum()),
                "correct_before": int((group & accepted & correct).sum()),
                "correct_after": int((group & updated & correct).sum()),
                "categories": sorted({row.category for row, keep in zip(rows, group.tolist()) if keep}),
            }
            for name, group in groups.items()
        }
    caught = (masks["test"] & caltech & accepted & ~updated).nonzero().flatten()
    if len(caught):
        i = int(caught[confidence[caught].argmax()])
        report["example"] = {
            "category": rows[i].category, "file": rows[i].name,
            "before": predictor.classes[int(labels[i])], "score": float(confidence[i]),
        }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint["scope_filter"] = {"weight": weight, "bias": bias, "threshold": threshold}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["splits"]["test"], ensure_ascii=False))
    print("误判样例:", report.get("example"))


if __name__ == "__main__":
    main()
