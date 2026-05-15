import argparse
import csv
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from datasets.mvtec import MyData
from datasets.visa import ViSA
from models.backbone import get_backbone
from models.patchcore import PatchCore
from models.winclip import WinCLIP
from utils.visualization import save_heatmap


MVTEC_CLASSES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

VISA_CLASSES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly detection experiments for MVTec and ViSA.")
    parser.add_argument("--dataset", choices=["mvtec", "visa", "all"], default="all")
    parser.add_argument("--model", choices=["patchcore", "winclip"], default="patchcore")
    parser.add_argument("--classes", nargs="+", default=["all"])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--k", type=int, default=10000, help="PatchCore memory bank coreset size.")
    parser.add_argument("--backbone", default="resnet18", help="PatchCore backbone: resnet18 or wide_resnet50_2.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--test-limit-per-class", type=int, default=None)

    parser.add_argument("--topk-heatmaps", type=int, default=8)
    parser.add_argument("--save-all-heatmaps", action="store_true")
    parser.add_argument("--no-pixel-auroc", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def selected_classes(dataset_name, requested):
    all_classes = MVTEC_CLASSES if dataset_name == "mvtec" else VISA_CLASSES
    if requested == ["all"]:
        return all_classes
    unknown = sorted(set(requested) - set(all_classes))
    if unknown:
        raise ValueError(f"Unknown {dataset_name} classes: {unknown}")
    return requested


def build_model(args, category, device):
    if args.model == "patchcore":
        backbone = get_backbone(args.backbone)
        return PatchCore(backbone, k=args.k, device=device)
    return WinCLIP(
        category=category,
        device=device,
        batch_size=args.batch_size,
        use_visual_gallery=True,
    )


def load_mask(mask_path, image_size):
    if mask_path is None:
        return torch.zeros(image_size, dtype=torch.uint8)
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((image_size[1], image_size[0]), Image.NEAREST)
    mask = torch.from_numpy(np.array(mask))
    return (mask > 0).to(torch.uint8)


def upsample_heatmap(heatmap, image_size):
    heatmap = heatmap.detach().cpu().float()
    return F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze()


def evaluate_items(model, datasets, out_dir, save_all_heatmaps, topk_heatmaps, compute_pixel_auroc):
    scores = []
    labels = []
    image_rows = []
    pixel_targets = []
    pixel_scores = []
    heatmap_records = []

    for dataset in datasets:
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            score, heatmap = model.predict(img)
            score_value = float(score.detach().cpu())
            is_anomaly = int(dataset.is_anomaly_label(label))
            image_path = dataset.get_image_path(idx)
            mask_path = dataset.get_mask_path(idx)

            scores.append(score_value)
            labels.append(is_anomaly)
            image_rows.append(
                {
                    "image_path": str(image_path),
                    "mask_path": "" if mask_path is None else str(mask_path),
                    "label": is_anomaly,
                    "score": score_value,
                }
            )
            heatmap_records.append((score_value, dataset, idx, img, int(label), heatmap))

            if compute_pixel_auroc:
                image_size = tuple(img.shape[-2:])
                mask = load_mask(mask_path, image_size)
                resized_heatmap = upsample_heatmap(heatmap, image_size)
                pixel_targets.append(mask.flatten().numpy())
                pixel_scores.append(resized_heatmap.flatten().numpy())

    image_auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    pixel_auroc = float("nan")
    if compute_pixel_auroc and pixel_targets:
        y_true = np.concatenate(pixel_targets)
        y_score = np.concatenate(pixel_scores)
        if len(np.unique(y_true)) > 1:
            pixel_auroc = roc_auc_score(y_true, y_score)

    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    heatmap_records = sorted(heatmap_records, key=lambda item: item[0], reverse=True)
    records_to_save = heatmap_records if save_all_heatmaps else heatmap_records[:topk_heatmaps]
    for rank, (score, dataset, idx, img, label, heatmap) in enumerate(records_to_save, start=1):
        label_name = dataset.get_classes()[label]
        image_stem = dataset.get_image_path(idx).stem
        save_path = heatmap_dir / f"{rank:04d}_{score:.4f}_{label_name}_{image_stem}.png"
        save_heatmap(img, heatmap, save_path, score=score, label_name=label_name)

    with (out_dir / "image_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "mask_path", "label", "score"])
        writer.writeheader()
        writer.writerows(image_rows)

    return image_auroc, pixel_auroc, scores, [record[5] for record in heatmap_records]


def run_mvtec_class(args, category, device, out_dir):
    train_data = MyData(
        category,
        phase="train",
        batch_size=args.batch_size,
        shuffle=False,
        limit=args.train_limit,
    )
    test_data = MyData(
        category,
        phase="test",
        batch_size=args.batch_size,
        shuffle=False,
        limit=args.test_limit,
        limit_per_class=args.test_limit_per_class,
    )

    model = build_model(args, category, device)
    model.fit(train_data)
    image_auroc, pixel_auroc, _, _ = evaluate_items(
        model,
        [test_data],
        out_dir,
        args.save_all_heatmaps,
        args.topk_heatmaps,
        not args.no_pixel_auroc,
    )

    return image_auroc, pixel_auroc, len(train_data), len(test_data)


def run_visa_class(args, category, device, out_dir):
    train_data = ViSA(
        category,
        phase="Normal",
        batch_size=args.batch_size,
        shuffle=False,
        limit=args.train_limit,
    )
    test_normal = ViSA(
        category,
        phase="Normal",
        batch_size=args.batch_size,
        shuffle=False,
        limit=args.test_limit,
    )
    test_anomaly = ViSA(
        category,
        phase="Anomaly",
        batch_size=args.batch_size,
        shuffle=False,
        limit=args.test_limit,
    )

    model = build_model(args, category, device)
    model.fit(train_data)
    image_auroc, pixel_auroc, _, _ = evaluate_items(
        model,
        [test_normal, test_anomaly],
        out_dir,
        args.save_all_heatmaps,
        args.topk_heatmaps,
        not args.no_pixel_auroc,
    )
    return image_auroc, pixel_auroc, len(train_data), len(test_normal) + len(test_anomaly)


def write_summary(output_dir, rows):
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "class",
                "model",
                "backbone",
                "k",
                "train_count",
                "test_count",
                "image_auroc",
                "pixel_auroc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = ["mvtec", "visa"] if args.dataset == "all" else [args.dataset]
    rows = []
    print(f"device: {device}")

    for dataset_name in datasets_to_run:
        for category in selected_classes(dataset_name, args.classes):
            class_out_dir = output_dir / dataset_name / category
            class_out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{dataset_name}/{category}] start")

            if dataset_name == "mvtec":
                image_auroc, pixel_auroc, train_count, test_count = run_mvtec_class(args, category, device, class_out_dir)
            else:
                image_auroc, pixel_auroc, train_count, test_count = run_visa_class(args, category, device, class_out_dir)

            row = {
                "dataset": dataset_name,
                "class": category,
                "model": args.model,
                "backbone": args.backbone if args.model == "patchcore" else "ViT-B-16-plus-240",
                "k": args.k if args.model == "patchcore" else "",
                "train_count": train_count,
                "test_count": test_count,
                "image_auroc": image_auroc,
                "pixel_auroc": pixel_auroc,
            }
            rows.append(row)
            write_summary(output_dir, rows)
            print(f"[{dataset_name}/{category}] image_auroc={image_auroc:.4f}, pixel_auroc={pixel_auroc:.4f}")

    summary_path = write_summary(output_dir, rows)
    print(f"summary saved: {summary_path}")


if __name__ == "__main__":
    main()
