from __future__ import annotations
import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class YoloObject:
    """YOLO-format bounding box: class_id, x_center, y_center, width, height.
    All coordinates are normalized to [0, 1].
    """
    cls: int
    x: float
    y: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.cls} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}"


def read_yolo_label(label_path: Path) -> List[YoloObject]:
    objects: List[YoloObject] = []

    if not label_path.exists():
        return objects

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                continue

            objects.append(YoloObject(cls=cls, x=x, y=y, w=w, h=h))

    return objects


def write_yolo_label(label_path: Path, objects: List[YoloObject]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with label_path.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(obj.to_line() + "\n")


def find_image_for_label(label_path: Path, image_dir: Path) -> Optional[Path]:
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    for ext in image_exts:
        candidate = image_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def collect_label_files(label_dir: Path) -> List[Path]:
    return sorted(label_dir.glob("*.txt"))


def compute_class_image_counts(label_files: List[Path]) -> Dict[int, int]:
    """Count how many images contain each category.

    This image-level category count is used as Ni in the EDC calculation.
    """
    counts: Dict[int, int] = {}

    for label_path in label_files:
        objects = read_yolo_label(label_path)
        classes_in_image = sorted(set(obj.cls for obj in objects))

        for cls in classes_in_image:
            counts[cls] = counts.get(cls, 0) + 1

    return counts


def build_candidate_index(label_files: List[Path]) -> Dict[int, List[Path]]:
    """Build class_id -> label files containing this class."""
    index: Dict[int, List[Path]] = {}

    for label_path in label_files:
        objects = read_yolo_label(label_path)
        classes_in_image = sorted(set(obj.cls for obj in objects))

        for cls in classes_in_image:
            index.setdefault(cls, []).append(label_path)

    return index


# -----------------------------
# EDC calculation
# -----------------------------

def compute_edc(
    class_ap: Dict[int, float],
    map_value: float,
    class_counts: Dict[int, int],
    total_train_samples: int,
) -> List[Dict[str, float]]:
    """Compute Enhancement Demand Coefficient for each category.

    EDC_i = D_i * S_i, if AP_i < mAP
          = 0,       if AP_i >= mAP

    D_i = (mAP - AP_i) / mAP
    S_i = 1 + log10(N_train / N_i)

    Parameters
    ----------
    class_ap:
        Category-level AP values. AP can be in [0, 1] or [0, 100], as long as
        it is consistent with map_value.
    map_value:
        Overall mAP value.
    class_counts:
        Image-level sample count of each category.
    total_train_samples:
        Number of training samples.

    Returns
    -------
    rows:
        A list of dictionaries containing AP, Ni, D_i, S_i, gamma_i, and raw_add.
    """
    rows: List[Dict[str, float]] = []

    if map_value <= 0:
        raise ValueError("mAP must be positive.")

    for cls in sorted(class_counts.keys()):
        ni = max(int(class_counts[cls]), 1)
        ap_i = float(class_ap.get(cls, map_value))

        if ap_i < map_value:
            deficiency = (map_value - ap_i) / map_value
            scarcity = 1.0 + math.log10(max(total_train_samples, 1) / ni)
            gamma = deficiency * scarcity
        else:
            deficiency = 0.0
            scarcity = 1.0 + math.log10(max(total_train_samples, 1) / ni)
            gamma = 0.0

        raw_add = int(round(gamma * ni))

        rows.append(
            {
                "class_id": cls,
                "AP_i": ap_i,
                "mAP": map_value,
                "N_i": ni,
                "D_i": deficiency,
                "S_i": scarcity,
                "gamma_i": gamma,
                "raw_add": raw_add,
            }
        )

    return rows


def allocate_with_budget(
    edc_rows: List[Dict[str, float]],
    target_total: Optional[int] = None,
) -> Dict[int, int]:
    """Convert EDC scores into category-wise additional sample counts.

    If target_total is None, this function uses raw_add = round(gamma_i * N_i).
    If target_total is given, the total number of augmented samples is fixed
    and distributed according to gamma_i * N_i.
    """
    if target_total is None:
        return {
            int(row["class_id"]): int(row["raw_add"])
            for row in edc_rows
            if int(row["raw_add"]) > 0
        }

    target_total = int(target_total)
    if target_total <= 0:
        return {}

    scores: Dict[int, float] = {}
    for row in edc_rows:
        cls = int(row["class_id"])
        score = float(row["gamma_i"]) * float(row["N_i"])
        if score > 0:
            scores[cls] = score

    score_sum = sum(scores.values())
    if score_sum <= 0:
        return {}

    exact_alloc = {
        cls: target_total * score / score_sum
        for cls, score in scores.items()
    }

    allocation = {
        cls: int(math.floor(value))
        for cls, value in exact_alloc.items()
    }

    remaining = target_total - sum(allocation.values())

    remainders = sorted(
        exact_alloc.items(),
        key=lambda kv: kv[1] - math.floor(kv[1]),
        reverse=True,
    )

    for cls, _ in remainders[:remaining]:
        allocation[cls] += 1

    return {cls: count for cls, count in allocation.items() if count > 0}


# -----------------------------
# TCC calculation
# -----------------------------

def compute_tcc_for_class(objects: List[YoloObject], focus_class: int) -> float:
    """Compute target coverage coefficient for a selected category.

    For YOLO normalized boxes, bbox coverage equals width * height.
    When multiple instances of the target class exist in the same image, the
    mean coverage is used.
    """
    coverages = [
        max(obj.w, 0.0) * max(obj.h, 0.0)
        for obj in objects
        if obj.cls == focus_class
    ]

    if not coverages:
        return 0.0

    return float(sum(coverages) / len(coverages))


def get_tcc_group(
    tcc: float,
    small_threshold: float = 0.10,
    large_threshold: float = 0.50,
) -> str:
    if tcc < small_threshold:
        return "small"
    if tcc < large_threshold:
        return "medium"
    return "large"


def choose_scale_aware_operation(
    tcc: float,
    rng: random.Random,
    small_threshold: float = 0.10,
    large_threshold: float = 0.50,
) -> str:
    """Select an augmentation operation according to TCC group.

    The selected operations are deliberately lightweight and annotation-safe.
    Geometric operations that change boxes are implemented with corresponding
    label transformation.
    """
    group = get_tcc_group(tcc, small_threshold, large_threshold)

    if group == "small":
        operations = [
            "local_sharpen",
            "contrast_enhance",
            "brightness_adjust",
            "horizontal_flip",
        ]
    elif group == "medium":
        operations = [
            "contrast_enhance",
            "brightness_adjust",
            "horizontal_flip",
            "rotate_pos",
            "rotate_neg",
        ]
    else:
        operations = [
            "brightness_adjust",
            "contrast_enhance",
            "horizontal_flip",
            "vertical_flip",
            "rotate_pos",
            "rotate_neg",
        ]

    return rng.choice(operations)


# -----------------------------
# Label-safe image augmentations
# -----------------------------

def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def transform_horizontal_flip(objects: List[YoloObject]) -> List[YoloObject]:
    return [
        YoloObject(
            cls=obj.cls,
            x=clip01(1.0 - obj.x),
            y=obj.y,
            w=obj.w,
            h=obj.h,
        )
        for obj in objects
    ]


def transform_vertical_flip(objects: List[YoloObject]) -> List[YoloObject]:
    return [
        YoloObject(
            cls=obj.cls,
            x=obj.x,
            y=clip01(1.0 - obj.y),
            w=obj.w,
            h=obj.h,
        )
        for obj in objects
    ]


def rotate_single_box(obj: YoloObject, angle_degree: float) -> YoloObject:
    """Approximate bbox update after image-center rotation.

    Coordinates are normalized. The new bbox is obtained by rotating the four
    corners and taking the enclosing rectangle.
    """
    angle = math.radians(angle_degree)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    x1 = obj.x - obj.w / 2.0
    y1 = obj.y - obj.h / 2.0
    x2 = obj.x + obj.w / 2.0
    y2 = obj.y + obj.h / 2.0

    corners = [
        (x1, y1),
        (x1, y2),
        (x2, y1),
        (x2, y2),
    ]

    rotated = []
    for x, y in corners:
        x0 = x - 0.5
        y0 = y - 0.5
        xr = cos_a * x0 - sin_a * y0 + 0.5
        yr = sin_a * x0 + cos_a * y0 + 0.5
        rotated.append((clip01(xr), clip01(yr)))

    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]

    nx1, nx2 = min(xs), max(xs)
    ny1, ny2 = min(ys), max(ys)

    nw = max(nx2 - nx1, 1e-6)
    nh = max(ny2 - ny1, 1e-6)
    nx = clip01((nx1 + nx2) / 2.0)
    ny = clip01((ny1 + ny2) / 2.0)

    return YoloObject(cls=obj.cls, x=nx, y=ny, w=clip01(nw), h=clip01(nh))


def transform_rotate(objects: List[YoloObject], angle_degree: float) -> List[YoloObject]:
    return [rotate_single_box(obj, angle_degree) for obj in objects]


def apply_augmentation(
    image: Image.Image,
    objects: List[YoloObject],
    operation: str,
    rng: random.Random,
) -> Tuple[Image.Image, List[YoloObject]]:
    """Apply augmentation and update labels when necessary."""
    image = image.convert("RGB")
    new_objects = list(objects)

    if operation == "local_sharpen":
        factor = rng.uniform(1.5, 2.5)
        image = ImageEnhance.Sharpness(image).enhance(factor)

    elif operation == "contrast_enhance":
        factor = rng.uniform(1.15, 1.45)
        image = ImageEnhance.Contrast(image).enhance(factor)

    elif operation == "brightness_adjust":
        factor = rng.uniform(0.75, 1.25)
        image = ImageEnhance.Brightness(image).enhance(factor)

    elif operation == "gaussian_blur":
        radius = rng.uniform(0.3, 0.8)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    elif operation == "horizontal_flip":
        image = ImageOps.mirror(image)
        new_objects = transform_horizontal_flip(objects)

    elif operation == "vertical_flip":
        image = ImageOps.flip(image)
        new_objects = transform_vertical_flip(objects)

    elif operation in {"rotate_pos", "rotate_neg"}:
        angle = rng.uniform(3.0, 8.0)
        if operation == "rotate_neg":
            angle = -angle

        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR

        image = image.rotate(
            angle,
            resample=resample,
            expand=False,
            fillcolor=(0, 0, 0),
        )
        new_objects = transform_rotate(objects, angle)

    else:
        raise ValueError(f"Unsupported augmentation operation: {operation}")

    return image, new_objects


# -----------------------------
# Dataset generation
# -----------------------------

def copy_original_split(
    image_dir: Path,
    label_dir: Path,
    out_image_dir: Path,
    out_label_dir: Path,
) -> None:
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            shutil.copy2(image_path, out_image_dir / image_path.name)

    for label_path in label_dir.glob("*.txt"):
        shutil.copy2(label_path, out_label_dir / label_path.name)


def load_metrics_json(metrics_path: Path) -> Tuple[float, Dict[int, float]]:
    """Load mAP and class-wise AP from JSON.

    Expected format:
    {
      "mAP": 90.0,
      "AP": {
        "0": 88.1,
        "1": 92.3
      }
    }
    """
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    map_value = data.get("mAP", data.get("map", None))
    if map_value is None:
        raise ValueError("metrics JSON must contain 'mAP' or 'map'.")

    ap_dict = data.get("AP", data.get("class_AP", None))
    if ap_dict is None:
        raise ValueError("metrics JSON must contain 'AP' or 'class_AP'.")

    class_ap = {int(k): float(v) for k, v in ap_dict.items()}

    return float(map_value), class_ap


def save_edc_table(edc_rows: List[Dict[str, float]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not edc_rows:
        return

    fieldnames = list(edc_rows[0].keys())

    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in edc_rows:
            writer.writerow(row)


def save_plan(plan_rows: List[Dict[str, object]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(plan_rows, f, indent=2, ensure_ascii=False)


def run_hierarchical_augmentation(
    data_root: Path,
    output_root: Path,
    metrics_path: Path,
    split: str = "train",
    target_total: Optional[int] = 96,
    seed: int = 42,
    small_threshold: float = 0.10,
    large_threshold: float = 0.50,
    copy_original: bool = True,
) -> None:
    rng = random.Random(seed)

    image_dir = data_root / "images" / split
    label_dir = data_root / "labels" / split

    out_image_dir = output_root / "images" / split
    out_label_dir = output_root / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    label_files = collect_label_files(label_dir)
    if not label_files:
        raise RuntimeError(f"No YOLO label files found in: {label_dir}")

    if copy_original:
        copy_original_split(image_dir, label_dir, out_image_dir, out_label_dir)

    total_train_samples = len(label_files)
    class_counts = compute_class_image_counts(label_files)
    map_value, class_ap = load_metrics_json(metrics_path)

    edc_rows = compute_edc(
        class_ap=class_ap,
        map_value=map_value,
        class_counts=class_counts,
        total_train_samples=total_train_samples,
    )

    allocation = allocate_with_budget(edc_rows, target_total=target_total)
    candidate_index = build_candidate_index(label_files)

    save_edc_table(edc_rows, output_root / "edc_table.csv")

    plan_rows: List[Dict[str, object]] = []
    generated_count = 0

    for cls, add_num in sorted(allocation.items()):
        candidates = candidate_index.get(cls, [])
        if not candidates:
            continue

        for idx in range(add_num):
            label_path = rng.choice(candidates)
            image_path = find_image_for_label(label_path, image_dir)

            if image_path is None:
                continue

            objects = read_yolo_label(label_path)
            if not objects:
                continue

            tcc = compute_tcc_for_class(objects, focus_class=cls)
            operation = choose_scale_aware_operation(
                tcc=tcc,
                rng=rng,
                small_threshold=small_threshold,
                large_threshold=large_threshold,
            )

            with Image.open(image_path) as img:
                aug_img, aug_objects = apply_augmentation(
                    image=img,
                    objects=objects,
                    operation=operation,
                    rng=rng,
                )

            out_stem = f"{image_path.stem}_aug_c{cls}_{idx:04d}_{operation}"
            out_image_path = out_image_dir / f"{out_stem}{image_path.suffix.lower()}"
            out_label_path = out_label_dir / f"{out_stem}.txt"

            aug_img.save(out_image_path)
            write_yolo_label(out_label_path, aug_objects)

            generated_count += 1

            plan_rows.append(
                {
                    "source_image": str(image_path.name),
                    "source_label": str(label_path.name),
                    "output_image": str(out_image_path.name),
                    "output_label": str(out_label_path.name),
                    "class_id": cls,
                    "tcc": round(tcc, 6),
                    "tcc_group": get_tcc_group(tcc, small_threshold, large_threshold),
                    "operation": operation,
                }
            )

    save_plan(plan_rows, output_root / "augmentation_plan.json")

    summary = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "split": split,
        "seed": seed,
        "target_total": target_total,
        "generated_count": generated_count,
        "small_threshold": small_threshold,
        "large_threshold": large_threshold,
        "allocation": allocation,
    }

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Hierarchical adaptive augmentation finished.")
    print(f"Generated augmented samples: {generated_count}")
    print(f"Output directory: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDC/TCC-guided hierarchical adaptive augmentation."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./CFD",
        help="Dataset root. Expected structure: images/train and labels/train.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="./CFD_augmented",
        help="Output root for the augmented dataset.",
    )

    parser.add_argument(
        "--metrics-json",
        type=str,
        default="./metrics_cfd.json",
        help="JSON file containing mAP and class-wise AP.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to augment, usually train.",
    )

    parser.add_argument(
        "--target-total",
        type=int,
        default=96,
        help="Total number of augmented samples. Use -1 to disable fixed budget.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--small-threshold",
        type=float,
        default=0.10,
        help="TCC threshold between small and medium defects.",
    )

    parser.add_argument(
        "--large-threshold",
        type=float,
        default=0.50,
        help="TCC threshold between medium and large defects.",
    )

    parser.add_argument(
        "--no-copy-original",
        action="store_true",
        help="Only save augmented samples, without copying original training samples.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_total: Optional[int]
    if args.target_total < 0:
        target_total = None
    else:
        target_total = args.target_total

    run_hierarchical_augmentation(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        metrics_path=Path(args.metrics_json),
        split=args.split,
        target_total=target_total,
        seed=args.seed,
        small_threshold=args.small_threshold,
        large_threshold=args.large_threshold,
        copy_original=not args.no_copy_original,
    )


if __name__ == "__main__":
    main()
