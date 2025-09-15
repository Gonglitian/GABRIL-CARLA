import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def iter_route_seed_paths(root: str) -> Iterable[Tuple[str, str, str]]:
    """Yield (route, seed, seed_path) for all route_*/seed_* under root."""
    if not os.path.isdir(root):
        return
    for route_name in sorted(os.listdir(root)):
        route_path = os.path.join(root, route_name)
        if not os.path.isdir(route_path) or not route_name.startswith("route_"):
            continue
        for seed_name in sorted(os.listdir(route_path)):
            seed_path = os.path.join(route_path, seed_name)
            if not os.path.isdir(seed_path) or not seed_name.startswith("seed_"):
                continue
            yield route_name, seed_name, seed_path


def safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # Malformed file; skip gracefully
        return None


def summarize_grounding(grounding: dict) -> Tuple[int, Counter]:
    """Return (total_detections, class_counts) from grounding_detections.json structure."""
    total = 0
    cls = Counter()
    if not grounding:
        return total, cls
    frames = grounding.get("frame_detections") or []
    for frame in frames:
        detections = frame.get("detections") or []
        total += len(detections)
        for det in detections:
            label = det.get("label") or "__unknown__"
            cls[label] += 1
    return total, cls


def summarize_vlm_filtered(vlm_filtered: dict) -> Tuple[int, Counter]:
    """Return (total_filtered, class_counts) from vlm_filtered_boxes.json structure."""
    total = 0
    cls = Counter()
    if not vlm_filtered:
        return total, cls
    results = vlm_filtered.get("results") or []
    for frame in results:
        filtered = frame.get("filtered") or []
        total += len(filtered)
        for det in filtered:
            label = det.get("label") or "__unknown__"
            cls[label] += 1
    return total, cls


def analyze_dataset(root: str) -> dict:
    """
    Analyze the dataset rooted at `root`.

    Returns a dict with:
    - overall: {detections_total, vlm_filtered_total, detection_classes, vlm_filtered_classes}
    - per_route_seed: list of {route, seed, detections, vlm_filtered}
    - missing: list of seed paths missing expected files
    """
    overall_detection_total = 0
    overall_vlm_total = 0
    overall_detection_classes: Counter = Counter()
    overall_vlm_classes: Counter = Counter()

    per_route_seed: List[Dict] = []
    filtered_records: List[Dict] = []
    missing: List[Dict] = []

    for route, seed, seed_path in iter_route_seed_paths(root):
        grounding_path = os.path.join(seed_path, "grounding_detections.json")
        vlm_filtered_path = os.path.join(seed_path, "vlm_filtered_boxes.json")

        grounding = safe_read_json(grounding_path)
        vlm_filtered = safe_read_json(vlm_filtered_path)

        det_total, det_cls = summarize_grounding(grounding)
        vlm_total, vlm_cls = summarize_vlm_filtered(vlm_filtered)

        # Track missing files explicitly
        if grounding is None or vlm_filtered is None:
            missing.append(
                {
                    "route": route,
                    "seed": seed,
                    "grounding_exists": grounding is not None,
                    "vlm_filtered_exists": vlm_filtered is not None,
                }
            )

        overall_detection_total += det_total
        overall_vlm_total += vlm_total
        overall_detection_classes.update(det_cls)
        overall_vlm_classes.update(vlm_cls)

        per_route_seed.append(
            {
                "route": route,
                "seed": seed,
                "detections": det_total,
                "vlm_filtered": vlm_total,
            }
        )

        # Collect per-frame filtered details for filtered.json
        if vlm_filtered:
            results = vlm_filtered.get("results") or []
            for frame_index, frame in enumerate(results):
                filtered = frame.get("filtered") or []
                if not filtered:
                    continue
                labels = [det.get("label") or "__unknown__" for det in filtered]
                filtered_records.append(
                    {
                        "route": route,
                        "seed": seed,
                        "frame_index": frame_index,
                        "labels": labels,
                        "num_filtered": len(labels),
                    }
                )

    # Build result
    result = {
        "root": root,
        "overall": {
            "detections_total": overall_detection_total,
            "vlm_filtered_total": overall_vlm_total,
            "detection_classes": dict(sorted(overall_detection_classes.items(), key=lambda x: (-x[1], x[0]))),
            "vlm_filtered_classes": dict(sorted(overall_vlm_classes.items(), key=lambda x: (-x[1], x[0]))),
        },
        "per_route_seed": per_route_seed,
        "missing": missing,
        "filtered_records": filtered_records,
    }
    return result


def print_summary(summary: dict, top_k: Optional[int] = None) -> None:
    overall = summary.get("overall", {})
    det_total = overall.get("detections_total", 0)
    vlm_total = overall.get("vlm_filtered_total", 0)

    print("Dataset root:", summary.get("root"))
    print("Total detections:", det_total)
    print("Total VLM filtered:", vlm_total)

    det_classes = overall.get("detection_classes", {})
    vlm_classes = overall.get("vlm_filtered_classes", {})

    def maybe_top(d: Dict[str, int]) -> List[Tuple[str, int]]:
        items = sorted(d.items(), key=lambda x: (-x[1], x[0]))
        return items[:top_k] if top_k else items

    print("\nDetection class counts:")
    for label, count in maybe_top(det_classes):
        print(f"  {label}: {count}")

    print("\nVLM-filtered class counts:")
    for label, count in maybe_top(vlm_classes):
        print(f"  {label}: {count}")

    missing = summary.get("missing", [])
    if missing:
        print(f"\nMissing/invalid files: {len(missing)} cases")
        for m in missing:
            print(
                f"  {m['route']}/{m['seed']}: grounding={'ok' if m['grounding_exists'] else 'missing'}, "
                f"vlm={'ok' if m['vlm_filtered_exists'] else 'missing'}"
            )


def save_json(summary: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_csv(per_route_seed: List[Dict], out_path: str) -> None:
    import csv

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["route", "seed", "detections", "vlm_filtered"]
        )
        writer.writeheader()
        for row in per_route_seed:
            writer.writerow(row)


def save_filtered(filtered_records: List[Dict], root: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "root": root,
        "records": filtered_records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize detections and VLM-filtered boxes across all route/seed "
            "under the given dataset root."
        )
    )
    parser.add_argument(
        "--root",
        default="/data3/vla-reasoning/saliency_exp_results",
        help="Dataset root containing route_*/seed_*",
    )
    parser.add_argument(
        "--save-json",
        dest="save_json_path",
        default=None,
        help="Optional path to save full summary JSON",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Optional path to save per-route/seed CSV",
    )
    parser.add_argument(
        "--save-filtered",
        dest="save_filtered_path",
        default=None,
        help="Optional path to save per-frame VLM filtered details as JSON",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Optionally limit printed class lists to top-K",
    )
    args = parser.parse_args()

    summary = analyze_dataset(args.root)
    print_summary(summary, top_k=args.top_k)

    if args.save_json_path:
        save_json(summary, args.save_json_path)
    if args.csv_path:
        save_csv(summary.get("per_route_seed", []), args.csv_path)
    if args.save_filtered_path:
        save_filtered(summary.get("filtered_records", []), summary.get("root", args.root), args.save_filtered_path)


if __name__ == "__main__":
    main()
