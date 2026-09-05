import argparse
from pathlib import Path

from openwakeword.model import Model


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate an wake word OpenWakeWord model")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "wakeword.onnx",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=root / "training-data" / "positive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")

    sample_paths = sorted(args.samples.glob("*.wav"))
    if not sample_paths:
        raise SystemExit(f"No WAV samples found in: {args.samples}")

    model = Model(wakeword_models=[str(args.model)], inference_framework="onnx")
    scores: list[tuple[Path, float]] = []

    for sample_path in sample_paths:
        predictions = model.predict_clip(str(sample_path), chunk_size=1280)
        model.reset()
        score = max(
            (float(value) for frame in predictions for value in frame.values()),
            default=0.0,
        )
        scores.append((sample_path, score))
        print(f"{sample_path.name}: {score:.4f}")

    print("\nRecall on held-out wake word recordings:")
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5):
        detected = sum(score >= threshold for _, score in scores)
        recall = detected / len(scores) * 100
        print(f"  threshold {threshold:.1f}: {detected}/{len(scores)} ({recall:.1f}%)")


if __name__ == "__main__":
    main()
