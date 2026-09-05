import argparse
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from scipy.signal import resample_poly


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Test an OpenWakeWord model live")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "wakeword.onnx",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--wakeword",
        help="Name shown in messages (defaults to the model filename; does not retrain the model)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")

    device = sd.query_devices(kind="input")
    input_rate = int(device["default_samplerate"])
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    model = Model(
        wakeword_models=[str(args.model)],
        inference_framework="onnx",
    )
    model_name = args.wakeword or args.model.stem
    pending = np.empty(0, dtype=np.int16)

    def callback(indata, frames, time, status) -> None:
        if status:
            print(f"Audio warning: {status}")
        audio_queue.put(indata[:, 0].copy())

    print(f"Microphone: {device['name']} ({input_rate} Hz)")
    print(f"Listening with {model_name} at threshold {args.threshold:.2f}. Ctrl+C stops.")

    try:
        with sd.InputStream(
            samplerate=input_rate,
            channels=1,
            dtype="int16",
            blocksize=max(1, input_rate // 10),
            callback=callback,
        ):
            while True:
                chunk = audio_queue.get()
                if input_rate != 16000:
                    chunk = resample_poly(chunk, 16000, input_rate).astype(np.int16)
                pending = np.concatenate((pending, chunk))

                while pending.size >= 1280:
                    frame, pending = pending[:1280], pending[1280:]
                    predictions = model.predict(frame)
                    score = float(next(iter(predictions.values())))
                    if score >= args.threshold:
                        print(f"Wake word detected: {model_name} ({score:.3f})")
                        model.reset()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
