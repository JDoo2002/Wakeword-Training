# Custom Wake Word Training

This repository contains the private recording, training, evaluation, and live-testing tools for a custom OpenWakeWord model. The main application only receives the final production ONNX model.

## Repository layout

```text
models/             Trained ONNX/TFLite models and backups
openwakeword/       Python evaluation and live-test scripts
recorder/           Rust microphone recording utility
training-data/      Local positive, sentence, and negative recordings
```

WAV recordings and ZIP datasets are ignored by Git so personal voice recordings are not committed.

## Setup and testing

Install Python 3.13, then launch the interactive menu from the repository root:

```powershell
py run.py
```

Choose a task by number and answer the prompts. Press Enter to accept a value shown in brackets.

Skip the menu with `py run.py livetest`, `py run.py setup`, `py run.py evaluate`, `py run.py record`, or `py run.py test`. Each task still prompts for its settings.

- **Set up Python tools** creates `.venv`, installs dependencies, and downloads OpenWakeWord's supporting models, including the melspectrogram and embedding models. Run this once and let it finish.
- **Test the model live** prompts for the model path, wake-word display name, and confidence threshold. The default model is `models/wakeword.onnx`. Press Ctrl+C to stop listening.
- **Evaluate saved recordings** prompts for the model and recordings directory.
- **Record samples** prompts for isolated words, sentence starters, or negative phrases, followed by the sample count. Recordings are saved under `training-data/`. This task requires Rust and Cargo.
- **Run Rust unit tests** runs the recorder's tests using Cargo. Rust's package manifest is `recorder/Cargo.toml`.

The wake-word name changes the displayed label only. The loaded model determines which phrase is detected; choose a model trained for your desired phrase.

## Train a new model

Training currently uses the official OpenWakeWord custom-model Colab notebook. See [openwakeword/README.md](openwakeword/README.md) for the training process.

After evaluating a new model, copy the approved ONNX file into your application's model directory.

In recorder prompts, substitute your chosen phrase for `<wake word>`. Customize the negative phrases in `recorder/src/main.rs` for similar-sounding words. Renaming an existing model does not change the phrase it detects.
