# Custom Wake Word Training

The first model is trained synthetically in OpenWakeWord's official Google Colab notebook. The 30 real recordings are kept out of training so they provide an honest accent-specific evaluation set.

## 1. Train the first model

1. Open the official notebook:
   https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing
2. Make a copy in Google Drive.
3. Enable a GPU runtime.
4. Use your chosen wake word as the target phrase.
5. Run every cell in order.
6. Download the resulting ONNX model.
7. Save it as `models/wakeword.onnx`.

Do not upload `wakeword-evaluation-samples.zip` for the first training run. Those recordings are our held-out test set.

## 2. Set up and evaluate the model

From the repository root, launch the interactive menu:

```powershell
py run.py
```

Choose **Set up Python tools** once to create the environment, install dependencies, and download the supporting feature models. Let setup finish, then launch the menu again and choose **Evaluate saved recordings**. Enter the model path and sample directory when prompted.

The evaluator prints every recording's maximum confidence and recall at several thresholds. A useful model needs high recall without becoming sensitive to unrelated speech; negative testing comes after this first positive test.

## Live test

Launch the same menu and choose **Test the model live**. Enter the model path, displayed wake-word name, and confidence threshold, or press Enter to use each default. The name is a label; the trained model determines the detected phrase.

You can also launch the live-test prompts directly with `py run.py livetest`.

## Record personal samples

Choose **Record samples** in the menu, then select the recording mode and sample count. Rust and Cargo are required. The recorder saves files under `training-data/`.
