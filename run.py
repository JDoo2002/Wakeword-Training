"""Interactive setup, recording, and testing menu. Run with py run.py."""

from pathlib import Path
import argparse
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def choose(prompt: str, options: dict[str, str]) -> str:
    for key, label in options.items():
        print(f"  {key}. {label}")
    while True:
        value = input(prompt).strip()
        if value in options:
            return value
        print("Enter one of the numbers above.")


def ask(prompt: str, default: str) -> str:
    return input(f"{prompt} [{default}]: ").strip() or default


def interactive_task(task: str | None = None) -> tuple[str, list[str]]:
    if task is None:
        print("\nWake Word Training\n")
        selection = choose("Choose a task: ", {
            "1": "Set up Python tools and download supporting models",
            "2": "Test the model live",
            "3": "Evaluate saved recordings",
            "4": "Record samples",
            "5": "Run Rust unit tests",
            "0": "Exit",
        })
        task = {"1": "setup", "2": "live", "3": "evaluate", "4": "record", "5": "test", "0": "exit"}[selection]
    if task == "livetest":
        task = "live"
    args: list[str] = []
    if task in ("live", "evaluate"):
        model = ask("Model path", "models/wakeword.onnx")
        args = ["--model", model]
        if task == "live":
            print("The display name does not change the phrase the model detects.")
            name = ask("Wake-word display name", Path(model).stem)
            while True:
                threshold = ask("Confidence threshold (0 to 1)", "0.5")
                try:
                    if 0 <= float(threshold) <= 1:
                        break
                except ValueError:
                    pass
                print("Enter a number between 0 and 1.")
            args += ["--wakeword", name, "--threshold", threshold]
        else:
            args += ["--samples", ask("Recordings directory", "training-data/positive")]
    elif task == "record":
        mode = choose("Recording mode: ", {"1": "Isolated wake word", "2": "Sentence starters", "3": "Negative phrases"})
        while True:
            count = ask("Number of samples", "30")
            if count.isascii() and count.isdecimal() and int(count) > 0:
                break
            print("Enter a positive whole number.")
        args = {"1": [], "2": ["sentences"], "3": ["negatives"]}[mode] + [count]
    return task, args


def main() -> None:
    parser = argparse.ArgumentParser(description="Wake-word tools with interactive prompts")
    parser.add_argument("task", nargs="?", choices=("setup", "livetest", "live", "evaluate", "record", "test"))
    task, args = interactive_task(parser.parse_args().task)
    if task == "setup":
        if not PYTHON.is_file():
            run(sys.executable, "-m", "venv", str(ROOT / ".venv"))
        run(str(PYTHON), "-m", "pip", "install", "-r", "openwakeword/requirements.txt")
        run(str(PYTHON), "-c", "from openwakeword.utils import download_models; download_models()")
    elif task in ("live", "evaluate"):
        if not PYTHON.is_file():
            raise SystemExit("Python environment missing. Run py run.py and choose setup first.")
        script = "live_test.py" if task == "live" else "evaluate.py"
        run(str(PYTHON), str(ROOT / "openwakeword" / script), *args)
    elif task == "record":
        run("cargo", "run", "--manifest-path", "recorder/Cargo.toml", "--", *args)
    elif task == "test":
        run("cargo", "test", "--manifest-path", "recorder/Cargo.toml", *args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
    except FileNotFoundError as error:
        sys.exit(f"Command not found: {error.filename}. Check that the required tool is installed.")
    except KeyboardInterrupt:
        sys.exit(130)
    except EOFError:
        sys.exit("\nInput closed.")
