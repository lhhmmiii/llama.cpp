import json
from pathlib import Path

from safetensors import safe_open


MODEL_DIR = Path("./models/Zamba2-1.2B")
CONFIG_PATH = MODEL_DIR / "config.json"
SAFETENSORS_PATH = MODEL_DIR / "model.safetensors"


def inspect_config():
    print("=" * 100)
    print("CONFIG")
    print("=" * 100)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    for key, value in config.items():
        print(f"{key}: {value}")


def inspect_safetensors():
    print("\n")
    print("=" * 100)
    print("SAFETENSORS TENSORS")
    print("=" * 100)

    with safe_open(
        SAFETENSORS_PATH,
        framework="pt",
        device="cpu",
    ) as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            print(
                f"{key:80} "
                f"shape={str(tuple(tensor.shape)):25} "
                f"dtype={tensor.dtype}"
            )


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    if not SAFETENSORS_PATH.exists():
        raise FileNotFoundError(
            f"Safetensors not found: {SAFETENSORS_PATH}"
        )

    inspect_config()
    inspect_safetensors()


if __name__ == "__main__":
    main()