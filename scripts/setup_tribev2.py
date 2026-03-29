import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tribe_helpers import check_tribev2, load_model


def main():
    parser = argparse.ArgumentParser(description="Setup TRIBE v2 model")
    parser.add_argument(
        "--cache-dir", type=str, default="./cache",
        help="Directory to cache model weights and features (default: ./cache)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Torch device: 'auto', 'cpu', or 'cuda' (default: auto)",
    )
    args = parser.parse_args()

    check_tribev2()

    model = load_model(cache_dir=args.cache_dir, device=args.device)
    print(f"model loaded. output vertices: {model._model.n_outputs}")



if __name__ == "__main__":
    main()
