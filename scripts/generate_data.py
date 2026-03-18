"""Command-line entry point for reproducible Lorenz dataset generation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics.dataset_generation import generate_dataset_bundle, save_dataset_bundle
from src.utils.config import build_config_argparser, load_config
from src.utils.seeds import set_seed


def main() -> None:
    parser = build_config_argparser("Generate Lorenz simulation datasets.")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    bundle = generate_dataset_bundle(config)
    save_dataset_bundle(bundle, config.get("output_root", "results/data"))


if __name__ == "__main__":
    main()
