import argparse
import sys
from pathlib import Path

# Support both direct execution and module-style execution
if __name__ == "__main__" and __package__ is None:
    # Add parent directory to path for direct script execution
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from utils.eigenvalues import extract_eigenvalues_from_spectrum


def parse_args():
    parser = argparse.ArgumentParser(description="Extract eigenvalues from spectrum.npz")
    parser.add_argument("--spectrum", type=Path, required=True, help="Path to spectrum.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("output_values"), help="Directory to write eigenvalues.csv")
    parser.add_argument(
        "--threshold",
        type=float,
        default=float("inf"),
        help="Maximum chi^2 to accept for minima (set to inf to keep all minima)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extract_eigenvalues_from_spectrum(args.spectrum, args.output_dir, threshold=args.threshold, refine=True)


if __name__ == "__main__":
    main()
