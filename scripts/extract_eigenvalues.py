import argparse
from pathlib import Path

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
    extract_eigenvalues_from_spectrum(args.spectrum, args.output_dir, threshold=args.threshold, window=1)


if __name__ == "__main__":
    main()
