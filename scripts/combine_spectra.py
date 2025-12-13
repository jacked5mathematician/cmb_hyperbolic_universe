#!/usr/bin/env python
"""
Combine spectrum chunks from HPC runs into a single spectrum.npz file.

Usage:
    python scripts/combine_spectra.py --input-dir output_values --output-file spectrum.npz

This script:
1. Finds all chunk_*/spectrum.npz files in the input directory
2. Loads and sorts by k_values
3. Concatenates all arrays
4. Saves combined result to output-file
"""
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def find_chunk_files(input_dir: Path) -> List[Tuple[int, Path]]:
    """Find all spectrum.npz files in chunk_* subdirectories."""
    chunks = []
    for chunk_dir in input_dir.glob("chunk_*"):
        if not chunk_dir.is_dir():
            continue
        try:
            chunk_index = int(chunk_dir.name.split("_")[1])
        except (IndexError, ValueError):
            LOGGER.warning("Skipping directory with invalid chunk name: %s", chunk_dir)
            continue
        
        spectrum_file = chunk_dir / "spectrum.npz"
        if spectrum_file.exists():
            chunks.append((chunk_index, spectrum_file))
        else:
            LOGGER.warning("No spectrum.npz found in %s", chunk_dir)
    
    return sorted(chunks, key=lambda x: x[0])


def load_and_validate_chunk(path: Path) -> Dict[str, np.ndarray]:
    """Load a spectrum chunk and validate it has required fields."""
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    
    if "k_values" not in arrays:
        raise ValueError(f"Missing k_values in {path}")
    
    return arrays


def combine_chunks(chunk_files: List[Tuple[int, Path]]) -> Dict[str, np.ndarray]:
    """Combine multiple spectrum chunks into one."""
    if not chunk_files:
        raise ValueError("No chunk files provided")
    
    # Load all chunks
    chunks_data = []
    for chunk_idx, path in chunk_files:
        LOGGER.info("Loading chunk %d from %s", chunk_idx, path)
        chunks_data.append(load_and_validate_chunk(path))
    
    # Get all keys from first chunk
    keys = set(chunks_data[0].keys())
    
    # Verify all chunks have the same keys
    for i, chunk in enumerate(chunks_data[1:], 1):
        if set(chunk.keys()) != keys:
            LOGGER.warning("Chunk %d has different keys: %s vs %s", 
                          i, set(chunk.keys()), keys)
            keys = keys.intersection(set(chunk.keys()))
    
    # Sort chunks by k_values to ensure correct ordering
    # (should already be sorted by chunk index, but verify)
    k_sorted_indices = np.argsort([chunk["k_values"][0] for chunk in chunks_data])
    chunks_data = [chunks_data[i] for i in k_sorted_indices]
    
    # Combine arrays
    combined = {}
    for key in keys:
        arrays = [chunk[key] for chunk in chunks_data]
        combined[key] = np.concatenate(arrays)
        LOGGER.info("Combined %s: %d total values", key, len(combined[key]))
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Combine spectrum chunks from HPC runs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing chunk_* subdirectories",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output file path (default: INPUT_DIR/spectrum.npz)",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    if not args.input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", args.input_dir)
        raise SystemExit(1)
    
    # Find all chunk files
    chunk_files = find_chunk_files(args.input_dir)
    if not chunk_files:
        LOGGER.error("No chunk files found in %s", args.input_dir)
        raise SystemExit(1)
    
    LOGGER.info("Found %d chunks to combine", len(chunk_files))
    
    # Combine chunks
    combined = combine_chunks(chunk_files)
    
    # Determine output path
    output_path = args.output_file
    if output_path is None:
        output_path = args.input_dir / "spectrum.npz"
    
    # Save combined result
    np.savez(output_path, **combined)
    LOGGER.info("Wrote combined spectrum to %s", output_path)
    LOGGER.info("Total k-values: %d", len(combined["k_values"]))


if __name__ == "__main__":
    main()
