# Legacy Code

This directory contains obsolete code that has been deprecated but preserved for reference.

## Deprecated Files

- **main2.py**: Old pipeline implementation with different architecture. Superseded by `main.py`.
- **combiner.py**: Old matrix combiner script. Superseded by `scripts/combine_spectra.py`.
- **spectrum.py**: Old spectrum computation script. Functionality integrated into `main.py`.
- **testcode_1.py**: Development test script, no longer needed.
- **testcode_2.py**: Development test script, no longer needed.
- **domain_builder_visualization.py**: Old visualization script for domain building.

## Current Pipeline

For the current, supported pipeline, see:
- `main.py` - Main pipeline entry point
- `utils/` - Core utility functions
- `scripts/combine_spectra.py` - Combine chunked HPC results

## Documentation

See `docs/algorithm.md` and the main `README.md` for current usage instructions.
